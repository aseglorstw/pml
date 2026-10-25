"""Training, early stopping, evaluation and results logging shared by all methods."""
from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch

from .metrics import evaluate
from .models.lightgcn import LightGCN, Popularity
from .splits import Split, load_split

RESULT_COLUMNS = ["timestamp", "method", "dataset", "seed", "hparams", "best_epoch",
                  "val_ndcg@20", "test_recall@10", "test_recall@20", "test_ndcg@10",
                  "test_ndcg@20", "runtime_s"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str | None) -> str:
    """'auto' -> cuda, else mps, else cpu. An unavailable cuda/mps request falls back to cpu."""
    name = (name or "auto").lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if name == "mps" and not has_mps:
        return "cpu"
    return name


def sync_device(device: str):
    """Wait for queued GPU work (for honest timings)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class NegativeSampler:
    """Uniform negatives not in the user's training items (vectorized rejection sampling)."""

    def __init__(self, train: np.ndarray, n_users: int, n_items: int, rng: np.random.Generator):
        self.n_items = n_items
        self.rng = rng
        self.R = sp.csr_matrix((np.ones(len(train), bool), (train[:, 0], train[:, 1])),
                               shape=(n_users, n_items))

    def sample(self, users: np.ndarray) -> np.ndarray:
        neg = self.rng.integers(0, self.n_items, len(users))
        bad = np.asarray(self.R[users, neg]).ravel()
        while bad.any():
            idx = np.where(bad)[0]
            neg[idx] = self.rng.integers(0, self.n_items, len(idx))
            bad[idx] = np.asarray(self.R[users[idx], neg[idx]]).ravel()
        return neg


def evaluate_split(score_fn, split: Split, which: str, device="cpu") -> dict:
    """val: exclude train. test: exclude train and val."""
    if which == "val":
        return evaluate(score_fn, split.n_users, split.n_items, split.val, [split.train], device=device)
    return evaluate(score_fn, split.n_users, split.n_items, split.test, [split.train, split.val],
                    device=device)


def warm_start_path(root: str, cfg: dict, seed: int) -> str:
    """Best-validation LightGCN checkpoint of this dataset and seed."""
    return os.path.join(root, "results/checkpoints", f"lightgcn_{cfg['dataset']}_seed{seed}.pt")


def build_model(split: Split, cfg: dict, seed: int, root: str, device: str):
    """Construct the model for cfg['method'] (call after set_seed for reproducible init)."""
    method = cfg["method"]
    common = dict(dim=cfg["dim"], n_layers=cfg["n_layers"], omega_s=cfg.get("omega_s", 0.5),
                  device=device)
    extra = {}
    social_train = split.social_train
    if cfg.get("social_edges_file"):  # e.g. a corrupted training social graph
        social_train = np.loadtxt(os.path.join(root, cfg["social_edges_file"]), dtype=np.int64, ndmin=2)
        extra["social_edges_file"] = cfg["social_edges_file"]
    if method in ("lightgcn", "lightgcn_s"):
        model = LightGCN(split.n_users, split.n_items, split.train,
                         social_edges=social_train if method == "lightgcn_s" else None, **common)
    elif method == "gbsr":
        from .models.gbsr import GBSR
        model = GBSR(split.n_users, split.n_items, split.train, social_train, beta=cfg["beta"],
                     sigma=cfg["sigma"], edge_bias=cfg.get("edge_bias", 0.5), tau=cfg.get("gbsr_tau", 0.2),
                     detach_mask=cfg.get("detach_mask", False), batch_size=cfg["batch_size"], **common)
    elif method in ("lightgcn_pu", "pugib"):
        from .models.pugib import PUGIB, pair_features
        from .pugib_data import load_pugib_inputs
        inp = load_pugib_inputs(split, os.path.join(root, cfg["data_dir"], f"seed_{seed}"), cfg)
        extra = {"n_pval_total": inp["n_pval_total"], "rho_obs": inp["rho_obs"], "rho_un": inp["rho_un"],
                 "support_size": int(len(inp["support"]))}
        if method == "lightgcn_pu":  # support C with all weights 1, static degree
            model = LightGCN(split.n_users, split.n_items, split.train, social_edges=inp["support"], **common)
        else:
            from .pugib_prep import cosine_on_pairs
            feats = pair_features(inp["e_tilde"], inp["support"], inp["r"], inp["is_obs"])
            tau_epochs = cfg.get("tau_epochs", cfg["max_epochs"])
            if tau_epochs in (None, "auto"):  # anneal over the whole training budget
                tau_epochs = cfg["max_epochs"]
            model = PUGIB(split.n_users, split.n_items, split.train, inp["support"], inp["is_obs"],
                          inp["rho"], feats, inp["ego"], deg_mode=cfg.get("norm", "static"),
                          beta=cfg["beta"], hidden=cfg.get("masker_hidden", 64),
                          tau_start=cfg.get("tau_start", 1.0), tau_end=cfg.get("tau_end", 0.2),
                          tau_epochs=int(tau_epochs), in_pval=inp["in_pval"],
                          pair_s=cosine_on_pairs(inp["e_tilde"], inp["support"]), pair_r=inp["r"],
                          **common)
    else:
        raise ValueError(method)
    # Warm start from the best LightGCN checkpoint is the default for every social method;
    # plain LightGCN defaults to random init. PU-GIB always starts from ẽ (set in its constructor).
    init = cfg.get("init", "random" if method == "lightgcn" else "e_tilde")
    extra["init"] = init
    if init == "e_tilde" and method != "pugib":
        ck = torch.load(warm_start_path(root, cfg, seed), map_location="cpu")
        with torch.no_grad():
            model.emb.weight.copy_(ck["ego"])
    return model.to(device), extra


def fit(split: Split, cfg: dict, seed: int, root: str, log=print) -> dict:
    """Generic BPR training with early stopping on val NDCG@20.

    The model must provide train_step_loss(users, pos, neg, reg) -> (loss, {name: scalar}) and
    score_fn(); optional start_epoch(epoch), eval_diagnostics(n_pval_total), best_snapshot().
    """
    device = resolve_device(cfg.get("device", "cpu"))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model, extra = build_model(split, cfg, seed, root, device)
    # freeze_backbone: embeddings stay at their (warm-start) values, only the masker trains.
    # lr_emb: separate learning rate for the embeddings; everything else uses lr.
    if cfg.get("freeze_backbone"):
        model.emb.weight.requires_grad_(False)
    # Default schedule for warm-started models: small lr for the (already trained) embeddings,
    # cfg['lr'] for everything else (masker). Set lr_emb explicitly (e.g. equal to lr) to override.
    if "lr_emb" not in cfg and extra.get("init") == "e_tilde":
        cfg = {**cfg, "lr_emb": 1e-4}
    emb_ids = {id(model.emb.weight)}
    emb_params = [p for p in model.parameters() if id(p) in emb_ids and p.requires_grad]
    other = [p for p in model.parameters() if id(p) not in emb_ids and p.requires_grad]
    groups = ([{"params": emb_params, "lr": cfg.get("lr_emb") or cfg["lr"]}] if emb_params else []) + \
             ([{"params": other, "lr": cfg["lr"]}] if other else [])
    extra.update({"freeze_backbone": bool(cfg.get("freeze_backbone")), "lr_emb": cfg.get("lr_emb"),
                  "n_trainable": int(sum(p.numel() for g in groups for p in g["params"]))})
    opt = torch.optim.Adam(groups) if groups else None
    sampler = NegativeSampler(split.train, split.n_users, split.n_items, rng)
    bs, reg = cfg["batch_size"], cfg["reg"]
    best = {"val_ndcg@20": -1.0, "epoch": 0, "state": None, "diag": None, "snapshot": None}
    bad_evals, epoch_times, history = 0, [], []
    use_social = cfg["method"] == "lightgcn_s"
    diag_init = model.social_diagnostic() if use_social else None
    has_diag = hasattr(model, "eval_diagnostics")

    epoch = 0
    for epoch in range(1, (cfg["max_epochs"] if opt is not None else 0) + 1):  # nothing to train -> skip
        model.train()
        if hasattr(model, "start_epoch"):
            model.start_epoch(epoch)
        t0 = time.perf_counter()
        perm = rng.permutation(len(split.train))
        sums = {}
        for start in range(0, len(perm), bs):
            b = split.train[perm[start:start + bs]]
            users, pos = b[:, 0], b[:, 1]
            neg = sampler.sample(users)
            loss, parts = model.train_step_loss(torch.as_tensor(users, device=device),
                                                torch.as_tensor(pos, device=device),
                                                torch.as_tensor(neg, device=device), reg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(b)
        sync_device(device)
        epoch_times.append(time.perf_counter() - t0)
        avg = {k: v / len(perm) for k, v in sums.items()}

        if epoch % cfg["eval_every"] == 0:
            model.eval()
            v = evaluate_split(model.score_fn(), split, "val", device)
            diag = model.eval_diagnostics(extra.get("n_pval_total")) if has_diag else {}
            history.append({"epoch": epoch, **avg, "val_recall@20": v["recall@20"],
                            "val_ndcg@20": v["ndcg@20"], **diag})
            improved = v["ndcg@20"] > best["val_ndcg@20"]
            if improved:
                best = {"val_ndcg@20": v["ndcg@20"], "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict()), "diag": diag,
                        "snapshot": model.best_snapshot() if hasattr(model, "best_snapshot") else None}
                bad_evals = 0
            else:
                bad_evals += 1
            msg = " ".join(f"{k} {x:.4f}" for k, x in avg.items())
            if diag:
                msg += " | " + " ".join(f"{k} {x:.3f}" for k, x in diag.items())
            log(f"  epoch {epoch:3d} {msg} val R@20 {v['recall@20']:.4f} "
                f"N@20 {v['ndcg@20']:.4f}{' *' if improved else ''}")
            if bad_evals >= cfg["patience"] and epoch >= cfg.get("min_epochs", 0):
                break

    model.eval()
    if cfg.get("eval_final") and best["state"] is not None:  # metrics of the LAST epoch, for comparison
        extra["final_epoch"] = epoch
        extra["final_test"] = evaluate_split(model.score_fn(), split, "test", device)
        extra["final_val_ndcg@20"] = evaluate_split(model.score_fn(), split, "val", device)["ndcg@20"]
        if has_diag:
            extra["final_diag"] = model.eval_diagnostics(extra.get("n_pval_total"))
    if best["state"] is None:  # no validation happened (fixed-epoch training): keep last state
        v = evaluate_split(model.score_fn(), split, "val", device)
        best.update({"val_ndcg@20": v["ndcg@20"], "epoch": epoch})
    else:
        model.load_state_dict(best["state"])
    test = evaluate_split(model.score_fn(), split, "test", device)
    res = {"best_epoch": best["epoch"], "val_ndcg@20": best["val_ndcg@20"], "test": test,
           "epochs_run": epoch, "sec_per_epoch": float(np.mean(epoch_times)) if epoch_times else 0.0, "device": device,
           "model": model, "history": history, "best_diag": best["diag"],
           "best_snapshot": best["snapshot"], "extra": extra}
    if use_social:
        res["social_diagnostic"] = {"init": diag_init, "best": model.social_diagnostic()}
    return res


def train_lightgcn(split: Split, cfg: dict, seed: int, use_social: bool, log=print, root: str = ".") -> dict:
    """Backward-compatible wrapper (LightGCN / LightGCN-S)."""
    return fit(split, {**cfg, "method": "lightgcn_s" if use_social else "lightgcn"}, seed, root, log)


def run_popularity(split: Split, cfg: dict, seed: int, log=print) -> dict:
    m = Popularity(split.train, split.n_items)
    v = evaluate_split(m.score_fn(), split, "val")
    t = evaluate_split(m.score_fn(), split, "test")
    return {"best_epoch": 0, "val_ndcg@20": v["ndcg@20"], "test": t, "epochs_run": 0,
            "sec_per_epoch": 0.0, "device": "cpu"}


def train_one(cfg: dict, seed: int, root: str, log=print) -> dict:
    """Train one (method, dataset, seed) configuration. Does not write results."""
    split = load_split(os.path.join(root, cfg["data_dir"], f"seed_{seed}"))
    t0 = time.perf_counter()
    method = cfg["method"]
    if method == "popularity":
        res = run_popularity(split, cfg, seed, log)
    elif method in ("lightgcn", "lightgcn_s", "lightgcn_pu", "pugib", "gbsr"):
        res = fit(split, cfg, seed, root, log)
    else:
        raise ValueError(method)
    res["runtime_s"] = time.perf_counter() - t0
    return res


def run_one(cfg: dict, seed: int, root: str, log=print, extra: dict | None = None) -> dict:
    """Train and append a row to results.csv."""
    res = train_one(cfg, seed, root, log)
    log_result(cfg, seed, res, root, extra)
    return res


def save_checkpoint(cfg: dict, seed: int, res: dict, root: str) -> str:
    """Save ego embeddings and final (layer-mean) user/item embeddings of the best checkpoint."""
    model = res["model"]
    path = os.path.join(root, "results/checkpoints", f"{cfg['method']}_{cfg['dataset']}_seed{seed}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        eu, ei = model.propagate()
    torch.save({"cfg": {k: v for k, v in cfg.items() if k != "tune"}, "seed": seed,
                "best_epoch": res["best_epoch"], "val_ndcg@20": res["val_ndcg@20"],
                "ego": model.emb.weight.detach().cpu(), "user_final": eu.cpu(), "item_final": ei.cpu(),
                "n_users": model.n_users, "n_items": model.n_items}, path)
    return path


def log_result(cfg: dict, seed: int, res: dict, root: str, extra: dict | None = None):
    method = cfg.get("name", cfg["method"])
    hp = {k: v for k, v in cfg.items() if k not in ("name", "dataset", "data_dir", "seeds",
                                                    "results_csv", "tune", "save_checkpoint")}
    hp.update({"device": res["device"], "sec_per_epoch": round(res["sec_per_epoch"], 4),
               "epochs_run": res["epochs_run"], **res.get("extra", {}), **(extra or {})})
    append_result(os.path.join(root, cfg.get("results_csv", "results/results.csv")), {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method, "dataset": cfg["dataset"], "seed": seed,
        "hparams": json.dumps(hp, sort_keys=True), "best_epoch": res["best_epoch"],
        "val_ndcg@20": res["val_ndcg@20"],
        **{f"test_{k}": res["test"][k] for k in ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]},
        "runtime_s": round(res["runtime_s"], 2)})
    return res


def append_result(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
