"""Run one experiment config.

If the config has a `tune` block {param: [values...], ...}, every combination (Cartesian
product) is trained on the tuning seed (`tune.seed`, default 0), the best by val NDCG@20 is
kept, and then every seed in `seeds` is trained with it. The tuning run of the best
combination is reused as the final run of the tuning seed (same config and seed).
All runs append a row to results/results.csv ("stage": "tune" / "final" in hparams); the
`method` column is cfg['name'] if given, else cfg['method'].

With `save_checkpoint: true`, final runs save embeddings to results/checkpoints/.
LightGCN-S runs write the social/interaction norm diagnostic to results/diagnostics/.
PU-GIB / LightGCN-PU runs write training histories to results/pugib_runs/<name>/ and, for
final runs, alpha at the best epoch.

Usage:
  python scripts/run_experiment.py configs/lastfm_lightgcn.yaml [--device cpu]
  python scripts/run_experiment.py configs/pugib/x.yaml --set norm=prior --seeds 0
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.train import log_result, save_checkpoint, train_one  # noqa: E402


def report(cfg, s, res):
    t = res["test"]
    print(f"[run] {cfg.get('name', cfg['method'])} seed={s} best_epoch={res['best_epoch']} "
          f"val N@20={res['val_ndcg@20']:.4f} test R@10={t['recall@10']:.4f} R@20={t['recall@20']:.4f} "
          f"N@10={t['ndcg@10']:.4f} N@20={t['ndcg@20']:.4f} "
          f"({res['sec_per_epoch']:.3f}s/epoch on {res['device']}, {res['runtime_s']:.0f}s)", flush=True)


def tag_of(combo: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in combo.items()) or "default"


def write_run_diag(cfg, s, res, stage, tag):
    if not res.get("history") or cfg["method"] not in ("pugib", "lightgcn_pu", "gbsr"):
        return
    d = os.path.join(ROOT, "results/pugib_runs", cfg.get("name", cfg["method"]))
    os.makedirs(d, exist_ok=True)
    out = {"seed": s, "stage": stage, "tag": tag, "best_epoch": res["best_epoch"],
           "val_ndcg@20": res["val_ndcg@20"], "test": res["test"], "best_diag": res.get("best_diag"),
           "extra": res.get("extra"), "history": res["history"],
           "cfg": {k: v for k, v in cfg.items() if k != "tune"}}
    snap = res.get("best_snapshot")
    if snap is not None:
        a = snap["alpha"]
        bins = np.linspace(0, 1, 21)
        out["alpha_hist_bins"] = bins.tolist()
        out["alpha_hist_obs"] = np.histogram(a[snap["is_obs"]], bins)[0].tolist()
        out["alpha_hist_cand"] = np.histogram(a[~snap["is_obs"]], bins)[0].tolist()
        if stage == "final":
            np.savez_compressed(os.path.join(d, f"alpha_seed{s}.npz"), **snap)
    with open(os.path.join(d, f"{stage}_seed{s}_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


def save_peruser(cfg, s, res):
    pu = res.get("peruser")
    if pu is None:
        return
    d = os.path.join(ROOT, "results/peruser", cfg.get("dataset", "x"), cfg.get("name", cfg["method"]))
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"seed{s}.npz"), **pu)


def save_state(cfg, s, res):
    """save_state: true -> full model state_dict of the best checkpoint (rebuild with build_model, then load)."""
    if not cfg.get("save_state") or "model" not in res:
        return
    import torch
    d = os.path.join(ROOT, "results/checkpoints/state")
    os.makedirs(d, exist_ok=True)
    torch.save({"state": res["model"].state_dict(), "cfg": {k: v for k, v in cfg.items() if k != "tune"}, "seed": s},
               os.path.join(d, f"{cfg.get('name', cfg['method'])}_seed{s}.pt"))


def finalize(cfg, s, res, tag):
    save_state(cfg, s, res)
    log_result(cfg, s, res, ROOT, extra={"stage": "final"})
    save_peruser(cfg, s, res)
    write_run_diag(cfg, s, res, "final", tag)
    if cfg.get("save_checkpoint") and "model" in res:
        print(f"[ckpt] {save_checkpoint(cfg, s, res, ROOT)}", flush=True)
    if "social_diagnostic" in res:
        path = os.path.join(ROOT, "results/diagnostics", f"{cfg['method']}_{cfg['dataset']}_seed{s}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(res["social_diagnostic"], f, indent=2)
        for d in res["social_diagnostic"]["best"]:
            print(f"[diag] layer {d['layer']}: |interaction| {d['interaction_norm']:.4f} "
                  f"|social| {d['social_norm']:.4f} ratio {d['ratio_social_over_interaction']:.3f}", flush=True)
    if res.get("best_diag"):
        print("[alpha] " + " ".join(f"{k}={v:.4f}" for k, v in res["best_diag"].items()), flush=True)
    report(cfg, s, res)


def parse_value(v: str):
    return yaml.safe_load(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default=None, help="override config device (auto/cuda/mps/cpu)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[], action="extend",
                    help="key=value overrides (applied before tuning); may be repeated")
    ap.add_argument("--quiet", action="store_true", help="do not print per-evaluation lines")
    ap.add_argument("--tune-only", action="store_true", help="only run the tuning grid")
    ap.add_argument("--no-tune", action="store_true", help="ignore the tune block")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
        if cfg.get("tune") and k in cfg["tune"]:
            del cfg["tune"][k]  # a fixed override removes that dimension from the grid
    if args.device:
        cfg["device"] = args.device
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    cfg["config_file"] = os.path.relpath(os.path.abspath(args.config), ROOT)
    log = (lambda *_: None) if args.quiet else (lambda m: print(m, flush=True))

    tune = cfg.pop("tune", None)
    if args.no_tune:
        tune = None
    reuse, best_tag = {}, tag_of({})
    if tune:
        tseed = tune.pop("seed", 0)
        keys = list(tune)
        best_val, best_combo, best_res = -1.0, None, None
        for values in itertools.product(*[tune[k] for k in keys]):
            combo = dict(zip(keys, values))
            c = {**cfg, **combo}
            print(f"[tune] {cfg.get('name', cfg['method'])} {combo} seed={tseed}", flush=True)
            res = train_one(c, tseed, ROOT, log)
            log_result(c, tseed, res, ROOT, extra={"stage": "tune"})
            write_run_diag(c, tseed, res, "tune", tag_of(combo))
            print(f"[tune] {combo}: val NDCG@20={res['val_ndcg@20']:.4f} best_epoch={res['best_epoch']} "
                  f"({res['runtime_s']:.0f}s)", flush=True)
            if res["val_ndcg@20"] > best_val:
                best_val, best_combo, best_res = res["val_ndcg@20"], combo, res
            else:
                del res
        print(f"[tune] best {best_combo} val NDCG@20={best_val:.4f}", flush=True)
        cfg.update(best_combo)
        best_tag = tag_of(best_combo)
        reuse[tseed] = best_res
    if args.tune_only:
        seeds = []

    for s in seeds:
        print(f"[run] {cfg.get('name', cfg['method'])} seed={s}"
              + (" (reusing tuning run)" if s in reuse else ""), flush=True)
        res = reuse.pop(s) if s in reuse else train_one(cfg, s, ROOT, log)
        finalize(cfg, s, res, best_tag)
