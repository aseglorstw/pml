"""PU-GIB preprocessing per seed (paper sections 3.1-3.2). Uses only the training social graph.

Embedding variants for the interaction-only similarity s_ij:
  best : best-validation LightGCN checkpoint (results/checkpoints/lightgcn_lastfm_seed{k}.pt,
         produced by `run_experiment.py configs/lastfm_lightgcn.yaml`)
  ep20 : LightGCN trained on R_train for exactly 20 epochs (trained here, on CPU)
Both use the final layer-mean user embeddings (the ones LightGCN scores with).

Outputs in data/processed/lastfm/seed_k/pugib/:
  omega.npz            pairs (i<j), y, in_pval, r          (aligned arrays)
  graph_stats.json     |Omega|, p_A, bridges, ...
  pool_ra.txt          top-K RA candidate pool
  <variant>/embeddings.pt, s.npy, g.npy, isotonic.npz,
            pool_s.txt, pool_random.txt, support.txt, priors.json, report.json
Figures in results/figures/. Summary in results/pugib_prep_summary.json.

Usage: python scripts/prepare_pugib.py [--seeds 0 1 2] [--variants best ep20] [--K 10]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.pugib_prep import (adjacency, anchor_estimates, anchor_g_top, build_omega, channel_estimates, cosine_on_pairs,  # noqa: E402
                            degree_stats, fit_isotonic, isin_pairs, pool_stats, random_pool,
                            resource_allocation, topk_pool, uninformed_prior, union_edges)
from src.splits import load_split  # noqa: E402
from src.train import train_lightgcn  # noqa: E402

KAPPAS = [0.02, 0.05, 0.1]
DATASET = "lastfm"  # set from --dataset
MAX_OMEGA = 50e6  # set from --max-omega


def checkpoint_path(seed):
    return os.path.join(ROOT, "results/checkpoints", f"lightgcn_{DATASET}_seed{seed}.pt")


def get_embeddings(variant, split, seed):
    if variant == "best":
        ck = torch.load(checkpoint_path(seed), map_location="cpu")
        return ck["user_final"].numpy(), {"source": os.path.relpath(checkpoint_path(seed), ROOT),
                                          "best_epoch": ck["best_epoch"], "reg": ck["cfg"]["reg"],
                                          "val_ndcg@20": ck["val_ndcg@20"]}, ck
    with open(os.path.join(ROOT, f"configs/{DATASET}_lightgcn.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("tune", None)
    if os.path.exists(checkpoint_path(seed)):  # use the tuned L2 weight when known
        cfg["reg"] = torch.load(checkpoint_path(seed), map_location="cpu")["cfg"]["reg"]
    cfg.update({"max_epochs": 20, "eval_every": 10 ** 9, "device": "cpu"})
    res = train_lightgcn(split, cfg, seed, use_social=False, log=lambda *_: None)
    m = res["model"]
    with torch.no_grad():
        eu, ei = m.propagate()
    ck = {"cfg": cfg, "seed": seed, "best_epoch": 20, "val_ndcg@20": res["val_ndcg@20"],
          "ego": m.emb.weight.detach().cpu(), "user_final": eu.cpu(), "item_final": ei.cpu(),
          "n_users": m.n_users, "n_items": m.n_items}
    return eu.numpy(), {"source": "trained 20 epochs", "reg": cfg["reg"],
                        "val_ndcg@20": res["val_ndcg@20"], "test": res["test"]}, ck


def plot(path, title, iso, s, y, in_pval, p_a, g_val):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    xs = np.linspace(s.min(), s.max(), 400)
    ax[0].plot(xs, iso.predict(xs), lw=2, label="g(s) isotonic")
    # empirical positive rate in s-quantile bins
    bins = np.quantile(s, np.linspace(0, 1, 41))
    idx = np.clip(np.searchsorted(bins, s, side="right") - 1, 0, 39)
    ax[0].scatter([s[idx == b].mean() for b in range(40)], [y[idx == b].mean() for b in range(40)],
                  s=10, c="gray", label="binned rate")
    ax[0].axhline(p_a, ls="--", c="k", lw=1, label=f"p_A = {p_a:.4f}")
    ax[0].axhline(g_val, ls=":", c="C3", lw=1.5, label=f"g_val = {g_val:.4f}")
    ax[0].set(xlabel="s = cos(e_i, e_j)", ylabel="P(A_train = 1 | s)", title="g(s) on Omega")
    ax[0].legend(fontsize=8)
    # same curve against the quantile of s (cosines crowd near 1, this spreads them out)
    q = np.linspace(0, 1, 1001)
    ax[2].plot(q, iso.predict(np.quantile(s, q)), lw=2)
    ax[2].axhline(p_a, ls="--", c="k", lw=1)
    ax[2].axhline(g_val, ls=":", c="C3", lw=1.5)
    ax[2].set(xlabel="quantile of s on Omega", ylabel="g", ylim=(0, max(0.3, 2 * g_val)),
              title="g against quantile of s (y clipped)")
    kw = dict(bins=60, density=True, histtype="step", lw=1.5)
    ax[1].hist(s[y == 1], label=f"train positives in Omega ({(y == 1).sum()})", **kw)
    ax[1].hist(s[in_pval], label=f"P_val ({in_pval.sum()})", **kw)
    ax[1].hist(s[y == 0], label=f"U_rank ({(y == 0).sum()})", **kw)
    ax[1].set(xlabel="s", ylabel="density", title="similarity distributions")
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def process_seed(seed, variants, K, data_dir):
    split = load_split(os.path.join(data_dir, f"seed_{seed}"))
    n = split.n_users
    out = os.path.join(data_dir, f"seed_{seed}", "pugib")
    os.makedirs(out, exist_ok=True)

    deg = np.bincount(split.social_train.ravel(), minlength=n).astype(np.float64)
    est = float((deg ** 2).sum() / 2)
    print(f"seed {seed}: |Omega| estimate sum(deg^2)/2 = {est:,.0f} (~{est * 48 / 1e9:.2f} GB)", flush=True)
    if est > MAX_OMEGA:
        print(f"seed {seed}: |Omega| estimate exceeds {MAX_OMEGA:,.0f}; stopping", flush=True)
        sys.exit(3)
    A = adjacency(split.social_train, n)
    omega = build_omega(A)
    y = isin_pairs(omega, split.social_train, n)
    in_pval = isin_pairs(omega, split.social_val, n)
    assert not (y & in_pval).any()
    u_mask = ~y
    p_val = omega[in_pval]
    r = resource_allocation(A, omega)
    assert (r > 0).all()
    p_a = float(y.mean())
    gs = {"seed": seed, "n_users": n, "omega": int(len(omega)), "positives": int(y.sum()), "p_A": p_a,
          "u_rank": int(u_mask.sum()), "p_val": int(in_pval.sum()),
          "e_train": int(len(split.social_train)), "e_val": int(len(split.social_val)),
          "train_bridges": int(len(split.social_train) - y.sum()),
          "heldout_bridges": int(len(split.social_val) - in_pval.sum()),
          "base_rate": float(in_pval.sum() / u_mask.sum()),
          "train_social_degree": degree_stats(split.social_train, n),
          "ra_auroc_pval_vs_urank": float(roc_auc_score(in_pval[u_mask], r[u_mask]))}
    np.savez_compressed(os.path.join(out, "omega.npz"), pairs=omega.astype(np.int32), y=y,
                        in_pval=in_pval, r=r.astype(np.float32))
    with open(os.path.join(out, "graph_stats.json"), "w") as f:
        json.dump(gs, f, indent=2)

    u_pairs = omega[u_mask]
    pool_ra = topk_pool(u_pairs, r[u_mask], K, np.random.default_rng([seed, 2]))
    np.savetxt(os.path.join(out, "pool_ra.txt"), pool_ra, fmt="%d")
    report = {"graph": gs, "variants": {}}

    for variant in variants:
        if variant == "best" and not os.path.exists(checkpoint_path(seed)):
            print(f"seed {seed}: no LightGCN checkpoint yet, skipping variant 'best'")
            continue
        vdir = os.path.join(out, variant)
        os.makedirs(vdir, exist_ok=True)
        emb, emb_info, ck = get_embeddings(variant, split, seed)
        torch.save(ck, os.path.join(vdir, "embeddings.pt"))
        s = cosine_on_pairs(emb, omega)
        iso, g = fit_isotonic(s, y)
        assert np.isclose(g.mean(), p_a), (g.mean(), p_a)
        gv = g[in_pval]
        g_val, se = float(gv.mean()), float(gv.std(ddof=1) / np.sqrt(len(gv)))
        informative = bool(g_val - 1.645 * se > p_a)
        priors = {"p_A": p_a, "g_val": g_val, "g_val_se": se, "n_p_val": int(len(gv)),
                  "informative": informative, "mean_g_omega": float(g.mean()),
                  "estimated": {str(k): channel_estimates(p_a, g_val, k) for k in KAPPAS},
                  "uninformed": {str(k): uninformed_prior(p_a, k) for k in KAPPAS},
                  "embeddings": emb_info}
        anc = anchor_g_top(s, y, np.random.default_rng([seed, 4]))
        priors["anchor_g_top"] = anc
        priors["anchor"] = {str(k): anchor_estimates(p_a, g_val, anc["g_top"], k) for k in KAPPAS}

        pool_s = topk_pool(u_pairs, s[u_mask], K, np.random.default_rng([seed, 2]))
        pool_rand = random_pool(u_pairs, len(pool_s), np.random.default_rng([seed, 3]))
        pools = {"s": pool_s, "ra": pool_ra, "random": pool_rand}
        pstats = {}
        for name, pool in pools.items():
            st = pool_stats(pool, p_val, len(u_pairs), n, n_e_val=len(split.social_val))
            sup = union_edges(split.social_train, pool)
            st["support_size"] = int(len(sup))
            st["support_degree"] = degree_stats(sup, n)
            pstats[name] = st
        sims = {"auroc_s_pval_vs_urank": float(roc_auc_score(in_pval[u_mask], s[u_mask])),
                "auroc_s_train_vs_urank": float(roc_auc_score(y, s)),
                "mean_s": {"train_pos": float(s[y].mean()), "p_val": float(s[in_pval].mean()),
                           "u_rank": float(s[u_mask].mean())}}

        np.save(os.path.join(vdir, "s.npy"), s.astype(np.float32))
        np.save(os.path.join(vdir, "g.npy"), g.astype(np.float32))
        np.savez(os.path.join(vdir, "isotonic.npz"), x=iso.X_thresholds_, y=iso.y_thresholds_)
        np.savetxt(os.path.join(vdir, "pool_s.txt"), pool_s, fmt="%d")
        np.savetxt(os.path.join(vdir, "pool_random.txt"), pool_rand, fmt="%d")
        np.savetxt(os.path.join(vdir, "support.txt"), union_edges(split.social_train, pool_s), fmt="%d")
        with open(os.path.join(vdir, "priors.json"), "w") as f:
            json.dump(priors, f, indent=2)
        vrep = {"priors": priors, "pools": pstats, "similarity": sims, "K": K}
        with open(os.path.join(vdir, "report.json"), "w") as f:
            json.dump(vrep, f, indent=2)
        report["variants"][variant] = vrep
        os.makedirs(os.path.join(ROOT, "results/figures"), exist_ok=True)
        plot(os.path.join(ROOT, "results/figures", f"{DATASET}_seed{seed}_{variant}.png"),
             f"{DATASET} seed {seed}, embeddings: {variant}", iso, s, y, in_pval, p_a, g_val)
        print_variant(seed, variant, gs, vrep)
    return report


def print_variant(seed, variant, gs, v):
    p = v["priors"]
    print(f"\n=== seed {seed} | {variant} | |Omega|={gs['omega']} p_A={gs['p_A']:.5f} "
          f"bridges train/val={gs['train_bridges']}/{gs['heldout_bridges']}")
    print(f"g_val={p['g_val']:.5f} ± {p['g_val_se']:.5f}  informative={p['informative']}  "
          f"AUROC(s: P_val vs U_rank)={v['similarity']['auroc_s_pval_vs_urank']:.3f}")
    if "anchor_g_top" in p:
        a = p["anchor_g_top"]
        print(f"anchor: g_top={a['g_top']:.4f} folds=" + ", ".join(
            f"[s>={f['s_lo']:.3f} n_fit={f['block_size_fit']} rate_fit={f['block_rate_fit']:.3f} "
            f"n_eval={f['n_eval']} rate_eval={f['rate_eval']:.3f}]" for f in a["folds"]))
        for k in KAPPAS:
            e = p["anchor"][str(k)]
            print(f"  kappa={k}: anchor c_EN={e['c_hat_en']:.4f} c_anchor={e['c_hat']:.4f} pi={e['pi_hat']:.4f} "
                  f"rho_obs={e['rho_obs']:.4f} rho_un={e['rho_un']:.5f} valid={e['valid']}")
    for k in KAPPAS:
        e, u = p["estimated"][str(k)], p["uninformed"][str(k)]
        print(f"  kappa={k}: est c={e['c_hat']:.4f} pi={e['pi_hat']:.4f} rho_obs={e['rho_obs']:.4f} "
              f"rho_un={e['rho_un']:.5f} adm={e['admissible_lower'] and e['admissible_upper']} | "
              f"unif c={u['c_hat']:.4f} pi={u['pi_hat']:.4f} rho_obs={u['rho_obs']:.4f} rho_un={u['rho_un']:.5f}")
    for name, st in v["pools"].items():
        print(f"  pool {name:6s}: size={st['size']} recall={st['recall']:.4f} precision={st['precision']:.4f} "
              f"(base {st['base_rate']:.5f}, lift {st['lift']:.2f}) |C|={st['support_size']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["best", "ep20"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--dataset", default="lastfm")
    ap.add_argument("--max-omega", type=float, default=50e6, help="exit with code 3 if the estimate is larger")
    args = ap.parse_args()
    DATASET = args.dataset
    MAX_OMEGA = args.max_omega
    data_dir = os.path.join(ROOT, f"data/processed/{DATASET}")
    summ = os.path.join(ROOT, "results/pugib_prep_summary.json" if DATASET == "lastfm"
                        else f"results/pugib_prep_summary_{DATASET}.json")
    allrep = json.load(open(summ)) if os.path.exists(summ) else {}
    for s in args.seeds:  # merge, so running a subset of seeds/variants keeps the rest
        rep = process_seed(s, args.variants, args.K, data_dir)
        old = allrep.get(str(s), {"variants": {}})
        old["graph"] = rep["graph"]
        old["variants"].update(rep["variants"])
        allrep[str(s)] = old
    with open(summ, "w") as f:
        json.dump(allrep, f, indent=2)
