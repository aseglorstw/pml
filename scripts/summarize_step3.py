"""Step-3 report: test metrics (mean ± std over seeds), normalization choice, alpha diagnostics.

Usage: python scripts/summarize_step3.py  -> results/step3_summary.md, results/figures/step3_*.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = ["test_recall@10", "test_recall@20", "test_ndcg@10", "test_ndcg@20"]
ORDER = ["popularity", "lightgcn", "lightgcn_s", "lightgcn_warm", "lightgcn_s_warm",
         "lightgcn_pu_s", "lightgcn_pu_ra", "pugib_s_anchor", "pugib_s_en", "pugib_s_unif",
         "pugib_ra_anchor", "pugib_ra_en", "pugib_ra_unif", "pugib_s_grid"]
TUNED = ["beta", "kappa", "norm", "rho_obs", "rho_un", "reg"]


def load():
    df = pd.read_csv(os.path.join(ROOT, "results/results.csv"))
    hp = df["hparams"].map(json.loads)
    df["stage"] = hp.map(lambda h: h.get("stage", "final"))
    df["hp"] = hp
    return df


def main():
    df = load()
    final = df[(df["stage"] == "final") & (df["dataset"] == "lastfm")].groupby(["method", "seed"]).tail(1)
    agg = {}
    for m in ORDER:
        g = final[final["method"] == m]
        if len(g):
            agg[m] = {"n": len(g), **{k: (g[k].mean(), g[k].std(ddof=1)) for k in METRICS + ["val_ndcg@20"]},
                      "config": {k: g["hp"].iloc[0].get(k) for k in TUNED if k in g["hp"].iloc[0]},
                      "best_epoch": g["best_epoch"].mean()}
    lines = ["## Test metrics (mean ± std over seeds)", "",
             "| Method | seeds | R@10 | R@20 | N@10 | N@20 | best epoch | config |", "|---|---|---|---|---|---|---|---|"]
    ref = agg.get("lightgcn_s")
    for m, a in agg.items():
        cells = []
        for k in METRICS:
            mu, sd = a[k]
            flag = ""
            if ref and m != "lightgcn_s" and abs(mu - ref[k][0]) <= ref[k][1]:
                flag = " ≈"  # within one std of LightGCN-S
            cells.append(f"{mu:.4f} ± {sd:.4f}{flag}")
        lines.append(f"| {m} | {a['n']} | " + " | ".join(cells) + f" | {a['best_epoch']:.0f} | {a['config']} |")
    lines += ["", "≈ : mean within one std of LightGCN-S (cold start)."]

    # normalization choice (main variant tuning on seed 0)
    t = df[(df["method"] == "pugib_s_anchor") & (df["stage"] == "tune")].copy()
    if len(t):
        t["norm"] = t["hp"].map(lambda h: h["norm"])
        t["beta"] = t["hp"].map(lambda h: h["beta"])
        t["kappa"] = t["hp"].map(lambda h: h["kappa"])
        lines += ["", "## Main variant (s pool, anchor prior) tuning on seed 0: val NDCG@20", "",
                  t.pivot_table(index=["beta", "kappa"], columns="norm", values="val_ndcg@20").round(4).to_string()]
        bn = t.groupby("norm")["val_ndcg@20"].max()
        lines += ["", "best per normalization: " + ", ".join(f"{k}={v:.4f}" for k, v in bn.items())]

    # alpha diagnostics of final runs
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results/pugib_runs/*"))):
        name = os.path.basename(d)
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "final_seed*.json")))]
        runs = [r for r in runs if r.get("best_diag")]
        if not runs:
            continue
        keys = ["alpha_mean_obs", "alpha_mean_cand", "frac_obs_gt05", "frac_cand_gt05",
                "posterior_recall_pval", "alpha_mean_pval_in_pool", "kl_mean"]
        row = {"method": name, "rho_obs": runs[0]["extra"]["rho_obs"], "rho_un": runs[0]["extra"]["rho_un"]}
        for k in keys:
            v = [r["best_diag"][k] for r in runs]
            row[k] = f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
        rows.append(row)
        plot_run(name, runs)
    if rows:
        lines += ["", "## Alpha diagnostics at the best epoch (final runs, mean ± std over seeds)", "",
                  "rho values are those of seed 0.", "", pd.DataFrame(rows).to_string(index=False)]
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(ROOT, "results/step3_summary.md"), "w") as f:
        f.write(out + "\n")


def plot_run(name, runs):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    bins = np.array(runs[0]["alpha_hist_bins"])
    ctr = (bins[:-1] + bins[1:]) / 2
    w = bins[1] - bins[0]
    obs = np.sum([r["alpha_hist_obs"] for r in runs], 0)
    cand = np.sum([r["alpha_hist_cand"] for r in runs], 0)
    ax[0].bar(ctr, obs / max(obs.sum(), 1), width=w, alpha=0.6, label="observed")
    ax[0].bar(ctr, cand / max(cand.sum(), 1), width=w, alpha=0.6, label="candidates")
    ax[0].axvline(runs[0]["extra"]["rho_obs"], c="C0", ls="--")
    ax[0].axvline(runs[0]["extra"]["rho_un"], c="C1", ls="--")
    ax[0].set(title="alpha at best epoch (all seeds)", xlabel="alpha", ylabel="fraction")
    ax[0].legend(fontsize=8)
    for r in runs:
        h = pd.DataFrame(r["history"])
        ax[1].plot(h["epoch"], h["bpr"], label=f"seed {r['seed']}")
        if "kl" in h:
            ax[2].plot(h["epoch"], h["kl"], label=f"seed {r['seed']}")
    ax[1].set(title="BPR (train, per epoch)", xlabel="epoch", yscale="log")
    ax[2].set(title="KL (train, per epoch)", xlabel="epoch", yscale="log")
    ax[1].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "results/figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "results/figures", f"step3_{name}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
