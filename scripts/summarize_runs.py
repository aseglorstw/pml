"""One table for a group of runs (by name prefix): best/last-epoch test metrics, epoch time and
alpha diagnostics at the best epoch (mean/std on observed edges and candidates, fraction of
candidates with alpha > 0.5, Spearman correlation of alpha with s and RA). GBSR rows show its
noise-free gate (without the +0.5 bias) in the 'observed' columns.

Reads results/jobs/*.csv and results/results.csv.
Usage: python scripts/summarize_runs.py --prefix db3_ [--out results/douban_stage3_summary.md]
"""
import argparse
import glob
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
files = glob.glob(os.path.join(ROOT, "results/jobs/*.csv")) + [os.path.join(ROOT, "results/results.csv")]
df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True).drop_duplicates()
df = df[df["method"].str.startswith(a.prefix)]
df = df[df["hparams"].map(lambda h: json.loads(h).get("stage", "final") == "final")]
df = df.sort_values("timestamp").groupby(["method", "dataset", "seed"]).tail(1).sort_values("method")


def f(v, n=4):
    return "–" if v is None or (isinstance(v, float) and v != v) else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def pm(d, m, s):
    return "–" if not d or d.get(m) is None else f"{d[m]:.3f} ± {d.get(s, float('nan')):.3f}"


rows = []
for _, r in df.iterrows():
    h = json.loads(r["hparams"])
    dfile = os.path.join(ROOT, "results/pugib_runs", r["method"], f"final_seed{r['seed']}_default.json")
    d = json.load(open(dfile)).get("best_diag") if os.path.exists(dfile) else None
    if d and "gate_mean" in d:  # GBSR
        d = {"alpha_mean_obs": d["gate_mean"], "alpha_std_obs": d["gate_std"]}
    rows.append([r["method"], r["dataset"], str(r["seed"]), str(r["best_epoch"]), f(r["val_ndcg@20"]),
                 f(r["test_recall@10"]), f(r["test_recall@20"]), f(r["test_ndcg@10"]), f(r["test_ndcg@20"]),
                 f((h.get("final_test") or {}).get("ndcg@20")), f(h.get("sec_per_epoch"), 2),
                 pm(d, "alpha_mean_obs", "alpha_std_obs"), pm(d, "alpha_mean_cand", "alpha_std_cand"),
                 f((d or {}).get("frac_cand_gt05"), 3),
                 f((d or {}).get("corr_alpha_s_obs"), 2) + " / " + f((d or {}).get("corr_alpha_s_cand"), 2),
                 f((d or {}).get("corr_alpha_ra_obs"), 2) + " / " + f((d or {}).get("corr_alpha_ra_cand"), 2)])
cols = ["run", "dataset", "seed", "best ep", "val N@20", "R@10", "R@20", "N@10", "N@20", "last-ep N@20", "s/epoch",
        "α obs", "α cand", "cand α>0.5", "ρ(α,s) obs/cand", "ρ(α,RA) obs/cand"]
out = "\n".join([f"## Runs '{a.prefix}*' (test metrics at the best val checkpoint)", "",
                 "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])
bench = os.path.join(ROOT, "results/douban_benchmark_cuda.txt")
if a.prefix.startswith("db") and os.path.exists(bench):
    out += "\n\nGPU benchmark (Douban-Book):\n\n```\n" + open(bench).read().strip() + "\n```"
print(out)
if a.out:
    open(os.path.join(ROOT, a.out), "w").write(out + "\n")
