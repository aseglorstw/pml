"""Step-5 table: best-checkpoint and last-epoch metrics plus alpha diagnostics for runs named s5_*.

Reads per-job CSVs (results/jobs/*.csv) and results/results.csv; writes results/step5_summary.md.
Usage: python scripts/summarize_step5.py [--prefix s5_]
"""
import argparse
import glob
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="s5_")
a = ap.parse_args()
files = glob.glob(os.path.join(ROOT, "results/jobs/*.csv")) + [os.path.join(ROOT, "results/results.csv")]
df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True).drop_duplicates()
df = df[df["method"].str.startswith(a.prefix)].sort_values("timestamp").groupby("method").tail(1)

rows, arows = [], []
for _, r in df.iterrows():
    h = json.loads(r["hparams"])
    ft = h.get("final_test") or {}
    rows.append({"run": r["method"], "best_ep": r["best_epoch"], "val N@20": r["val_ndcg@20"],
                 "R@10": r["test_recall@10"], "R@20": r["test_recall@20"], "N@10": r["test_ndcg@10"],
                 "N@20": r["test_ndcg@20"], "last-epoch N@20": ft.get("ndcg@20"),
                 "last-epoch val N@20": h.get("final_val_ndcg@20"), "s/ep": h.get("sec_per_epoch")})
    diag_file = os.path.join(ROOT, "results/pugib_runs", r["method"], "final_seed0_default.json")
    bd = json.load(open(diag_file)).get("best_diag") if os.path.exists(diag_file) else None
    fd = h.get("final_diag")
    for when, d in (("best", bd), ("last", fd)):
        if d and "alpha_mean_obs" in d:
            arows.append({"run": r["method"], "at": when, **{k: d.get(k) for k in [
                "alpha_mean_obs", "alpha_std_obs", "alpha_mean_cand", "alpha_std_cand", "frac_cand_gt05",
                "corr_alpha_s_obs", "corr_alpha_s_cand", "corr_alpha_ra_obs", "corr_alpha_ra_cand"]}})


def md(rows):
    if not rows:
        return "none"
    cols = list(rows[0])
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows])


out = "\n".join(["## Step 5: test metrics (best val checkpoint) and last epoch", "", md(rows), "",
                 "## Alpha diagnostics (Spearman correlations with s and RA)", "", md(arows)])
print(out)
open(os.path.join(ROOT, "results/step5_summary.md"), "w").write(out + "\n")
