"""Mean ± std over seeds of the final runs in results/results.csv (latest row per method/dataset/seed).

Usage: python scripts/summarize.py [--csv results/results.csv] [--out results/summary.md]
"""
import argparse
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = ["val_ndcg@20", "test_recall@10", "test_recall@20", "test_ndcg@10", "test_ndcg@20"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results/results.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/summary.md"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    hp = df["hparams"].map(json.loads)
    df["stage"] = hp.map(lambda h: h.get("stage", "final"))
    df["reg"] = hp.map(lambda h: h.get("reg"))
    df["sec_per_epoch"] = hp.map(lambda h: h.get("sec_per_epoch"))
    df["device"] = hp.map(lambda h: h.get("device"))
    final = df[df["stage"] == "final"].groupby(["method", "dataset", "seed"]).tail(1)

    rows = []
    for (m, d), g in final.groupby(["method", "dataset"], sort=False):
        r = {"method": m, "dataset": d, "seeds": len(g), "reg": g["reg"].iloc[0],
             "best_epoch": f"{g['best_epoch'].mean():.0f}",
             "s/epoch": f"{g['sec_per_epoch'].mean():.3f} ({g['device'].iloc[0]})"}
        for k in METRICS:
            r[k] = f"{g[k].mean():.4f} ± {g[k].std(ddof=1):.4f}"
        rows.append(r)
    table = pd.DataFrame(rows)
    cols = list(table.columns)
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                   + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in table.iterrows()])
    tune = df[df["stage"] == "tune"][["method", "seed", "reg", "best_epoch", "val_ndcg@20"]]
    out = "## Final runs (mean ± std over seeds)\n\n" + md
    if len(tune):
        out += "\n\n## L2 tuning (seed 0, val NDCG@20)\n\n" + tune.to_string(index=False)
    print(out)
    with open(args.out, "w") as f:
        f.write(out + "\n")
