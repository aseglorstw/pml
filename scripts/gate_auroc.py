"""Does the GBSR gate pick out injected ties? AUROC of the saved noise-free gate (original vs injected ties).

Reads results/pugib_runs/<run>/alpha_seed<k>.npz (written by run_experiment.py for gbsr runs) and the injected
ties data/processed/<dataset>/seed_<k>/truth/injected.txt. CPU, seconds. Writes results/gate_auroc.md.
Usage: python scripts/gate_auroc.py [--datasets epinions_n50 epinions_fof50] [--seeds 0 1 2]
"""
import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def key(e):
    e = np.sort(np.asarray(e, dtype=np.int64).reshape(-1, 2), axis=1)
    return e[:, 0] * (1 << 32) + e[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["epinions_n50", "epinions_fof50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs", nargs="+", default=["gbsr_A_w1"], help="run suffixes: real_<dataset>_<suffix>")
    a = ap.parse_args()
    lines = ["# GBSR gate vs injected ties (noise-free gate without edge bias, best-validation checkpoint)", "",
             "AUROC > 0.5 = the gate is lower on injected ties than on original ones.", "",
             "| run | seed | ties | injected | gate mean | gate < 0.5 | mean gate original / injected | AUROC |",
             "|---|---|---|---|---|---|---|---|"]
    for ds in a.datasets:
        for suf in a.runs:
            run = f"real_{ds}_{suf}"
            for s in a.seeds:
                f = os.path.join(ROOT, "results/pugib_runs", run, f"alpha_seed{s}.npz")
                t = os.path.join(ROOT, "data/processed", ds, f"seed_{s}", "truth", "injected.txt")
                if not (os.path.exists(f) and os.path.exists(t)):
                    lines.append(f"| {run} | {s} | missing {'gate' if not os.path.exists(f) else 'truth'} |||||| ")
                    continue
                z = np.load(f)
                g = z["alpha"].astype(float)
                inj = np.isin(key(z["edges"]), key(np.loadtxt(t, dtype=np.int64, ndmin=2)))
                auc = roc_auc_score(~inj, g) if 0 < inj.sum() < len(inj) else float("nan")
                lines.append(f"| {run} | {s} | {len(g)} | {int(inj.sum())} | {g.mean():.3f} | {(g < 0.5).mean():.3f} | "
                             f"{g[~inj].mean():.3f} / {g[inj].mean():.3f} | {auc:.3f} |")
    out = os.path.join(ROOT, "results/gate_auroc.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
