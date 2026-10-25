"""Test NDCG@20 per user group from per-user metric files (results/peruser/<dataset>/<run>/seed<k>.npz).

Groups (computed per seed from the split): social degree in the TRAINING social graph (0, 1-2, 3+) and number
of training interactions (bottom quartile vs rest). Averages over the seeds available for each run;
Δ columns are paired differences (same seed) to a reference run.
Usage: python scripts/subgroups.py --dataset synth --ref sg_lgcns_A [--runs a b c] [--out results/x.md]
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.splits import load_split  # noqa: E402


def groups(dataset, seed):
    sp = load_split(os.path.join(ROOT, "data/processed", dataset, f"seed_{seed}"))
    deg = np.bincount(sp.social_train.ravel(), minlength=sp.n_users)
    ninter = np.bincount(sp.train[:, 0], minlength=sp.n_users)
    q25 = np.quantile(ninter, 0.25)
    low = ninter <= q25
    if low.mean() > 0.5:  # degenerate (e.g. LastFM: almost every user has the capped count) -> strictly below
        low = ninter < q25
    return {"deg 0": deg == 0, "deg 1-2": (deg >= 1) & (deg <= 2), "deg 3+": deg >= 3,
            "few inter": low, "other inter": ~low, "all": np.ones(sp.n_users, bool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = os.path.join(ROOT, "results/peruser", a.dataset)
    runs = a.runs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "*")))
    G = {}
    data = {}
    for r in runs:
        for f in sorted(glob.glob(os.path.join(base, r, "seed*.npz"))):
            s = int(os.path.basename(f)[4:-4])
            z = np.load(f)
            G.setdefault(s, groups(a.dataset, s))
            full = np.full(len(G[s]["all"]), np.nan)
            full[z["users"]] = z["ndcg"]
            data[(r, s)] = full
    gnames = list(next(iter(G.values())).keys())
    seeds = sorted(G)
    sizes = {g: np.mean([int((G[s][g] & ~np.isnan(data[next(k for k in data if k[1] == s)])).sum()) for s in seeds]) for g in gnames}
    lines = [f"## {a.dataset}: test NDCG@20 per user group (mean over seeds; Δ = paired difference to `{a.ref}`)", "",
             "Group sizes (users with test items, mean over seeds): " + ", ".join(f"{g}: {sizes[g]:.0f}" for g in gnames), "",
             "| run | seeds | " + " | ".join(gnames) + " |", "|---|---|" + "---|" * len(gnames)]
    for r in runs:
        ss = [s for s in seeds if (r, s) in data]
        if not ss:
            continue
        cells = []
        for g in gnames:
            v = np.mean([np.nanmean(data[(r, s)][G[s][g]]) for s in ss])
            if r != a.ref and all((a.ref, s) in data for s in ss):
                d = np.mean([np.nanmean(data[(r, s)][G[s][g]]) - np.nanmean(data[(a.ref, s)][G[s][g]]) for s in ss])
                cells.append(f"{v:.4f} ({d:+.4f})")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {r} | {len(ss)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(os.path.join(ROOT, a.out), "w").write(text + "\n")


if __name__ == "__main__":
    main()
