"""How well can ties be selected at all? Errors of the MAP graph as predicted by the posterior itself.

If rho is calibrated, the MAP graph (keep rho >= 0.5) is expected to lose
    sum_{removed} rho / sum_all rho            of the genuine ties, and to keep
    sum_{kept} (1 - rho) / #kept               false ties among the kept ones.
Where the truth is known (synthetic graphs, truth/Z.txt) the actual values are printed next to the prediction.
Also: maximum rho, share of ties with rho >= 0.9, and users who have ties before but none after pruning.
Only reads data/processed/<ds>/seed_<k>/pugib/mixture_prior.npz (CPU, seconds).
Writes results/posterior_diagnostics.md (and prints it).

Usage: python scripts/posterior_diagnostics.py [--datasets synth synth50 douban_book epinions yelp] [--seeds 0 1 2]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import isin_pairs  # noqa: E402


def row(ds, s):
    d = os.path.join(ROOT, "data/processed", ds, f"seed_{s}")
    f = os.path.join(d, "pugib/mixture_prior.npz")
    if not os.path.exists(f):
        return None
    mp = np.load(f)
    e, rho = mp["edges"], mp["rho"]
    keep = rho >= 0.5
    users_before = np.unique(e)
    users_after = np.unique(e[keep])
    r = {"ds": ds, "seed": s, "ties": len(e), "kept": int(keep.sum()), "max_rho": rho.max(),
         "rho_ge_09": (rho >= 0.9).mean(),
         "pred_lost": rho[~keep].sum() / max(rho.sum(), 1e-12),
         "pred_false_kept": (1 - rho[keep]).sum() / max(keep.sum(), 1),
         "users_with_ties": len(users_before), "users_isolated": len(users_before) - len(users_after)}
    tz = os.path.join(d, "truth/Z.txt")
    if os.path.exists(tz):
        n = int(max(e.max(), np.loadtxt(tz, dtype=np.int64).max()) + 1)
        true = isin_pairs(e, np.loadtxt(tz, dtype=np.int64), n)
        r["true_lost"] = (true & ~keep).sum() / true.sum()
        r["true_false_kept"] = (~true[keep]).mean() if keep.any() else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "synth50", "douban_book", "epinions", "yelp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    lines = ["# Posterior diagnostics: errors of the MAP graph predicted by rho (truth where known)", "",
             "| dataset | seed | ties | kept | max ρ | ρ ≥ 0.9 | genuine lost: pred / true | "
             "false among kept: pred / true | users with ties → isolated |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds in a.datasets:
        for s in a.seeds:
            r = row(ds, s)
            if r is None:
                continue
            tl = f" / {r['true_lost']:.3f}" if "true_lost" in r else ""
            tf = f" / {r['true_false_kept']:.3f}" if "true_false_kept" in r else ""
            lines.append(f"| {ds} | {s} | {r['ties']} | {r['kept']} ({r['kept'] / r['ties']:.0%}) | {r['max_rho']:.3f} | "
                         f"{r['rho_ge_09']:.1%} | {r['pred_lost']:.3f}{tl} | {r['pred_false_kept']:.3f}{tf} | "
                         f"{r['users_with_ties']} → {r['users_isolated']} |")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, "results/posterior_diagnostics.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
