"""PU-GIB preprocessing numbers per seed (|Omega|, p_A, g_val, AUROC of s, g_top, c_hat, priors, pool lift).

Usage: python scripts/summarize_prep.py --dataset douban_book  -> results/<dataset>_prep_summary.md
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from overnight_summary import fmt, prep_numbers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="douban_book")
a = ap.parse_args()
path = os.path.join(ROOT, "results/pugib_prep_summary.json" if a.dataset == "lastfm"
                    else f"results/pugib_prep_summary_{a.dataset}.json")
rep = json.load(open(path))
seeds = sorted(rep, key=int)
cols = {s: prep_numbers(rep, s) for s in seeds}
cols = {s: c for s, c in cols.items() if c}
keys = list(next(iter(cols.values())))
lines = [f"## PU-GIB preprocessing, {a.dataset}, best-checkpoint embeddings", "",
         "| quantity | " + " | ".join(f"seed {s}" for s in cols) + " |", "|---|" + "---|" * len(cols)]
lines += [f"| {k} | " + " | ".join(fmt(cols[s].get(k)) for s in cols) + " |" for k in keys]
out = "\n".join(lines)
print(out)
open(os.path.join(ROOT, f"results/{a.dataset}_prep_summary.md"), "w").write(out + "\n")
