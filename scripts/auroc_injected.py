"""AUROC of 1 - alpha for separating injected social edges from genuine ones (observed edges only).

Usage: python scripts/auroc_injected.py --dataset lastfm_inject40 --seed 0 --run pugib_inj_anchor gbsr_inj
Appends JSON lines to results/diagnostics/auroc_injected.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import isin_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--run", nargs="+", required=True)
a = ap.parse_args()
inj = np.loadtxt(os.path.join(ROOT, "data/processed", a.dataset, f"seed_{a.seed}", "injected.txt"), dtype=np.int64, ndmin=2)
for run in a.run:
    path = os.path.join(ROOT, "results/pugib_runs", run, f"alpha_seed{a.seed}.npz")
    if not os.path.exists(path):
        print(f"{run}: no alpha snapshot at {path}")
        continue
    z = np.load(path)
    edges, alpha, obs = z["edges"].astype(np.int64), z["alpha"], z["is_obs"].astype(bool)
    edges = np.sort(edges, axis=1)
    e, al = edges[obs], alpha[obs]
    n = int(max(e.max(), inj.max())) + 1
    y = isin_pairs(e, inj, n)
    auc = float(roc_auc_score(y, 1 - al)) if 0 < y.sum() < len(y) else float("nan")
    rec = {"run": run, "dataset": a.dataset, "seed": a.seed, "n_observed": int(len(e)), "n_injected": int(y.sum()),
           "auroc_1_minus_alpha": auc, "alpha_mean_injected": float(al[y].mean()),
           "alpha_mean_genuine": float(al[~y].mean())}
    print(json.dumps(rec))
    os.makedirs(os.path.join(ROOT, "results/diagnostics"), exist_ok=True)
    with open(os.path.join(ROOT, "results/diagnostics/auroc_injected.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
