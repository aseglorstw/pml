"""Global vs two-hop candidate pools on a synthetic world (diagnostic, uses ground truth for stats/ceilings).

Global pool: for each user the top-K users by s = cosine of interaction-only embeddings ẽ among ALL users not
linked in the training graph (held-out ties stay eligible, as in U_rank); symmetric union.
Two-hop pool: the existing pool_s_k{K} (top-K within U_rank = Omega minus training ties).
Writes seed_k/pugib/best/global_pool_k{K}.txt, and run graphs in seed_k/ceiling/:
  {glob,hop}_ceil_k{K}.txt = A_train ∪ (pool ∩ Z)      {glob,hop}_hard_k{K}.txt = A_train ∪ pool
and seed_k/ceiling/pool_stats.json (size, precision wrt Z, fraction outside Omega, recall of Z \\ A_train).
Usage: python scripts/make_global_pools.py --seed 0 --K 5 10 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data import symmetrize_edges  # noqa: E402
from src.pugib_prep import isin_pairs  # noqa: E402
from src.splits import load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, nargs="+", default=[5, 10, 20])
a = ap.parse_args()
d = os.path.join(ROOT, "data/processed/synth", f"seed_{a.seed}")
sp = load_split(d)
M = sp.n_users
Z = np.loadtxt(os.path.join(d, "truth/Z.txt"), dtype=np.int64)
A = sp.social_train
om = np.load(os.path.join(d, "pugib/omega.npz"))["pairs"].astype(np.int64)
e = torch.load(os.path.join(d, "pugib/best/embeddings.pt"), map_location="cpu")["user_final"].numpy()
e = e / np.linalg.norm(e, axis=1, keepdims=True)
S = e @ e.T
S[np.arange(M), np.arange(M)] = -np.inf
S[A[:, 0], A[:, 1]] = -np.inf
S[A[:, 1], A[:, 0]] = -np.inf
missing = Z[~isin_pairs(Z, A, M)]  # true ties not in the training graph
out = os.path.join(d, "ceiling")
os.makedirs(out, exist_ok=True)
stats = {}
for k in a.K:
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    gpool = symmetrize_edges(np.stack([np.repeat(np.arange(M), k), nn.ravel()], 1))
    np.savetxt(os.path.join(d, f"pugib/best/global_pool_k{k}.txt"), gpool, fmt="%d")
    hpool = np.loadtxt(os.path.join(d, f"pugib/best/pool_s_k{k}.txt"), dtype=np.int64, ndmin=2)
    for kind, pool in (("glob", gpool), ("hop", hpool)):
        tz = isin_pairs(pool, Z, M)
        stats[f"{kind}_k{k}"] = {"size": int(len(pool)), "precision": float(tz.mean()),
                                 "frac_outside_omega": float((~isin_pairs(pool, om, M)).mean()),
                                 "recall_missing_true": float(isin_pairs(missing, pool, M).mean()),
                                 "n_true": int(tz.sum())}
        np.savetxt(os.path.join(out, f"{kind}_ceil_k{k}.txt"), np.unique(np.concatenate([A, pool[tz]]), axis=0), fmt="%d")
        np.savetxt(os.path.join(out, f"{kind}_hard_k{k}.txt"), np.unique(np.concatenate([A, pool]), axis=0), fmt="%d")
stats["n_missing_true"] = int(len(missing))
json.dump(stats, open(os.path.join(out, "pool_stats.json"), "w"), indent=1)
for key, v in stats.items():
    print(f"seed {a.seed} {key}: {v}")
