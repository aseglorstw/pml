"""Candidate files for the separate candidate channel and for targeted addition (synthetic, uses Z).

seed_k/ceiling/glob_truecand_k20.txt   global pool K=20 ∩ Z (candidates only)
seed_k/ceiling/target_pool_k10.txt     top-10 global candidates (by s, all unlinked users) of users with
                                       training social degree <= --maxdeg, symmetrized (candidates only)
seed_k/ceiling/target_truecand_k10.txt ... ∩ Z
seed_k/ceiling/target_{ceil,hard}_k10.txt  A_train ∪ the two sets above (joint form)
Usage: python scripts/make_cand_files.py --seed 0 [--maxdeg 2]
"""
import argparse
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
ap.add_argument("--maxdeg", type=int, default=2)
ap.add_argument("--K", type=int, default=10)
a = ap.parse_args()
d = os.path.join(ROOT, "data/processed/synth", f"seed_{a.seed}")
sp = load_split(d)
M, A = sp.n_users, sp.social_train
Z = np.loadtxt(os.path.join(d, "truth/Z.txt"), dtype=np.int64)
out = os.path.join(d, "ceiling")
g20 = np.loadtxt(os.path.join(d, "pugib/best/global_pool_k20.txt"), dtype=np.int64)
np.savetxt(os.path.join(out, "glob_truecand_k20.txt"), g20[isin_pairs(g20, Z, M)], fmt="%d")

e = torch.load(os.path.join(d, "pugib/best/embeddings.pt"), map_location="cpu")["user_final"].numpy()
e = e / np.linalg.norm(e, axis=1, keepdims=True)
S = e @ e.T
S[np.arange(M), np.arange(M)] = -np.inf
S[A[:, 0], A[:, 1]] = -np.inf
S[A[:, 1], A[:, 0]] = -np.inf
deg = np.bincount(A.ravel(), minlength=M)
low = np.where(deg <= a.maxdeg)[0]
nn = np.argpartition(-S[low], a.K, axis=1)[:, : a.K]
pool = symmetrize_edges(np.stack([np.repeat(low, a.K), nn.ravel()], 1))
tz = isin_pairs(pool, Z, M)
for name, x in [("target_pool", pool), ("target_truecand", pool[tz]),
                ("target_ceil", np.concatenate([A, pool[tz]])), ("target_hard", np.concatenate([A, pool]))]:
    np.savetxt(os.path.join(out, f"{name}_k{a.K}.txt"), np.unique(x, axis=0), fmt="%d")
print(f"seed {a.seed}: users with social degree <= {a.maxdeg}: {len(low)} of {M}; targeted pool {len(pool)} "
      f"(precision {tz.mean():.3f}); global K20 true candidates {int(isin_pairs(g20, Z, M).sum())}")
