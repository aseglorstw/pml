"""Oracle ceiling graphs on a synthetic world (diagnostic; uses the ground truth Z).

  a_true.txt          training ties that are true (A_train ∩ Z)                 -> denoising ceiling
  a_plus_truecand.txt A_train ∪ (pool_s K=10 ∩ Z)                               -> completion ceiling
  a_true_truecand.txt (A_train ∩ Z) ∪ (pool_s K=10 ∩ Z)                         -> perfect PU-GIB on its support
Usage: python scripts/make_synth_ceilings.py --seed 0
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import isin_pairs  # noqa: E402
from src.splits import load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dataset", default="synth")
a = ap.parse_args()
d = os.path.join(ROOT, "data/processed", a.dataset, f"seed_{a.seed}")
sp = load_split(d)
M = sp.n_users
Z = np.loadtxt(os.path.join(d, "truth/Z.txt"), dtype=np.int64)
pool = np.loadtxt(os.path.join(d, "pugib/best/pool_s_k10.txt"), dtype=np.int64)
A = sp.social_train
a_true = A[isin_pairs(A, Z, M)]
cand_true = pool[isin_pairs(pool, Z, M)]
out = os.path.join(d, "ceiling")
os.makedirs(out, exist_ok=True)
for name, e in [("a_true", a_true), ("a_plus_truecand", np.concatenate([A, cand_true])),
                ("a_true_truecand", np.concatenate([a_true, cand_true]))]:
    e = np.unique(e, axis=0)
    np.savetxt(os.path.join(out, f"{name}.txt"), e, fmt="%d")
    print(f"seed {a.seed}: {name}: {len(e)} edges")
print(f"A_train {len(A)} ({len(a_true)} true, {len(A) - len(a_true)} false); true candidates in pool {len(cand_true)} of {len(pool)}")
