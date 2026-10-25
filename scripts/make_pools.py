"""Write top-K candidate pools for several K (same ranking and tie-breaking as prepare_pugib.py).

Usage: python scripts/make_pools.py --dataset synth --seed 0 --K 2 5 10
Writes seed_k/pugib/best/pool_s_k{K}.txt and seed_k/pugib/pool_ra_k{K}.txt
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import topk_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="synth")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, nargs="+", default=[2, 5, 10])
a = ap.parse_args()
pd_ = os.path.join(ROOT, "data/processed", a.dataset, f"seed_{a.seed}", "pugib")
om = np.load(os.path.join(pd_, "omega.npz"))
pairs, y, r = om["pairs"].astype(np.int64), om["y"], om["r"]
s = np.load(os.path.join(pd_, "best", "s.npy"))
u = ~y
for k in a.K:
    for name, score, path in [("s", s, f"best/pool_s_k{k}.txt"), ("ra", r, f"pool_ra_k{k}.txt")]:
        pool = topk_pool(pairs[u], score[u], k, np.random.default_rng([a.seed, 2]))
        np.savetxt(os.path.join(pd_, path), pool, fmt="%d")
        print(f"{a.dataset} seed {a.seed}: {name} pool K={k}: {len(pool)} pairs")
