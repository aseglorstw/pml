"""Inject random false social edges into the training graph (LastFM corruption check).

False edges are sampled uniformly from Omega pairs that are not linked in the training graph and
not held out (E_val), so they are genuinely unobserved pairs of the two-hop universe.
mode=inject: E_train + n false edges (n = frac * |E_train|)
mode=replace: n = |E_train| false edges only (the observed graph is discarded)
Usage: python scripts/make_corrupted_social.py --seed 0 --frac 0.4 --mode inject
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
ap.add_argument("--dataset", default="lastfm")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frac", type=float, default=0.4)
ap.add_argument("--mode", choices=["inject", "replace"], default="inject")
ap.add_argument("--universe", choices=["omega", "all"], default="omega",
                help="omega: unlinked two-hop pairs (plausible-looking noise); all: uniform over all unlinked pairs")
a = ap.parse_args()
d = os.path.join(ROOT, "data/processed", a.dataset, f"seed_{a.seed}")
sp = load_split(d)
om = np.load(os.path.join(d, "pugib/omega.npz"))
pairs = om["pairs"].astype(np.int64)
free = pairs[~om["y"] & ~isin_pairs(pairs, sp.social_val, sp.n_users)]
rng = np.random.default_rng([a.seed, 7, int(a.frac * 100)])
n = int(round(a.frac * len(sp.social_train)))
if a.universe == "omega":
    fake = free[np.sort(rng.choice(len(free), n, replace=False))]
else:  # uniform over all user pairs not linked in train or held out
    taken = set(map(tuple, np.concatenate([sp.social_train, sp.social_val]).tolist()))
    fake = set()
    while len(fake) < n:
        i, j = sorted(rng.integers(0, sp.n_users, 2).tolist())
        if i != j and (i, j) not in taken:
            fake.add((i, j))
    fake = np.array(sorted(fake), dtype=np.int64)
edges = fake if a.mode == "replace" else np.unique(np.concatenate([sp.social_train, fake]), axis=0)
os.makedirs(os.path.join(d, "corrupt"), exist_ok=True)
name = f"{a.mode}{int(a.frac * 100)}{'' if a.universe == 'omega' else '_uniform'}.txt"
np.savetxt(os.path.join(d, "corrupt", name), edges, fmt="%d")
print(f"seed {a.seed}: {name}: {len(fake)} false edges ({a.universe}); graph has {len(edges)} edges")
