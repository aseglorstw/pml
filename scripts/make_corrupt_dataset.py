"""Build a dataset folder whose training social graph is a corrupted one.

Copies interactions, held-out social edges and meta of <src>/seed_k, replaces social_train with the
given edge file, writes injected.txt (edges not in the original training graph) and copies the
interaction-only LightGCN checkpoint (R_train is unchanged, so it is the same model).
Usage: python scripts/make_corrupt_dataset.py --src lastfm --seed 0 \
         --edges data/processed/lastfm/seed_0/corrupt/inject40.txt --name lastfm_inject40
"""
import argparse
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import isin_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="lastfm")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--edges", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
src = os.path.join(ROOT, "data/processed", a.src, f"seed_{a.seed}")
dst = os.path.join(ROOT, "data/processed", a.name, f"seed_{a.seed}")
os.makedirs(dst, exist_ok=True)
for f in ["train.txt", "val.txt", "test.txt", "social_val.txt", "meta.json"]:
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
orig = np.loadtxt(os.path.join(src, "social_train.txt"), dtype=np.int64, ndmin=2)
new = np.unique(np.loadtxt(os.path.join(ROOT, a.edges), dtype=np.int64, ndmin=2), axis=0)
n = int(max(orig.max(), new.max())) + 1
injected = new[~isin_pairs(new, orig, n)]
np.savetxt(os.path.join(dst, "social_train.txt"), new, fmt="%d")
np.savetxt(os.path.join(dst, "injected.txt"), injected, fmt="%d")
ck = os.path.join(ROOT, "results/checkpoints")
shutil.copy(os.path.join(ck, f"lightgcn_{a.src}_seed{a.seed}.pt"), os.path.join(ck, f"lightgcn_{a.name}_seed{a.seed}.pt"))
print(f"{a.name} seed {a.seed}: social_train {len(new)} edges, {len(injected)} injected")
