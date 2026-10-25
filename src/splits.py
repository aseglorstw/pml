"""Seeded interaction and social splits (paper sections 3.1 and 4.1).

Files written per seed to data/processed/<dataset>/seed_<k>/:
  train.txt, val.txt, test.txt     user item           (interactions)
  social_train.txt, social_val.txt u v with u < v      (undirected edges)
  meta.json                        n_users, n_items, counts, seed
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .data import Dataset


@dataclass
class Split:
    n_users: int
    n_items: int
    train: np.ndarray  # (n, 2) [user, item]
    val: np.ndarray
    test: np.ndarray
    social_train: np.ndarray  # (E, 2) undirected u < v
    social_val: np.ndarray


def split_interactions(inter: np.ndarray, n_users: int, rng: np.random.Generator,
                       frac=(0.8, 0.1, 0.1)):
    """Per-user random 80/10/10 split; every user keeps >= 1 train interaction."""
    order = np.argsort(inter[:, 0], kind="stable")
    inter = inter[order]
    bounds = np.searchsorted(inter[:, 0], np.arange(n_users + 1))
    tr, va, te = [], [], []
    for u in range(n_users):
        rows = inter[bounds[u]:bounds[u + 1]]
        n = len(rows)
        if n == 0:
            continue
        perm = rng.permutation(n)
        n_val = int(round(frac[1] * n))
        n_test = int(round(frac[2] * n))
        # keep at least one training interaction
        while n_val + n_test > n - 1 and (n_val + n_test) > 0:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            else:
                n_val -= 1
        te.append(rows[perm[:n_test]])
        va.append(rows[perm[n_test:n_test + n_val]])
        tr.append(rows[perm[n_test + n_val:]])
    cat = lambda xs: np.concatenate(xs).astype(np.int64) if xs else np.zeros((0, 2), np.int64)
    return cat(tr), cat(va), cat(te)


def split_social(edges: np.ndarray, rng: np.random.Generator, val_frac=0.1):
    """Hold out a uniformly random fraction of undirected edges."""
    perm = rng.permutation(len(edges))
    n_val = int(round(val_frac * len(edges)))
    val = edges[np.sort(perm[:n_val])]
    train = edges[np.sort(perm[n_val:])]
    return train, val


def make_split(ds: Dataset, seed: int) -> Split:
    # Independent generators so the social hold-out does not depend on the interaction split.
    rng_i = np.random.default_rng([seed, 0])
    rng_s = np.random.default_rng([seed, 1])
    tr, va, te = split_interactions(ds.interactions, ds.n_users, rng_i)
    s_tr, s_va = split_social(ds.social, rng_s)
    return Split(ds.n_users, ds.n_items, tr, va, te, s_tr, s_va)


def save_split(sp: Split, folder: str, seed: int) -> None:
    os.makedirs(folder, exist_ok=True)
    for name in ["train", "val", "test", "social_train", "social_val"]:
        np.savetxt(os.path.join(folder, f"{name}.txt"), getattr(sp, name), fmt="%d")
    meta = {"seed": seed, "n_users": sp.n_users, "n_items": sp.n_items,
            **{f"n_{k}": int(len(getattr(sp, k)))
               for k in ["train", "val", "test", "social_train", "social_val"]}}
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_split(folder: str) -> Split:
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)

    def rd(name):
        a = np.loadtxt(os.path.join(folder, f"{name}.txt"), dtype=np.int64, ndmin=2)
        return a.reshape(-1, 2)

    return Split(meta["n_users"], meta["n_items"], rd("train"), rd("val"), rd("test"),
                 rd("social_train"), rd("social_val"))
