"""The authors' own Douban-Book split (GBSR release) in our split format, for the GBSR port check.

douban_gbsr      : train = traindata.npy, test = testdata.npy, val = test (selection on test, as the
                   official code does); social_train = user_users_d.npy as DIRECTED entries, self-loops kept;
                   n_users / n_items = 13024 / 22347 as in the official run script.
douban_gbsr_val  : same, but 10% of each user's training interactions (rounded, users keep >= 1) are carved
                   out as a validation set used for selection; test unchanged.
lastfm_gbsrfmt   : (--smoke) LastFM seed 0 in the same format (val = test, directed social entries with two
                   self-loops added), only for quick local smoke tests.
The authors' files are read with the restricted unpickler from data/raw/gbsr_repo/src/datasets/douban_book
or external/KDD24-GBSR/datasets/douban_book (whichever exists).
Usage: python scripts/make_gbsr_split.py [--smoke]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data import safe_load_pickled_npy  # noqa: E402
from src.splits import load_split  # noqa: E402


def write(name, n_users, n_items, train, val, test, social, val_is_test, extra=None):
    d = os.path.join(ROOT, "data/processed", name, "seed_0")
    os.makedirs(d, exist_ok=True)
    for f, a in [("train", train), ("val", val), ("test", test), ("social_train", social),
                 ("social_val", np.zeros((0, 2), np.int64))]:
        np.savetxt(os.path.join(d, f"{f}.txt"), np.asarray(a, dtype=np.int64).reshape(-1, 2), fmt="%d")
    meta = {"seed": 0, "n_users": int(n_users), "n_items": int(n_items), "val_is_test": val_is_test,
            "social_directed": True, "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "n_social_entries": int(len(social)), **(extra or {})}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(name, json.dumps(meta))


def carve_val(train, frac, rng):
    """10% of each user's training interactions -> validation (rounded; every user keeps >= 1)."""
    order = np.argsort(train[:, 0], kind="stable")
    train = train[order]
    tr, va = [], []
    for u in np.unique(train[:, 0]):
        rows = train[train[:, 0] == u]
        k = min(int(round(frac * len(rows))), len(rows) - 1)
        p = rng.permutation(len(rows))
        va.append(rows[p[:k]])
        tr.append(rows[p[k:]])
    return np.concatenate(tr), np.concatenate(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sp = load_split(os.path.join(ROOT, "data/processed/lastfm/seed_0"))
        e = sp.social_train
        social = np.concatenate([e, e[:, ::-1], [[0, 0], [1, 1]]])
        write("lastfm_gbsrfmt", sp.n_users, sp.n_items, sp.train, sp.test, sp.test, social, True,
              {"source": "LastFM seed 0, smoke test only"})
        return
    for base in [os.path.join(ROOT, "data/raw/gbsr_repo/src/datasets/douban_book"),
                 os.path.join(ROOT, "external/KDD24-GBSR/datasets/douban_book")]:
        if os.path.exists(os.path.join(base, "traindata.npy")):
            break
    else:
        raise FileNotFoundError("authors' douban_book files not found (clone KDD24-GBSR into external/)")
    tr = safe_load_pickled_npy(os.path.join(base, "traindata.npy"))
    te = safe_load_pickled_npy(os.path.join(base, "testdata.npy"))
    so = safe_load_pickled_npy(os.path.join(base, "user_users_d.npy"))
    train = np.array([(u, i) for u, items in tr.items() for i in items], dtype=np.int64)
    test = np.array([(u, i) for u, items in te.items() for i in items], dtype=np.int64)
    social = np.array([(u, v) for u, vs in so.items() for v in vs], dtype=np.int64)
    src = {"source": base, "note": "authors' split; social entries directed as given (8 self-loops)"}
    write("douban_gbsr", 13024, 22347, train, test, test, social, True, src)
    tr2, va2 = carve_val(train, 0.1, np.random.default_rng(0))
    write("douban_gbsr_val", 13024, 22347, tr2, va2, test, social, False, {**src, "val": "10% of train per user"})


if __name__ == "__main__":
    main()
