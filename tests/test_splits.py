import os

import numpy as np
import pandas as pd
import pytest

from src.data import build_dataset, symmetrize_edges
from src.splits import load_split, make_split, save_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(ROOT, "data/processed/lastfm")


def _as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


def _check_split(sp):
    tr, va, te = _as_set(sp.train), _as_set(sp.val), _as_set(sp.test)
    assert not (tr & va) and not (tr & te) and not (va & te)
    # every user keeps at least one train interaction
    assert set(np.unique(sp.train[:, 0])) == set(range(sp.n_users))
    s_tr, s_va = _as_set(sp.social_train), _as_set(sp.social_val)
    assert not (s_tr & s_va)
    for edges in (sp.social_train, sp.social_val):
        assert (edges[:, 0] < edges[:, 1]).all()  # canonical form, no self-loops
        assert len(_as_set(edges)) == len(edges)  # no duplicates
    # symmetric adjacency built from the canonical edge list
    n = sp.n_users
    A = np.zeros((n, n), bool)
    A[sp.social_train[:, 0], sp.social_train[:, 1]] = True
    A = A | A.T
    assert (A == A.T).all() and not A.diagonal().any()


def test_symmetrize():
    e = symmetrize_edges(np.array([[1, 2], [2, 1], [3, 3], [0, 2], [2, 0], [0, 2]]))
    assert _as_set(e) == {(1, 2), (0, 2)}


def test_build_dataset_filters_and_remaps():
    ui = pd.DataFrame({"user": [10, 10, 20, 30], "item": [5, 7, 5, 9], "weight": [3, 100, 1, 2]})
    uu = pd.DataFrame({"user": [10, 20, 10, 99], "friend": [20, 10, 10, 10]})
    ds = build_dataset(ui, uu)
    assert ds.n_users == 3 and ds.n_items == 3
    assert _as_set(ds.social) == {(0, 1)}  # 99 dropped, self-loop dropped, dedup


def test_synthetic_split():
    rng = np.random.default_rng(0)
    pairs = np.unique(np.stack([rng.integers(0, 50, 2000), rng.integers(0, 80, 2000)], 1), axis=0)
    pairs = np.concatenate([pairs, [[50, 0]]])  # user with a single interaction
    edges = symmetrize_edges(rng.integers(0, 51, (300, 2)))
    from src.data import Dataset
    sp = make_split(Dataset(51, 80, pairs, edges), seed=3)
    _check_split(sp)
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(pairs)
    assert len(sp.social_train) + len(sp.social_val) == len(edges)


def test_saved_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    pairs = np.unique(np.stack([rng.integers(0, 20, 400), rng.integers(0, 30, 400)], 1), axis=0)
    from src.data import Dataset
    sp = make_split(Dataset(20, 30, pairs, symmetrize_edges(rng.integers(0, 20, (60, 2)))), 0)
    save_split(sp, str(tmp_path), 0)
    sp2 = load_split(str(tmp_path))
    for k in ["train", "val", "test", "social_train", "social_val"]:
        assert (getattr(sp, k) == getattr(sp2, k)).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_processed_lastfm(seed):
    folder = os.path.join(PROCESSED, f"seed_{seed}")
    if not os.path.exists(folder):
        pytest.skip("run scripts/prepare_data.py first")
    sp = load_split(folder)
    _check_split(sp)
    # splits are deterministic: regenerating gives identical files
    from src.data import load_lastfm
    sp2 = make_split(load_lastfm(os.path.join(ROOT, "data/raw")), seed)
    assert (sp.test == sp2.test).all() and (sp.social_val == sp2.social_val).all()


def test_seeds_differ():
    if not os.path.exists(os.path.join(PROCESSED, "seed_1")):
        pytest.skip("run scripts/prepare_data.py first")
    a, b = load_split(os.path.join(PROCESSED, "seed_0")), load_split(os.path.join(PROCESSED, "seed_1"))
    assert _as_set(a.test) != _as_set(b.test)
    assert _as_set(a.social_val) != _as_set(b.social_val)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_processed_douban_book(seed):
    folder = os.path.join(ROOT, "data/processed/douban_book", f"seed_{seed}")
    if not os.path.exists(folder):
        pytest.skip("run scripts/prepare_data.py --dataset douban_book first")
    _check_split(load_split(folder))
