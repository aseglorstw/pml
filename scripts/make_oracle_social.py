"""DIAGNOSTIC ONLY (deliberate test leak): oracle social graphs that test whether social propagation works.

Two oracles, each user connected to its top-K users by overlap (ties broken at random, only positive
overlap), then symmetrized:
  test_test  : |T_u ∩ T_v|           (users who will consume the same held-out items)
  test_train : |T_u ∩ R_v^train|     (v already consumed u's test items -> v's embedding points at them;
                                      the most direct leak through social propagation)
Written to data/processed/<dataset>/seed_k/oracle/<kind>_k<K>.txt (u v, u < v).
Usage: python scripts/make_oracle_social.py --dataset douban_book --seed 0 --K 10
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data import symmetrize_edges  # noqa: E402
from src.metrics import to_csr  # noqa: E402
from src.splits import load_split  # noqa: E402


def topk_neighbors(S: sp.csr_matrix, K: int, rng) -> np.ndarray:
    S = S.tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    pairs = []
    for u in range(S.shape[0]):
        cols, vals = S.indices[S.indptr[u]:S.indptr[u + 1]], S.data[S.indptr[u]:S.indptr[u + 1]]
        if len(cols) == 0:
            continue
        order = np.lexsort((rng.random(len(cols)), -vals))[:K]
        pairs += [(u, v) for v in cols[order]]
    return symmetrize_edges(np.array(pairs))


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="douban_book")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, default=10)
a = ap.parse_args()
d = os.path.join(ROOT, "data/processed", a.dataset, f"seed_{a.seed}")
spl = load_split(d)
T = to_csr(spl.test, spl.n_users, spl.n_items).astype(np.float32)
R = to_csr(spl.train, spl.n_users, spl.n_items).astype(np.float32)
os.makedirs(os.path.join(d, "oracle"), exist_ok=True)
for kind, S in [("test_test", T @ T.T), ("test_train", T @ R.T)]:
    e = topk_neighbors(S, a.K, np.random.default_rng([a.seed, 11]))
    deg = np.bincount(e.ravel(), minlength=spl.n_users)
    path = os.path.join(d, "oracle", f"{kind}_k{a.K}.txt")
    np.savetxt(path, e, fmt="%d")
    print(f"{kind}: {len(e)} undirected edges, users with ties {int((deg > 0).sum())}/{spl.n_users}, "
          f"median degree {np.median(deg[deg > 0]):.0f}, max {deg.max()} -> {os.path.relpath(path, ROOT)}")
