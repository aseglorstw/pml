"""PU-GIB preprocessing (paper sections 3.1-3.2), on the training social graph only.

Pairs are canonical (i < j) int64 arrays of shape (n, 2). Omega pairs are sorted
lexicographically, and every per-pair array (y, r, s, g) is aligned with them.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------- graph ---------
def adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric binary adjacency from canonical undirected edges."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    return A


def pair_keys(pairs: np.ndarray, n: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0] * n + pairs[:, 1]


def isin_pairs(pairs: np.ndarray, other: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask: which canonical `pairs` are in canonical `other`."""
    return np.isin(pair_keys(pairs, n), pair_keys(other, n))


def build_omega(A: sp.csr_matrix) -> np.ndarray:
    """All unordered pairs {i, j}, i != j, with >= 1 common neighbor (off-diagonal of A @ A)."""
    A2 = sp.triu(A @ A, k=1).tocoo()
    pairs = np.stack([A2.row, A2.col], 1).astype(np.int64)
    pairs = pairs[A2.data > 0]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def resource_allocation(A: sp.csr_matrix, pairs: np.ndarray) -> np.ndarray:
    """r_ij = sum_k A_ik A_kj / deg(k) = (A D^-1 A)_ij on the given pairs."""
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.zeros_like(deg)
    dinv[deg > 0] = 1.0 / deg[deg > 0]
    W = (A @ sp.diags(dinv) @ A).tocsr()
    return np.asarray(W[pairs[:, 0], pairs[:, 1]]).ravel()


def cosine_on_pairs(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ij->i", e[pairs[:, 0]], e[pairs[:, 1]])


# ------------------------------------------------------- channel estimation ------
def fit_isotonic(s: np.ndarray, y: np.ndarray) -> tuple[IsotonicRegression, np.ndarray]:
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    g = iso.fit_transform(s, y.astype(np.float64))
    return iso, g


def channel_estimates(p_a: float, g_val: float, kappa: float) -> dict:
    """Noise-corrected Elkan-Noto (eq. 6) and the resulting prior (eq. 4)."""
    eps = kappa * p_a
    upper = 1 - eps * (1 - p_a) / p_a
    c = p_a * (g_val - eps) / (p_a - eps)
    pi = (p_a - eps) / (c - eps)
    out = prior_from(p_a, c, pi)
    out.update({"kappa": kappa, "eps": eps, "upper_bound": upper,
                "admissible_upper": bool(g_val < upper), "admissible_lower": bool(g_val > p_a)})
    return out


def top_block_range(s: np.ndarray, y: np.ndarray, min_size: int = 1000) -> tuple[float, int, float]:
    """Fit isotonic g on (s, y); take its top block, merged downward until >= min_size pairs.

    Returns (lower s threshold of the merged block, its size, its fitted mean).
    """
    _, g = fit_isotonic(s, y)
    levels = np.unique(g)[::-1]  # fitted block values, highest first
    for lv in levels:
        mask = g >= lv
        if mask.sum() >= min_size:
            break
    return float(s[mask].min()), int(mask.sum()), float(y[mask].mean())


def anchor_g_top(s: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                 min_size: int = 1000) -> dict:
    """Cross-fitted positive rate in the top isotonic block.

    Omega is split into halves H1, H2. The top block (>= min_size pairs) of g fitted on one
    half defines an s-range [lo, inf); the positive rate of the other half in that range is
    measured. Both directions are averaged.
    """
    perm = rng.permutation(len(s))
    halves = [perm[: len(s) // 2], perm[len(s) // 2:]]
    folds = []
    for a, b in [(0, 1), (1, 0)]:
        fa, fb = halves[a], halves[b]
        lo, size, fit_mean = top_block_range(s[fa], y[fa], min_size)
        sel = s[fb] >= lo
        folds.append({"s_lo": lo, "block_size_fit": size, "block_rate_fit": fit_mean,
                      "n_eval": int(sel.sum()), "rate_eval": float(y[fb][sel].mean())})
    return {"g_top": float(np.mean([f["rate_eval"] for f in folds])), "folds": folds,
            "min_size": min_size}


def anchor_estimates(p_a: float, g_val: float, g_top: float, kappa: float) -> dict:
    """c_anchor = max(c_EN(eps), g_top); pi and rho as before."""
    en = channel_estimates(p_a, g_val, kappa)
    eps = en["eps"]
    c = max(en["c_hat"], g_top)
    pi = (p_a - eps) / (c - eps)
    out = prior_from(p_a, c, pi)
    out.update({"kappa": kappa, "eps": eps, "c_hat_en": en["c_hat"], "g_top": g_top,
                "anchor_active": bool(g_top > en["c_hat"])})
    return out


def uninformed_prior(p_a: float, kappa: float) -> dict:
    """Fallback: c_u = (1 + p_A) / 2, eps_u = kappa p_A, pi from eq. 6, rho from eq. 4."""
    eps = kappa * p_a
    c = (1 + p_a) / 2
    pi = (p_a - eps) / (c - eps)
    out = prior_from(p_a, c, pi)
    out.update({"kappa": kappa, "eps": eps})
    return out


def prior_from(p_a: float, c: float, pi: float) -> dict:
    rho_obs = pi * c / p_a
    rho_un = pi * (1 - c) / (1 - p_a)
    return {"c_hat": c, "pi_hat": pi, "rho_obs": rho_obs, "rho_un": rho_un,
            "valid": bool(0 < rho_un < rho_obs < 1)}


# --------------------------------------------------------------- candidates ------
def topk_pool(pairs: np.ndarray, score: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric union of each user's top-K pairs by score (ties broken at random).

    A pair is kept if either endpoint ranks the other in its top-K.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    u = np.concatenate([pairs[:, 0], pairs[:, 1]])
    idx = np.concatenate([np.arange(len(pairs))] * 2)
    sc = np.concatenate([score, score])
    order = np.lexsort((rng.random(len(u)), -sc, u))  # by user, then score desc, then random
    u_sorted = u[order]
    start = np.searchsorted(u_sorted, u_sorted, side="left")
    rank = np.arange(len(u_sorted)) - start
    keep = np.unique(idx[order][rank < K])
    return pairs[keep]


def random_pool(pairs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    sel = np.sort(rng.choice(len(pairs), size=min(size, len(pairs)), replace=False))
    return pairs[sel]


def pool_stats(pool: np.ndarray, p_val: np.ndarray, n_u_rank: int, n: int,
               n_e_val: int | None = None) -> dict:
    hits = int(isin_pairs(pool, p_val, n).sum())
    base = len(p_val) / n_u_rank
    out = {"size": int(len(pool)), "hits": hits,
           "recall": hits / max(len(p_val), 1),
           "precision": hits / max(len(pool), 1),
           "base_rate": base,
           "lift": (hits / max(len(pool), 1)) / base if base > 0 else float("nan"),
           "random_expected_recall": len(pool) / n_u_rank}
    if n_e_val is not None:
        out["recall_of_all_E_val"] = hits / max(n_e_val, 1)
    return out


def union_edges(*edge_sets: np.ndarray) -> np.ndarray:
    cat = np.concatenate([np.asarray(e, dtype=np.int64).reshape(-1, 2) for e in edge_sets])
    return np.unique(cat, axis=0)


def degree_stats(edges: np.ndarray, n: int) -> dict:
    d = np.bincount(np.asarray(edges).ravel(), minlength=n)
    return {"min": int(d.min()), "median": float(np.median(d)), "mean": float(d.mean()),
            "max": int(d.max()), "zero_degree_users": int((d == 0).sum())}
