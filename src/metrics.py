"""Full-ranking Recall@K and NDCG@K with binary relevance."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def to_csr(pairs: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    pairs = np.asarray(pairs).reshape(-1, 2)
    m = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])),
                      shape=(n_users, n_items))
    m.data[:] = 1.0
    return m


def recall_ndcg(topk: np.ndarray, truth: sp.csr_matrix, ks=(10, 20)) -> dict:
    """topk: (n_users, max_k) ranked item ids. truth: csr of held-out items.

    Averages over users with at least one held-out item.
    """
    max_k = max(ks)
    n_true = np.diff(truth.indptr)
    users = np.where(n_true > 0)[0]
    # hits[u, r] = True if the item at rank r is relevant
    rows = np.repeat(users, max_k)
    cols = topk[users, :max_k].ravel()
    hits = np.asarray(truth[rows, cols]).reshape(len(users), max_k) > 0
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    out = {}
    for k in ks:
        h = hits[:, :k]
        out[f"recall@{k}"] = float((h.sum(1) / n_true[users]).mean())
        dcg = (h * discounts[:k]).sum(1)
        ideal_len = np.minimum(n_true[users], k)
        idcg = np.cumsum(discounts[:k])[ideal_len - 1]
        out[f"ndcg@{k}"] = float((dcg / idcg).mean())
    return out


def rank_topk(score_fn, n_users: int, exclude: sp.csr_matrix, k: int,
              batch_size: int = 1024, device="cpu") -> np.ndarray:
    """score_fn(user_idx_tensor) -> (b, n_items) scores. Excluded items get -inf."""
    out = np.zeros((n_users, k), dtype=np.int64)
    for start in range(0, n_users, batch_size):
        users = np.arange(start, min(start + batch_size, n_users))
        with torch.no_grad():
            # clone: score_fn may return a view (e.g. expanded popularity) that must not be masked in place
            s = score_fn(torch.as_tensor(users, device=device)).float().cpu().clone()
        ex = exclude[users].tocoo()
        s[torch.as_tensor(ex.row, dtype=torch.long), torch.as_tensor(ex.col, dtype=torch.long)] = -float("inf")
        out[users] = torch.topk(s, k, dim=1).indices.numpy()
    return out


def evaluate(score_fn, n_users: int, n_items: int, truth_pairs: np.ndarray,
             exclude_pairs: list[np.ndarray], ks=(10, 20), device="cpu") -> dict:
    """Full ranking over all items, excluding exclude_pairs (e.g. train, or train+val)."""
    exclude = to_csr(np.concatenate(exclude_pairs) if exclude_pairs else np.zeros((0, 2), int),
                     n_users, n_items)
    truth = to_csr(truth_pairs, n_users, n_items)
    topk = rank_topk(score_fn, n_users, exclude, max(ks), device=device)
    return recall_ndcg(topk, truth, ks)


def evaluate_per_user(score_fn, n_users: int, n_items: int, truth_pairs: np.ndarray,
                      exclude_pairs: list[np.ndarray], k: int = 20, device="cpu") -> dict:
    """Per-user Recall@k and NDCG@k (users with >= 1 held-out item), same protocol as evaluate()."""
    exclude = to_csr(np.concatenate(exclude_pairs) if exclude_pairs else np.zeros((0, 2), int), n_users, n_items)
    truth = to_csr(truth_pairs, n_users, n_items)
    topk = rank_topk(score_fn, n_users, exclude, k, device=device)
    n_true = np.diff(truth.indptr)
    users = np.where(n_true > 0)[0]
    hits = np.asarray(truth[np.repeat(users, k), topk[users, :k].ravel()]).reshape(len(users), k) > 0
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.cumsum(disc)[np.minimum(n_true[users], k) - 1]
    return {"users": users, "recall": hits.sum(1) / n_true[users], "ndcg": (hits * disc).sum(1) / idcg}
