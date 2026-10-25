"""LightGCN (He et al., SIGIR 2020) and LightGCN-S (paper eq. 8).

User update:
    e_u^{l+1} = sum_{i in N_I(u)} e_i^l / sqrt(d_u^I d_i^I)
                + omega_S * sum_{v in C(u)} w_uv / sqrt(d_u^C d_v^C) * e_v^l
Item update as in LightGCN. Final embedding = mean over layers 0..L.

d^C is the STATIC number of support edges of a user (not the sum of weights), so
down-weighting all of a user's ties actually reduces social aggregation.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def norm_bipartite_adj(train: np.ndarray, n_users: int, n_items: int) -> sp.coo_matrix:
    """D^{-1/2} A D^{-1/2} for the (M+N)x(M+N) user-item graph."""
    n = n_users + n_items
    r, c = train[:, 0], train[:, 1] + n_users
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    dinv = np.zeros(n)
    dinv[deg > 0] = deg[deg > 0] ** -0.5
    vals = dinv[rows] * dinv[cols]
    return sp.coo_matrix((vals.astype(np.float32), (rows, cols)), shape=(n, n))


def sparse_tensor(m: sp.coo_matrix, device) -> torch.Tensor:
    idx = torch.as_tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    t = torch.sparse_coo_tensor(idx, torch.as_tensor(m.data), m.shape, check_invariants=False).coalesce()
    return t.to(device)


class SocialGraph:
    """Undirected support edges with static degree normalization.

    edges: (E, 2) undirected pairs. Weights are given per undirected edge (shape (E,))
    and are applied to both directions, so the propagation stays symmetric.
    """

    def __init__(self, edges: np.ndarray, n_users: int, device, deg: np.ndarray | None = None):
        """deg: optional fixed per-user degree for the normalization (default: static edge count)."""
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.n_edges = len(edges)
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        if deg is None:
            deg = np.bincount(src, minlength=n_users)
        deg = np.asarray(deg, dtype=np.float64)
        norm = 1.0 / np.sqrt(deg[src] * deg[dst]) if len(src) else np.zeros(0)
        self.src = torch.as_tensor(src, device=device)  # message from src ...
        self.dst = torch.as_tensor(dst, device=device)  # ... to dst
        self.norm = torch.as_tensor(norm, dtype=torch.float32, device=device)
        self.n_users = n_users

    def propagate(self, e_users: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        """sum_v w_uv / sqrt(d_u d_v) e_v for every user u. Differentiable in w."""
        if w is None:
            coef = self.norm
        elif w.shape[0] == 2 * self.n_edges:  # one weight per direction (src -> dst order)
            coef = self.norm * w
        else:  # one weight per undirected edge, shared by both directions
            coef = self.norm * torch.cat([w, w])
        msg = e_users[self.src] * coef[:, None]
        return torch.zeros_like(e_users).index_add_(0, self.dst, msg)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, train: np.ndarray, dim=64, n_layers=3,
                 social_edges: np.ndarray | None = None, omega_s=0.5, device="cpu",
                 social_deg: np.ndarray | None = None):
        super().__init__()
        self.n_users, self.n_items, self.n_layers, self.omega_s = n_users, n_items, n_layers, omega_s
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.adj = sparse_tensor(norm_bipartite_adj(train, n_users, n_items), device)
        self.social = (SocialGraph(social_edges, n_users, device, social_deg)
                       if social_edges is not None else None)

    def propagate(self, social_w: torch.Tensor | None = None):
        """Returns final (user, item) embeddings. social_w: per-edge weights or None (=1)."""
        e = self.emb.weight
        layers = [e]
        for _ in range(self.n_layers):
            nxt = torch.sparse.mm(self.adj, e)
            if self.social is not None:
                soc = self.social.propagate(e[: self.n_users], social_w)
                nxt = torch.cat([nxt[: self.n_users] + self.omega_s * soc, nxt[self.n_users:]])
            e = nxt
            layers.append(e)
        out = torch.stack(layers).mean(0)
        return out[: self.n_users], out[self.n_users:]

    @torch.no_grad()
    def social_diagnostic(self, social_w: torch.Tensor | None = None) -> list[dict]:
        """Per layer: mean L2 norm over users of the interaction term and of the social
        term (omega_S * weighted neighbor sum) in the user update."""
        assert self.social is not None
        has_soc = torch.bincount(self.social.dst, minlength=self.n_users) > 0
        e = self.emb.weight
        out = []
        for layer in range(1, self.n_layers + 1):
            inter = torch.sparse.mm(self.adj, e)
            soc = self.omega_s * self.social.propagate(e[: self.n_users], social_w)
            ni, ns = inter[: self.n_users].norm(dim=1), soc.norm(dim=1)
            out.append({"layer": layer, "interaction_norm": ni.mean().item(),
                        "social_norm": ns.mean().item(),
                        "social_norm_users_with_ties": ns[has_soc].mean().item(),
                        "ratio_social_over_interaction": (ns[has_soc] / ni[has_soc].clamp_min(1e-12)).mean().item()})
            e = torch.cat([inter[: self.n_users] + soc, inter[self.n_users:]])
        return out

    def bpr_loss(self, users, pos, neg, reg: float, social_w=None):
        """Mean BPR loss + reg * 0.5 * ||ego embeddings of batch||^2 / batch (LightGCN convention)."""
        eu, ei = self.propagate(social_w)
        self._last_prop = (eu, ei)  # reused by GBSR's HSIC term
        u, p, n = eu[users], ei[pos], ei[neg]
        bpr = nn.functional.softplus(-((u * p).sum(1) - (u * n).sum(1))).mean()
        u0 = self.emb.weight[users]
        p0 = self.emb.weight[self.n_users + pos]
        n0 = self.emb.weight[self.n_users + neg]
        l2 = 0.5 * (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / len(users)
        return bpr + reg * l2, bpr.detach()

    def train_step_loss(self, users, pos, neg, reg: float):
        loss, bpr = self.bpr_loss(users, pos, neg, reg)
        return loss, {"bpr": bpr}

    @torch.no_grad()
    def score_fn(self, social_w=None):
        eu, ei = self.propagate(social_w)
        return lambda users: eu[users] @ ei.T


class Popularity:
    """Scores every item by its number of training interactions (same for all users)."""

    def __init__(self, train: np.ndarray, n_items: int):
        self.pop = torch.as_tensor(np.bincount(train[:, 1], minlength=n_items), dtype=torch.float32)

    def score_fn(self):
        return lambda users: self.pop.expand(len(users), -1)
