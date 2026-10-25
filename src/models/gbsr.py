"""GBSR (Yang et al., KDD 2024), ported from the official code
(github.com/yimutianyang/KDD24-GBSR, torch_version/GBSR.py) onto our LightGCN-S backbone (eq. 8).

Kept from the official code:
  * masker: Linear(2d, d) - ReLU - Linear(d, 1) on [e_u || e_v] of the trainable ego user
    embeddings, one gate per DIRECTED social entry (u receives from v);
  * gate = sigmoid((logit + logistic noise) / 0.2) + edge_bias (0.5), resampled every step;
  * loss = BPR + L2 + beta * HSIC(unmasked, masked) on L2-normalized propagated embeddings of the
    batch's unique users and unique positive items, kernel exp((x.y - 1) / sigma), m = batch size.
Changed for our protocol:
  * backbone: our LightGCN-S (static social degree, omega_S) instead of one jointly normalized
    user-item-user adjacency; warm start from the LightGCN checkpoint (set by build_model);
  * evaluation uses the deterministic gate sigmoid(logit / 0.2) + edge_bias (noise median)
    instead of a fresh random sample, so early stopping is reproducible;
  * detach_mask: the official PyTorch code detaches the gate (`weights = weights.detach()`),
    so the masker receives no gradient. detach_mask=True reproduces that; False lets gradients
    reach the masker, as the paper describes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightgcn import LightGCN


def kernel_matrix(x, sigma):
    return torch.exp((x @ x.T - 1) / sigma)


def hsic(Kx, Ky, m):
    Kxy = Kx @ Ky
    h = torch.trace(Kxy) / m ** 2 + Kx.mean() * Ky.mean() - 2 * Kxy.mean() / m
    return h * (m / (m - 1)) ** 2


class GBSR(nn.Module):
    def __init__(self, n_users, n_items, train, social_edges, beta=5.0, sigma=0.25, edge_bias=0.5,
                 tau=0.2, detach_mask=False, batch_size=2048, dim=64, n_layers=3, omega_s=0.5,
                 device="cpu"):
        super().__init__()
        self.backbone = LightGCN(n_users, n_items, train, dim=dim, n_layers=n_layers,
                                 social_edges=social_edges, omega_s=omega_s, device=device)
        self.linear_1 = nn.Linear(2 * dim, dim)
        self.linear_2 = nn.Linear(dim, 1)
        self.beta, self.sigma, self.edge_bias, self.tau = beta, sigma, edge_bias, tau
        self.detach_mask, self.m = detach_mask, batch_size
        self.n_users = n_users

    @property
    def emb(self):
        return self.backbone.emb

    def logits(self):
        s = self.backbone.social
        eu = self.backbone.emb.weight[: self.n_users]
        cat = torch.cat([eu[s.dst], eu[s.src]], 1)  # row = receiving user, col = neighbour
        return self.linear_2(F.relu(self.linear_1(cat))).view(-1)

    def gates(self, sample: bool):
        lg = self.logits()
        if sample:
            u = torch.rand_like(lg).clamp(1e-6, 1 - 1e-6)
            lg = lg + u.log() - (1 - u).log()
        g = torch.sigmoid(lg / self.tau) + self.edge_bias
        return g.detach() if self.detach_mask else g

    def train_step_loss(self, users, pos, neg, reg: float):
        w = self.gates(sample=True)
        rec, bpr = self.backbone.bpr_loss(users, pos, neg, reg, social_w=w)
        eu_m, ei_m = self.backbone._last_prop
        eu_o, ei_o = self.backbone.propagate(None)  # unmasked graph
        uu, ii = torch.unique(users), torch.unique(pos)
        ib = (hsic(kernel_matrix(F.normalize(eu_o[uu], dim=1), self.sigma),
                   kernel_matrix(F.normalize(eu_m[uu], dim=1), self.sigma), self.m)
              + hsic(kernel_matrix(F.normalize(ei_o[ii], dim=1), self.sigma),
                     kernel_matrix(F.normalize(ei_m[ii], dim=1), self.sigma), self.m))
        return rec + self.beta * ib, {"bpr": bpr, "hsic": ib.detach()}

    @torch.no_grad()
    def score_fn(self):
        return self.backbone.score_fn(social_w=self.gates(sample=False))

    @torch.no_grad()
    def eval_diagnostics(self, n_pval_total=None) -> dict:
        g = self.gates(sample=False) - self.edge_bias
        return {"gate_mean": g.mean().item(), "gate_std": g.std().item(),
                "gate_frac_lt05": (g < 0.5).float().mean().item(),
                "gate_min": g.min().item(), "gate_max": g.max().item()}

    @torch.no_grad()
    def best_snapshot(self) -> dict:
        """Noise-free gate without edge bias, averaged over the two directions of each edge."""
        g = (self.gates(sample=False) - self.edge_bias).cpu().numpy()
        E = self.backbone.social.n_edges
        und = (g[:E] + g[E:]) / 2
        s = self.backbone.social
        edges = s.src[:E].cpu().numpy(), s.dst[:E].cpu().numpy()
        return {"alpha": und, "is_obs": np.ones(E, bool), "in_pval": np.zeros(E, bool),
                "edges": np.stack(edges, 1)}
