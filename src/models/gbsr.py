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
    """Options (all default to the behaviour described in the GBSR paper; the official code differs):
    detach_mask : True = official PyTorch code (torch_version/GBSR.py line 68), masker gets no gradient.
    eval_gate   : 'mean'   = noise-free gate sigmoid(logit/tau) + bias at evaluation;
                  'sample' = fresh logistic-noise gate at evaluation (official eval_test).
    gate_mode   : 'learned' = MLP gate; 'fixed1' = every social weight fixed to 1 (masked graph = observed
                  graph), HSIC kept (ablation).
    hsic_items  : False = HSIC on users only (ablation).
    """

    def __init__(self, n_users, n_items, train, social_edges, beta=5.0, sigma=0.25, edge_bias=0.5,
                 tau=0.2, detach_mask=False, batch_size=2048, dim=64, n_layers=3, omega_s=0.5,
                 device="cpu", social_form="ours", social_directed=False, init_std=0.1,
                 eval_gate="mean", gate_mode="learned", hsic_items=True):
        super().__init__()
        self.backbone = LightGCN(n_users, n_items, train, dim=dim, n_layers=n_layers,
                                 social_edges=social_edges, omega_s=omega_s, device=device,
                                 social_form=social_form, social_directed=social_directed, init_std=init_std)
        self.linear_1 = nn.Linear(2 * dim, dim)
        self.linear_2 = nn.Linear(dim, 1)
        self.beta, self.sigma, self.edge_bias, self.tau = beta, sigma, edge_bias, tau
        self.detach_mask, self.m = detach_mask, batch_size
        self.eval_gate, self.gate_mode, self.hsic_items = eval_gate, gate_mode, hsic_items
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
        if self.gate_mode == "fixed1":
            return torch.ones(self.backbone.social.src.shape[0], device=self.backbone.emb.weight.device)
        lg = self.logits()
        if sample:
            u = torch.rand_like(lg).clamp(1e-6, 1 - 1e-6)
            lg = lg + u.log() - (1 - u).log()
        g = torch.sigmoid(lg / self.tau) + self.edge_bias
        return g.detach() if self.detach_mask else g

    def train_step_loss(self, users, pos, neg, reg: float):
        w = self.gates(sample=True)
        rec, bpr = self.backbone.bpr_loss(users, pos, neg, reg, social_w=w)
        if self.beta == 0:
            return rec, {"bpr": bpr}
        eu_m, ei_m = self.backbone._last_prop
        eu_o, ei_o = self.backbone.propagate(None)  # observed (unmasked) graph
        uu = torch.unique(users)
        ib = hsic(kernel_matrix(F.normalize(eu_o[uu], dim=1), self.sigma),
                  kernel_matrix(F.normalize(eu_m[uu], dim=1), self.sigma), self.m)
        if self.hsic_items:
            ii = torch.unique(pos)
            ib = ib + hsic(kernel_matrix(F.normalize(ei_o[ii], dim=1), self.sigma),
                           kernel_matrix(F.normalize(ei_m[ii], dim=1), self.sigma), self.m)
        return rec + self.beta * ib, {"bpr": bpr, "hsic": ib.detach()}

    @torch.no_grad()
    def score_fn(self):
        return self.backbone.score_fn(social_w=self.gates(sample=self.eval_gate == "sample"))

    @torch.no_grad()
    def eval_diagnostics(self, n_pval_total=None) -> dict:
        g = self.gates(sample=False) - (0 if self.gate_mode == "fixed1" else self.edge_bias)
        return {"gate_mean": g.mean().item(), "gate_std": g.std().item(),
                "gate_frac_lt05": (g < 0.5).float().mean().item(),
                "gate_min": g.min().item(), "gate_max": g.max().item()}

    @torch.no_grad()
    def best_snapshot(self) -> dict:
        """Noise-free gate without edge bias. Undirected graphs: averaged over both directions of an edge."""
        g = (self.gates(sample=False) - (0 if self.gate_mode == "fixed1" else self.edge_bias)).cpu().numpy()
        s = self.backbone.social
        if s.directed:
            edges = np.stack([s.dst.cpu().numpy(), s.src.cpu().numpy()], 1)
            return {"alpha": g, "is_obs": np.ones(len(g), bool), "in_pval": np.zeros(len(g), bool), "edges": edges}
        E = s.n_edges
        und = (g[:E] + g[E:]) / 2
        edges = s.src[:E].cpu().numpy(), s.dst[:E].cpu().numpy()
        return {"alpha": und, "is_obs": np.ones(E, bool), "in_pval": np.zeros(E, bool),
                "edges": np.stack(edges, 1)}
