"""PU-GIB (paper section 3.3): prior-anchored edge masker on a LightGCN-S backbone.

Loss = BPR (+ L2 on batch ego embeddings) + beta * mean_{C} KL(Bern(alpha) || Bern(rho)).
Training uses Binary Concrete edge weights; evaluation uses alpha.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .lightgcn import LightGCN

EPS = 1e-6


def kl_bernoulli(alpha: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    a = alpha.clamp(EPS, 1 - EPS)
    r = rho.clamp(EPS, 1 - EPS)
    return a * (a.log() - r.log()) + (1 - a) * ((1 - a).log() - (1 - r).log())


def binary_concrete(logits: torch.Tensor, tau: float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp(EPS, 1 - EPS)
    return torch.sigmoid((logits + u.log() - (1 - u).log()) / tau)


def pair_features(emb: np.ndarray, pairs: np.ndarray, r: np.ndarray, is_obs: np.ndarray) -> np.ndarray:
    """[e_i + e_j, |e_i - e_j|, e_i * e_j, log(1 + r), A]  -- symmetric in (i, j)."""
    ei, ej = emb[pairs[:, 0]], emb[pairs[:, 1]]
    f = np.concatenate([ei + ej, np.abs(ei - ej), ei * ej, np.log1p(r)[:, None],
                        is_obs.astype(np.float64)[:, None]], 1)
    # column standardization: a fixed affine map per column, so symmetry is preserved
    mu, sd = f.mean(0), f.std(0)
    sd[sd < 1e-8] = 1.0
    return ((f - mu) / sd).astype(np.float32)


class PUGIB(nn.Module):
    def __init__(self, n_users, n_items, train, support, is_obs, rho, feats, ego_init,
                 dim=64, n_layers=3, omega_s=0.5, deg_mode="static", beta=0.1,
                 hidden=64, tau_start=1.0, tau_end=0.2, tau_epochs=1500, device="cpu",
                 in_pval=None):
        super().__init__()
        rho = np.asarray(rho, dtype=np.float64)
        if deg_mode == "static":
            deg = None
        elif deg_mode == "prior":  # expected degree under the prior, fixed during training
            deg = np.bincount(np.concatenate([support[:, 0], support[:, 1]]),
                              weights=np.concatenate([rho, rho]), minlength=n_users)
        else:
            raise ValueError(deg_mode)
        self.backbone = LightGCN(n_users, n_items, train, dim=dim, n_layers=n_layers,
                                 social_edges=support, omega_s=omega_s, device=device,
                                 social_deg=deg)
        with torch.no_grad():
            self.backbone.emb.weight.copy_(torch.as_tensor(ego_init))
        self.masker = nn.Sequential(nn.Linear(feats.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("feats", torch.as_tensor(feats))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.float32))
        self.register_buffer("is_obs", torch.as_tensor(np.asarray(is_obs, bool)))
        self.register_buffer("in_pval", torch.as_tensor(np.zeros(len(support), bool) if in_pval is None
                                                         else np.asarray(in_pval, bool)))
        self.beta, self.tau_start, self.tau_end, self.tau_epochs = beta, tau_start, tau_end, tau_epochs
        self.tau = tau_start
        self.n_users = n_users
        self.support_np = np.asarray(support, dtype=np.int64)

    @property
    def emb(self):
        return self.backbone.emb

    def start_epoch(self, epoch: int):
        """Linear temperature annealing tau_start -> tau_end over tau_epochs epochs."""
        frac = min(1.0, (epoch - 1) / max(self.tau_epochs - 1, 1))
        self.tau = self.tau_start + frac * (self.tau_end - self.tau_start)

    def logits(self):
        return self.masker(self.feats).squeeze(-1)

    def train_step_loss(self, users, pos, neg, reg: float):
        lg = self.logits()
        alpha = torch.sigmoid(lg)
        z = binary_concrete(lg, self.tau)
        rec, bpr = self.backbone.bpr_loss(users, pos, neg, reg, social_w=z)
        kl = kl_bernoulli(alpha, self.rho).mean()
        return rec + self.beta * kl, {"bpr": bpr, "kl": kl.detach()}

    @torch.no_grad()
    def alpha(self):
        return torch.sigmoid(self.logits())

    @torch.no_grad()
    def score_fn(self):
        return self.backbone.score_fn(social_w=self.alpha())

    @torch.no_grad()
    def eval_diagnostics(self, n_pval_total: int | None = None) -> dict:
        a = self.alpha()
        obs, cand = a[self.is_obs], a[~self.is_obs]
        pv = a[self.in_pval]
        n_pv = int(self.in_pval.sum()) if n_pval_total is None else n_pval_total
        return {"tau": self.tau,
                "alpha_mean_obs": obs.mean().item(), "alpha_mean_cand": cand.mean().item(),
                "frac_obs_gt05": (obs > 0.5).float().mean().item(),
                "frac_cand_gt05": (cand > 0.5).float().mean().item(),
                "alpha_mean_pval_in_pool": pv.mean().item() if len(pv) else float("nan"),
                "posterior_recall_pval": (pv > 0.5).sum().item() / max(n_pv, 1),
                "kl_mean": kl_bernoulli(a, self.rho).mean().item(),
                "alpha_std_obs": obs.std().item(), "alpha_std_cand": cand.std().item(),
                "alpha_min": a.min().item(), "alpha_max": a.max().item()}

    @torch.no_grad()
    def best_snapshot(self) -> dict:
        a = self.alpha().cpu().numpy()
        return {"alpha": a, "is_obs": self.is_obs.cpu().numpy(), "in_pval": self.in_pval.cpu().numpy(),
                "edges": self.support_np}
