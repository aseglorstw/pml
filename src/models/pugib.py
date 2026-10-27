"""PU-GIB (paper section 3.3): prior-anchored edge masker on a LightGCN-S backbone.

Loss = BPR (+ L2 on batch ego embeddings) + beta * mean_{C} KL(Bern(alpha) || Bern(rho)).
Training uses Binary Concrete edge weights; evaluation uses alpha.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """PU-GIB (paper Section 3.3).

    Support C = observed training ties (first n_obs rows of `support`) + candidates (the rest).
    Mask: alpha = sigmoid(MLP[e_i+e_j, |e_i-e_j|, e_i*e_j, log(1+r), A]) on frozen interaction-only ẽ.
    Propagation: observed ties in GBSR's joint adjacency (weight omega, static joint degrees), candidates in a
    separate channel with their own static degrees and weight omega_c; both multiplied by the relaxed mask.
    Objective: BPR + L2 + hsic_lambda * HSIC(observed-graph reps, masked reps) [users + items, as GBSR]
               + beta * mean_C KL(Bern(alpha) || Bern(rho_hat)).
    Evaluation weights every tie by alpha.
    """

    def __init__(self, n_users, n_items, train, support, n_obs, rho, feats, dim=64, n_layers=3,
                 omega_s=1.0, omega_c=0.25, beta=0.001, hsic_lambda=0.0, hsic_sigma=2.5, hsic_items=True,
                 batch_size=2048, hidden=64, tau_start=1.0, tau_end=0.2, tau_epochs=800, device="cpu",
                 in_prec=None, in_pest=None, pair_s=None, pair_r=None, social_form="gbsr", init_std=0.1):
        super().__init__()
        support = np.asarray(support, dtype=np.int64)
        obs, cand = support[:n_obs], support[n_obs:]
        self.backbone = LightGCN(n_users, n_items, train, dim=dim, n_layers=n_layers, social_edges=obs,
                                 omega_s=omega_s, device=device, social_form=social_form,
                                 cand_edges=cand if len(cand) else None, omega_c=omega_c, init_std=init_std)
        self.masker = nn.Sequential(nn.Linear(feats.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
        z = np.zeros(len(support), bool)
        self.register_buffer("feats", torch.as_tensor(feats))
        self.register_buffer("rho", torch.as_tensor(np.asarray(rho, dtype=np.float64), dtype=torch.float32))
        is_obs = np.zeros(len(support), bool)
        is_obs[:n_obs] = True
        self.register_buffer("is_obs", torch.as_tensor(is_obs))
        self.register_buffer("in_prec", torch.as_tensor(z if in_prec is None else np.asarray(in_prec, bool)))
        self.register_buffer("in_pest", torch.as_tensor(z if in_pest is None else np.asarray(in_pest, bool)))
        self.n_obs, self.n_users, self.m = n_obs, n_users, batch_size
        self.beta, self.hsic_lambda, self.hsic_sigma, self.hsic_items = beta, hsic_lambda, hsic_sigma, hsic_items
        self.tau_start, self.tau_end, self.tau_epochs = tau_start, tau_end, tau_epochs
        self.tau = tau_start
        self.support_np = support
        self.pair_s = None if pair_s is None else np.asarray(pair_s, dtype=np.float64)  # diagnostics only
        self.pair_r = None if pair_r is None else np.asarray(pair_r, dtype=np.float64)

    @property
    def emb(self):
        return self.backbone.emb

    def start_epoch(self, epoch: int):
        """Linear temperature annealing tau_start -> tau_end over tau_epochs epochs (the full budget)."""
        frac = min(1.0, (epoch - 1) / max(self.tau_epochs - 1, 1))
        self.tau = self.tau_start + frac * (self.tau_end - self.tau_start)

    def logits(self):
        return self.masker(self.feats).squeeze(-1)

    def _split(self, w):
        return w[: self.n_obs], (w[self.n_obs:] if len(w) > self.n_obs else None)

    def train_step_loss(self, users, pos, neg, reg: float):
        lg = self.logits()
        alpha = torch.sigmoid(lg)
        z_obs, z_cand = self._split(binary_concrete(lg, self.tau))  # one sample per undirected tie
        rec, bpr = self.backbone.bpr_loss(users, pos, neg, reg, social_w=z_obs, cand_w=z_cand)
        kl = kl_bernoulli(alpha, self.rho).mean()
        loss, parts = rec + self.beta * kl, {"bpr": bpr, "kl": kl.detach()}
        if self.hsic_lambda > 0:
            from .gbsr import hsic, kernel_matrix
            eu_m, ei_m = self.backbone._last_prop
            eu_o, ei_o = self.backbone.propagate(None, None, use_cand=False)  # observed graph, Z ≡ 1
            uu = torch.unique(users)
            ib = hsic(kernel_matrix(F.normalize(eu_o[uu], dim=1), self.hsic_sigma),
                      kernel_matrix(F.normalize(eu_m[uu], dim=1), self.hsic_sigma), self.m)
            if self.hsic_items:
                ii = torch.unique(pos)
                ib = ib + hsic(kernel_matrix(F.normalize(ei_o[ii], dim=1), self.hsic_sigma),
                               kernel_matrix(F.normalize(ei_m[ii], dim=1), self.hsic_sigma), self.m)
            loss = loss + self.hsic_lambda * ib
            parts["hsic"] = ib.detach()
        return loss, parts

    @torch.no_grad()
    def alpha(self):
        return torch.sigmoid(self.logits())

    @torch.no_grad()
    def score_fn(self):
        a_obs, a_cand = self._split(self.alpha())
        return self.backbone.score_fn(social_w=a_obs, cand_w=a_cand)

    @torch.no_grad()
    def eval_diagnostics(self, n_pval_total: int | None = None) -> dict:
        a = self.alpha()
        obs, cand = a[self.is_obs], a[~self.is_obs]
        out = {"tau": self.tau, "alpha_mean_obs": obs.mean().item(),
               "alpha_std_obs": obs.std(unbiased=False).item(),
               "alpha_mean_cand": cand.mean().item() if len(cand) else float("nan"),
               "alpha_std_cand": cand.std(unbiased=False).item() if len(cand) else float("nan"),
               "frac_cand_gt05": (cand > 0.5).float().mean().item() if len(cand) else float("nan"),
               "kl_mean": kl_bernoulli(a, self.rho).mean().item(),
               "alpha_min": a.min().item(), "alpha_max": a.max().item(),
               **self._recovery(a), **self._correlations(a)}
        return out

    def _recovery(self, a: torch.Tensor) -> dict:
        """AUROC of alpha over candidates: P_rec positives, P_est excluded from the negatives."""
        c = (~self.is_obs).cpu().numpy()
        if not c.any():
            return {}
        pos, est = self.in_prec.cpu().numpy() & c, self.in_pest.cpu().numpy() & c
        keep = c & ~est
        y = pos[keep]
        if y.sum() == 0 or y.sum() == keep.sum():
            return {"auroc_cand_prec": float("nan"), "n_prec_in_pool": int(pos.sum())}
        from sklearn.metrics import roc_auc_score
        return {"auroc_cand_prec": float(roc_auc_score(y, a.cpu().numpy()[keep])), "n_prec_in_pool": int(pos.sum())}

    def _correlations(self, a: torch.Tensor) -> dict:
        """Spearman correlation of alpha with s (cosine of ẽ) and with RA, on observed ties and candidates."""
        if self.pair_s is None:
            return {}
        from scipy.stats import spearmanr
        an, obs = a.detach().cpu().numpy(), self.is_obs.cpu().numpy()
        out = {}
        for name, x in (("s", self.pair_s), ("ra", self.pair_r)):
            for part, m in (("obs", obs), ("cand", ~obs)):
                ok = m.sum() > 2 and np.std(an[m]) > 0 and np.std(x[m]) > 0
                out[f"corr_alpha_{name}_{part}"] = float(spearmanr(an[m], x[m])[0]) if ok else float("nan")
        return out

    @torch.no_grad()
    def best_snapshot(self) -> dict:
        return {"alpha": self.alpha().cpu().numpy(), "is_obs": self.is_obs.cpu().numpy(),
                "in_pval": (self.in_prec | self.in_pest).cpu().numpy(), "in_prec": self.in_prec.cpu().numpy(),
                "in_pest": self.in_pest.cpu().numpy(), "edges": self.support_np}
