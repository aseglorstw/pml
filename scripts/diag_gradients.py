"""Gradient diagnostics for the PU-GIB masker (LastFM seed 0, CPU).

Trains PU-GIB exactly like fit() for a fixed number of epochs and, for every batch, measures
separately the gradient of the BPR term and of the beta*KL term with respect to alpha, and the
gradient of the BPR term with respect to the masker parameters phi.

Usage:
  python scripts/diag_gradients.py --beta 0 --epochs 50 --out results/diagnostics/grad_beta0.json
  python scripts/diag_gradients.py --beta 0.01 --kappa 0.1 --norm prior --epochs 50 --out ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.metrics import evaluate  # noqa: E402,F401
from src.models.pugib import binary_concrete, kl_bernoulli  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.train import NegativeSampler, build_model, evaluate_split, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pugib/pugib_s_anchor.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--norm", default="prior")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    cfg.pop("tune", None)
    cfg.update({"beta": args.beta, "kappa": args.kappa, "norm": args.norm, "device": "cpu"})
    split = load_split(os.path.join(ROOT, cfg["data_dir"], f"seed_{args.seed}"))
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, extra = build_model(split, cfg, args.seed, ROOT, "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sampler = NegativeSampler(split.train, split.n_users, split.n_items, rng)
    phi = list(model.masker.parameters())
    is_obs = model.is_obs
    alpha0 = model.alpha().clone()

    # connectivity checks: the sample and the weights must be in the autograd graph
    lg = model.logits()
    z = binary_concrete(lg, model.tau)
    checks = {"logits_requires_grad": lg.requires_grad, "z_requires_grad": z.requires_grad,
              "z_grad_fn": type(z.grad_fn).__name__ if z.grad_fn else None}
    eu, _ = model.backbone.propagate(z)
    g = torch.autograd.grad(eu.sum(), z)[0]
    checks["d_user_emb_d_z_nonzero_frac"] = float((g != 0).float().mean())
    eu0, _ = model.backbone.propagate(torch.zeros_like(z))
    eu1, _ = model.backbone.propagate(torch.ones_like(z))
    checks["max_abs_emb_change_w0_vs_w1"] = float((eu1 - eu0).abs().max())

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.start_epoch(epoch)
        t0 = time.perf_counter()
        perm = rng.permutation(len(split.train))
        n_b = 0
        s_bpr = torch.zeros(len(is_obs), dtype=torch.float64)
        s2_bpr = torch.zeros_like(s_bpr)
        acc = {"bpr_alpha_l2": 0.0, "bpr_alpha_meanabs": 0.0, "kl_alpha_l2": 0.0, "kl_alpha_meanabs": 0.0,
               "bpr_phi_l2": 0.0, "kl_phi_l2": 0.0, "bpr": 0.0, "kl": 0.0}
        for start in range(0, len(perm), cfg["batch_size"]):
            b = split.train[perm[start:start + cfg["batch_size"]]]
            users, pos = torch.as_tensor(b[:, 0]), torch.as_tensor(b[:, 1])
            neg = torch.as_tensor(sampler.sample(b[:, 0]))
            lg = model.logits()
            alpha = torch.sigmoid(lg)
            lg_a = torch.log(alpha) - torch.log1p(-alpha)  # same value as lg, routes gradient via alpha
            z = binary_concrete(lg_a, model.tau)
            rec, bpr = model.backbone.bpr_loss(users, pos, neg, cfg["reg"], social_w=z)
            klt = model.beta * kl_bernoulli(alpha, model.rho).mean()
            g_bpr = torch.autograd.grad(rec, [alpha] + phi, retain_graph=True)
            g_kl = (torch.autograd.grad(klt, [alpha] + phi, retain_graph=True) if model.beta > 0
                    else [torch.zeros_like(alpha)] + [torch.zeros_like(p) for p in phi])
            ga, gk = g_bpr[0].detach().double(), g_kl[0].detach().double()
            s_bpr += ga
            s2_bpr += ga ** 2
            acc["bpr_alpha_l2"] += ga.norm().item()
            acc["bpr_alpha_meanabs"] += ga.abs().mean().item()
            acc["kl_alpha_l2"] += gk.norm().item()
            acc["kl_alpha_meanabs"] += gk.abs().mean().item()
            acc["bpr_phi_l2"] += torch.sqrt(sum((x ** 2).sum() for x in g_bpr[1:])).item()
            acc["kl_phi_l2"] += torch.sqrt(sum((x ** 2).sum() for x in g_kl[1:])).item()
            acc["bpr"] += bpr.item()
            acc["kl"] += klt.item()
            loss = rec + klt
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_b += 1
        rec_ = {k: v / n_b for k, v in acc.items()}
        mean = s_bpr / n_b
        std = torch.sqrt(torch.clamp(s2_bpr / n_b - mean ** 2, min=0))
        rec_["bpr_alpha_snr_median"] = float(torch.median(mean.abs() / (std + 1e-30)))
        rec_["bpr_alpha_meanabs_obs"] = float(mean[is_obs].abs().mean())
        rec_["bpr_alpha_meanabs_cand"] = float(mean[~is_obs].abs().mean())
        a = model.alpha()
        rec_.update({"epoch": epoch, "tau": model.tau, "alpha_std": a.std().item(), "alpha_min": a.min().item(),
                     "alpha_max": a.max().item(), "alpha_mean_obs": a[is_obs].mean().item(),
                     "alpha_mean_cand": a[~is_obs].mean().item(),
                     "alpha_max_abs_change_from_init": (a - alpha0).abs().max().item(),
                     "sec": time.perf_counter() - t0})
        hist.append(rec_)
        if epoch in (1, 2, 5) or epoch % 10 == 0:
            print(f"ep {epoch:3d} tau {model.tau:.2f} |dBPR/da| {rec_['bpr_alpha_meanabs']:.2e} "
                  f"(L2 {rec_['bpr_alpha_l2']:.2e}) |dKL/da| {rec_['kl_alpha_meanabs']:.2e} "
                  f"|dBPR/dphi| {rec_['bpr_phi_l2']:.2e} |dKL/dphi| {rec_['kl_phi_l2']:.2e} "
                  f"SNR {rec_['bpr_alpha_snr_median']:.3f} | alpha std {rec_['alpha_std']:.4f} "
                  f"min {rec_['alpha_min']:.4f} max {rec_['alpha_max']:.4f} obs {rec_['alpha_mean_obs']:.3f} "
                  f"cand {rec_['alpha_mean_cand']:.3f}", flush=True)
    model.eval()
    val = evaluate_split(model.score_fn(), split, "val", "cpu")
    out = {"args": vars(args), "checks": checks, "rho_obs": extra["rho_obs"], "rho_un": extra["rho_un"],
           "val_end": val, "history": hist}
    print("checks:", checks, "\nval at end:", val)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
