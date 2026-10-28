"""Per-tie prior for observed ties from a mixture-proportion estimate against random pairs.

Observed training ties are treated as a mixture: genuine ties plus noise ties whose similarity
s = cos(ẽ_i, ẽ_j) is distributed like that of uniformly random user pairs. Separately for ties inside
Omega and for bridges (outside Omega):
  q̂_g  = noise share, P_obs(s <= t) / P_rand(s <= t) at low quantiles t of the random pairs (median
          over quantiles 0.05, 0.1, 0.2, 0.3); an upper bound on the noise share if genuine ties
          rarely fall that low;
  ρ_ij = P(genuine | s_ij, g) = 1 - q̂_g * f_rand(s_ij) / f_obs,g(s_ij), the density ratio from an
          isotonic fit of P(observed | s) on observed ties of g vs random pairs.

Writes data/processed/<ds>/seed_<k>/pugib/mixture_prior.npz (edges = social_train order, rho, in_omega,
q_hat) and map_prior.txt (ties with rho >= 0.5). Compares with the truth when truth/Z.txt exists.

Usage: python scripts/mixture_prior.py --dataset synth50 --seeds 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import cosine_on_pairs, isin_pairs, pair_keys  # noqa: E402

QUANTILES = (0.05, 0.1, 0.2, 0.3)


def noise_share(s_obs, s_rand):
    ratios = [(s_obs <= np.quantile(s_rand, q)).mean() / q for q in QUANTILES]
    return float(np.clip(np.median(ratios), 0.0, 1.0)), ratios


def posterior(s_obs, s_rand, q):
    """rho = 1 - q * f_rand / f_obs, density ratio via isotonic P(observed | s)."""
    x = np.concatenate([s_obs, s_rand])
    y = np.concatenate([np.ones(len(s_obs)), np.zeros(len(s_rand))])
    h = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9).fit(x, y).predict(s_obs)
    ratio_rand_obs = (1 - h) / h * len(s_obs) / len(s_rand)  # f_rand(s) / f_obs(s)
    return np.clip(1 - q * ratio_rand_obs, 0.0, 1.0)


def process(ds, seed, n_rand, variant):
    d = os.path.join(ROOT, "data/processed", ds, f"seed_{seed}")
    n = json.load(open(os.path.join(d, "meta.json")))["n_users"]
    e = np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64, ndmin=2)
    emb = torch.load(os.path.join(d, "pugib", variant, "embeddings.pt"), map_location="cpu")["user_final"].numpy()
    om = np.load(os.path.join(d, "pugib", "omega.npz"))
    in_omega = np.isin(pair_keys(e, n), pair_keys(om["pairs"].astype(np.int64), n))
    rng = np.random.default_rng([seed, 7])
    r = np.sort(rng.integers(0, n, (n_rand, 2)), 1)
    r = r[r[:, 0] != r[:, 1]]
    r = r[~isin_pairs(r, e, n)]
    s_obs, s_rand = cosine_on_pairs(emb, e), cosine_on_pairs(emb, r)
    rho = np.zeros(len(e))
    info = {"dataset": ds, "seed": seed, "n_ties": int(len(e)), "n_random_pairs": int(len(r)), "groups": {}}
    for name, m in (("in_omega", in_omega), ("bridges", ~in_omega)):
        if m.sum() == 0:
            continue
        q, ratios = noise_share(s_obs[m], s_rand)
        rho[m] = posterior(s_obs[m], s_rand, q)
        info["groups"][name] = {"n": int(m.sum()), "q_hat": q, "ratios": ratios}
    keep = rho >= 0.5
    info["map_kept"] = int(keep.sum())
    tz = os.path.join(d, "truth", "Z.txt")
    if os.path.exists(tz):
        true = isin_pairs(e, np.loadtxt(tz, dtype=np.int64), n)
        info["truth"] = {"false_share": float((~true).mean()),
                         "false_share_in_omega": float((~true[in_omega]).mean()) if in_omega.any() else None,
                         "false_share_bridges": float((~true[~in_omega]).mean()),
                         "auroc_rho": float(roc_auc_score(true, rho)),
                         "mean_rho_true": float(rho[true].mean()), "mean_rho_false": float(rho[~true].mean()),
                         "map_false_removed": float((~true & ~keep).sum() / max((~true).sum(), 1)),
                         "map_true_lost": float((true & ~keep).sum() / true.sum()),
                         "map_false_share_left": float((~true[keep]).mean())}
    out = os.path.join(d, "pugib")
    np.savez_compressed(os.path.join(out, "mixture_prior.npz"), edges=e, rho=rho, in_omega=in_omega,
                        q_hat=np.array([info["groups"].get(g, {}).get("q_hat", np.nan) for g in ("in_omega", "bridges")]))
    np.savetxt(os.path.join(out, "map_prior.txt"), e[keep], fmt="%d")
    json.dump(info, open(os.path.join(out, "mixture_prior.json"), "w"), indent=2)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-random", type=int, default=200000)
    ap.add_argument("--variant", default="best")
    a = ap.parse_args()
    for s in a.seeds:
        process(a.dataset, s, a.n_random, a.variant)
