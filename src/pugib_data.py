"""Load the frozen PU-GIB inputs written by scripts/prepare_pugib.py for one seed."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .pugib_prep import isin_pairs, pair_keys
from .splits import Split


def prior_values(priors: dict, cfg: dict) -> tuple[float, float]:
    mode = cfg["prior"]
    if mode == "grid":
        return float(cfg["rho_obs"]), float(cfg["rho_un"])
    key = {"en": "estimated", "anchor": "anchor", "uninformed": "uninformed"}[mode]
    est = priors[key][str(cfg["kappa"])]
    return float(est["rho_obs"]), float(est["rho_un"])


def load_pugib_inputs(split: Split, seed_dir: str, cfg: dict) -> dict:
    """Support C = E_train (observed) + frozen candidate pool; per-edge prior, features, ẽ."""
    pdir = os.path.join(seed_dir, "pugib")
    vdir = os.path.join(pdir, cfg.get("embedding_variant", "best"))
    n = split.n_users
    om = np.load(os.path.join(pdir, "omega.npz"))
    omega, r_omega = om["pairs"].astype(np.int64), om["r"].astype(np.float64)
    # pool: s / ra / random / none (support = observed ties only). pool_k: use the top-K pool written
    # by scripts/make_pools.py (best/pool_s_k{K}.txt, pool_ra_k{K}.txt) instead of the default K=10 pool.
    k = cfg.get("pool_k")
    if cfg["pool"] == "none":
        pool = np.zeros((0, 2), dtype=np.int64)
    else:
        pool_file = {"s": os.path.join(vdir, f"pool_s_k{k}.txt" if k else "pool_s.txt"),
                     "ra": os.path.join(pdir, f"pool_ra_k{k}.txt" if k else "pool_ra.txt"),
                     "random": os.path.join(vdir, "pool_random.txt")}[cfg["pool"]]
        pool = np.loadtxt(pool_file, dtype=np.int64, ndmin=2).reshape(-1, 2)
    train = split.social_train
    assert not isin_pairs(pool, train, n).any()
    support = np.concatenate([train, pool])
    is_obs = np.concatenate([np.ones(len(train), bool), np.zeros(len(pool), bool)])

    # RA on the support: Omega lookup, 0 for training bridges (no common neighbour)
    keys_o, keys_c = pair_keys(omega, n), pair_keys(support, n)
    pos = np.searchsorted(keys_o, keys_c)
    pos = np.clip(pos, 0, len(keys_o) - 1)
    found = keys_o[pos] == keys_c
    r = np.where(found, r_omega[pos], 0.0)
    assert found[~is_obs].all()  # candidates are in Omega

    with open(os.path.join(vdir, "priors.json")) as f:
        priors = json.load(f)
    # LightGCN-PU has no prior (all weights 1); rho is then unused
    rho_obs, rho_un = prior_values(priors, cfg) if "prior" in cfg else (1.0, 1.0)
    rho = np.where(is_obs, rho_obs, rho_un)

    ck = torch.load(os.path.join(vdir, "embeddings.pt"), map_location="cpu")
    in_pval = isin_pairs(support, split.social_val, n)
    n_pval_total = int(om["in_pval"].sum())  # P_val = E_val ∩ Omega
    return {"support": support, "is_obs": is_obs, "r": r, "rho": rho, "rho_obs": rho_obs,
            "rho_un": rho_un, "e_tilde": ck["user_final"].numpy(), "ego": ck["ego"],
            "in_pval": in_pval, "n_pval_total": n_pval_total}
