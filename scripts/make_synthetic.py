"""Synthetic social-recommendation world with a known latent graph Z (sanity suite).

Users and items have latent taste vectors (dimension 16, unit norm). Each user gets ~30
interactions sampled without replacement with probability proportional to
exp((u.i + b_i) / T) (Gumbel top-k), b_i a mild item popularity.
True graph Z: each user linked to its K most similar users by taste (cosine), symmetrized.
Observed graph A: every Z edge kept with probability c; false edges added uniformly among
non-Z pairs so that a fraction `false_frac` of the observed edges is false.
The same split protocol as the real data is applied (80/10/10 per user, 10% social hold-out).

Writes data/processed/<name>/seed_0/ (+ stats.json) and data/processed/<name>/truth/:
  Z.txt, A_true.txt, A_false.txt, tastes.npz, truth.json (generation parameters, global c/eps,
  and the channel on Omega of the TRAINING graph: pi, c, eps, p_A, rho_obs, rho_un, kappa*).
Usage: python scripts/make_synthetic.py [--name synth] [--users 2000] [--items 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data import Dataset, save_stats, symmetrize_edges  # noqa: E402
from src.pugib_prep import adjacency, build_omega, isin_pairs  # noqa: E402
from src.splits import make_split, save_split  # noqa: E402


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth")
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--items", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--inter", type=int, default=30, help="mean interactions per user")
    ap.add_argument("--temp", type=float, default=0.1, help="taste temperature of the interaction model")
    ap.add_argument("--pop", type=float, default=0.05, help="std of item popularity bias")
    ap.add_argument("--K", type=int, default=10, help="true neighbours per user")
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--false-frac", type=float, default=0.2, help="fraction of observed edges that are false")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    M, N = a.users, a.items

    U, V = unit(rng.normal(size=(M, a.dim))), unit(rng.normal(size=(N, a.dim)))
    bias = rng.normal(0, a.pop, N)
    n_u = rng.integers(int(a.inter * 2 / 3), int(a.inter * 4 / 3) + 1, M)
    pairs = []
    for u in range(M):
        keys = (U[u] @ V.T + bias) / a.temp + rng.gumbel(size=N)  # Gumbel top-k = sampling w/o replacement
        pairs += [(u, i) for i in np.argpartition(-keys, n_u[u])[: n_u[u]]]
    inter = np.unique(np.array(pairs, dtype=np.int64), axis=0)

    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, a.K, axis=1)[:, : a.K]
    Z = symmetrize_edges(np.stack([np.repeat(np.arange(M), a.K), nn.ravel()], 1))
    keep = rng.random(len(Z)) < a.c
    A_true = Z[keep]
    n_false = int(round(a.false_frac / (1 - a.false_frac) * len(A_true)))
    false = set()
    zset = set(map(tuple, Z.tolist()))
    while len(false) < n_false:
        i, j = sorted(rng.integers(0, M, 2))
        if i != j and (i, j) not in zset:
            false.add((i, j))
    A_false = np.array(sorted(false), dtype=np.int64)
    A = np.unique(np.concatenate([A_true, A_false]), axis=0)
    n_pairs = M * (M - 1) // 2
    eps_global = len(A_false) / (n_pairs - len(Z))

    ds = Dataset(M, N, inter, A)
    out = os.path.join(ROOT, "data/processed", a.name)
    stats = save_stats(ds, os.path.join(out, "stats.json"))
    sp = make_split(ds, a.seed)
    save_split(sp, os.path.join(out, f"seed_{a.seed}"), a.seed)

    # channel on Omega of the training graph (what the estimator sees)
    om = build_omega(adjacency(sp.social_train, M))
    y = isin_pairs(om, sp.social_train, M)
    z = isin_pairs(om, Z, M)
    p_a = float(y.mean())
    truth = {"params": vars(a), "n_Z": int(len(Z)), "n_A": int(len(A)), "n_A_true": int(len(A_true)),
             "n_A_false": int(len(A_false)), "false_frac_observed": len(A_false) / len(A),
             "c_global": float(keep.mean()), "eps_global": eps_global,
             "omega": {"size": int(len(om)), "p_A": p_a, "pi": float(z.mean()), "c": float(y[z].mean()),
                       "eps": float(y[~z].mean()), "rho_obs": float(z[y].mean()), "rho_un": float(z[~y].mean()),
                       "kappa_star": float(y[~z].mean() / p_a),
                       "train_edges_in_omega": int(y.sum()),
                       "false_train_edges": int((~isin_pairs(sp.social_train, Z, M)).sum()),
                       "false_train_edges_in_omega": int((y & ~z).sum())}}
    td = os.path.join(out, "truth")
    os.makedirs(td, exist_ok=True)
    for nm, e in [("Z", Z), ("A_true", A_true), ("A_false", A_false)]:
        np.savetxt(os.path.join(td, f"{nm}.txt"), e, fmt="%d")
    np.savez(os.path.join(td, "tastes.npz"), users=U, items=V, item_bias=bias)
    json.dump(truth, open(os.path.join(td, "truth.json"), "w"), indent=2)
    print(json.dumps({"interactions": stats["n_interactions"], "social_edges": stats["n_social_edges_undirected"],
                      **{k: truth[k] for k in ["n_Z", "n_A_true", "n_A_false", "false_frac_observed", "eps_global"]},
                      "omega": truth["omega"]}, indent=1))


if __name__ == "__main__":
    main()
