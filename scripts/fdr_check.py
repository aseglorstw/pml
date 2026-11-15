"""Does tie selection control the noise share among KEPT ties? (FDR view of the mixture prior; CPU, seconds per world.)

Each tie is a hypothesis "this tie is noise". Its conformal p-value against the reference of its group is
p = (1 + #{reference pairs with s >= s_tie}) / (n_ref + 1); a small p means the tie is more taste-similar than noise.
Our q̂ is Storey's estimate of the null share with λ = 1 - a, and ρ = 1 - local FDR. If noise ties are exchangeable with
the reference pairs, Benjamini-Hochberg on these p-values keeps a set with E[noise share among kept] <= α
(Storey-BH: level α / q̂). Reported against the truth: realized noise share among kept (FDP), share of genuine ties kept
(power), and for the MAP graph (ρ >= 0.5) the estimated FDR mean(1 - ρ) over kept ties vs the realized one.

Truth: truth/Z.txt (synthetic: noise = not in Z) or truth/injected.txt (real data with injected ties: noise = injected;
the original ties count as genuine, so the realized FDP there is the injected share only).
Usage: python scripts/fdr_check.py [--datasets synth synth50 ...] [--seeds 0 1 2] [--out fdr_check]
Writes results/<out>.md and .csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.pugib_prep import adjacency, build_omega, cosine_on_pairs, isin_pairs  # noqa: E402
from mixture_prior import noise_share, posterior  # noqa: E402

ALPHAS = (0.05, 0.1, 0.2)
SYNTH = ["synth", "synth50"] + [f"synth_sweep_f{f}" for f in range(0, 80, 10)] + ["synth_uni100", "synth_fof100"]


def pvalues(s_obs, s_ref):
    """Conformal p-values: (1 + #{ref >= s}) / (n + 1)."""
    ref = np.sort(s_ref)
    ge = len(ref) - np.searchsorted(ref, s_obs, side="left")
    return (1 + ge) / (len(ref) + 1)


def bh(p, alpha):
    """Benjamini-Hochberg rejections (here: ties kept) at level alpha."""
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if ok.any():
        keep[o[: np.flatnonzero(ok).max() + 1]] = True
    return keep


def world(ds, seed, reference, n_rand=200000):
    d = os.path.join(ROOT, "data/processed", ds, f"seed_{seed}")
    n = json.load(open(os.path.join(d, "meta.json")))["n_users"]
    e = np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64, ndmin=2)
    emb_path = os.path.join(d, "pugib", "best", "embeddings.pt")
    if not os.path.exists(emb_path):
        emb_path = os.path.join(ROOT, "results/checkpoints", f"lightgcn_{ds}_seed{seed}.pt")
    if not os.path.exists(emb_path):
        return None
    emb = torch.load(emb_path, map_location="cpu")["user_final"].numpy()
    if os.path.exists(os.path.join(d, "truth", "Z.txt")):
        noise = ~isin_pairs(e, np.loadtxt(os.path.join(d, "truth", "Z.txt"), dtype=np.int64), n)
    elif os.path.exists(os.path.join(d, "truth", "injected.txt")):
        noise = isin_pairs(np.sort(e, 1), np.sort(np.loadtxt(os.path.join(d, "truth", "injected.txt"),
                                                             dtype=np.int64, ndmin=2), 1), n)
    else:
        return None
    A = adjacency(e, n).tocsr()
    in_omega = np.asarray(A[e[:, 0]].multiply(A[e[:, 1]]).sum(1)).ravel() > 0
    rng = np.random.default_rng([seed, 7])  # same reference pairs as mixture_prior.py
    r = np.sort(rng.integers(0, n, (n_rand, 2)), 1)
    r = r[r[:, 0] != r[:, 1]]
    r = r[~isin_pairs(r, e, n)]
    s_obs, s_rand = cosine_on_pairs(emb, e), cosine_on_pairs(emb, r)
    ref = {"in_omega": s_rand, "bridges": s_rand}
    if reference == "matched":
        th = build_omega(A)
        th = th[~isin_pairs(th, e, n)]
        if len(th) > n_rand:
            th = th[np.sort(rng.choice(len(th), n_rand, replace=False))]
        cn_r = np.asarray(A[r[:, 0]].multiply(A[r[:, 1]]).sum(1)).ravel()
        ref = {"in_omega": cosine_on_pairs(emb, th), "bridges": s_rand[cn_r == 0]}
    p, rho, qhat = np.ones(len(e)), np.zeros(len(e)), np.ones(len(e))
    for g, m in (("in_omega", in_omega), ("bridges", ~in_omega)):
        if m.sum() == 0:
            continue
        q, _ = noise_share(s_obs[m], ref[g])
        p[m], rho[m], qhat[m] = pvalues(s_obs[m], ref[g]), posterior(s_obs[m], ref[g], q), q
    q_all = float(np.average([qhat[in_omega].mean() if in_omega.any() else 0, qhat[~in_omega].mean()],
                             weights=[in_omega.sum(), (~in_omega).sum()]))
    out = {"ds": ds, "seed": seed, "reference": reference, "ties": len(e), "noise_share": float(noise.mean())}
    for a in ALPHAS:
        for tag, lvl in (("bh", a), ("sbh", min(a / max(q_all, 1e-6), 1.0))):
            k = bh(p, lvl)
            out[f"{tag}{a}_kept"] = float(k.mean())
            out[f"{tag}{a}_fdp"] = float(noise[k].mean()) if k.any() else 0.0
            out[f"{tag}{a}_power"] = float(k[~noise].mean())
    k = rho >= 0.5
    out.update({"map_kept": float(k.mean()), "map_fdp": float(noise[k].mean()) if k.any() else 0.0,
                "map_fdr_est": float((1 - rho[k]).mean()) if k.any() else 0.0, "map_power": float(k[~noise].mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=SYNTH)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="fdr_check")
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        for s in a.seeds:
            for ref in ("uniform", "matched"):
                r = world(ds, s, ref)
                if r is None:
                    print(f"skip {ds} seed {s}: no embeddings or truth")
                    break
                rows.append(r)
                print(f"{ds} s{s} {ref}: noise {r['noise_share']:.2f} | BH .1 FDP {r['bh0.1_fdp']:.3f} kept "
                      f"{r['bh0.1_kept']:.2f} | MAP FDP {r['map_fdp']:.3f} est {r['map_fdr_est']:.3f}", flush=True)
    if not rows:
        return
    with open(os.path.join(ROOT, f"results/{a.out}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["# Noise share among kept ties: BH / Storey-BH on conformal p-values vs the MAP graph", "",
         "Per dataset and reference: mean over seeds; [max] = worst seed. FDP = realized noise share among kept ties; "
         "power = share of genuine ties kept. Guarantee: E[FDP] <= α if noise ties are exchangeable with the reference.",
         "", "| dataset | reference | noise share | " + " | ".join(f"BH {x}: FDP [max] / power" for x in ALPHAS)
         + " | " + " | ".join(f"Storey-BH {x}: FDP [max] / power" for x in ALPHAS)
         + " | MAP: FDP / est. / power |", "|" + "---|" * (4 + 2 * len(ALPHAS))]
    for ds in dict.fromkeys(r["ds"] for r in rows):
        for ref in ("uniform", "matched"):
            g = [r for r in rows if r["ds"] == ds and r["reference"] == ref]
            if not g:
                continue
            m = lambda k: np.mean([r[k] for r in g])  # noqa: E731
            mx = lambda k: np.max([r[k] for r in g])  # noqa: E731
            cells = [f"{m(f'{t}{x}_fdp'):.3f} [{mx(f'{t}{x}_fdp'):.3f}] / {m(f'{t}{x}_power'):.2f}"
                     for t in ("bh", "sbh") for x in ALPHAS]
            L.append(f"| {ds} | {ref} | {m('noise_share'):.2f} | " + " | ".join(cells)
                     + f" | {m('map_fdp'):.3f} / {m('map_fdr_est'):.3f} / {m('map_power'):.2f} |")
    open(os.path.join(ROOT, f"results/{a.out}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
