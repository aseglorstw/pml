"""Harder noise than random ties: does the noise estimate survive, and which estimator variant fixes it? (CPU only)

Noise types injected into a graph (in memory, nothing is written to data/processed):
  uniform  - uniformly random unlinked pairs (the current assumption);
  fof      - uniformly random unlinked friends of friends (pairs with >= 1 common neighbour);
  triadic  - triadic closure: a random friend of a random friend (random user u, random neighbour v of u,
             random neighbour w of v), so pairs with many common neighbours are drawn more often.
False ties never coincide with training, held-out or (synthetic) latent true ties.

Estimator variants (all use the same similarity s, posterior and MAP rule as scripts/mixture_prior.py):
  base     - two groups (triangle ties / bridges), uniform reference pairs, median over quantiles (the paper);
  pooled   - one group;
  q10      - single quantile 0.1 instead of the median over (0.05, 0.1, 0.2, 0.3);
  degree   - reference pairs with endpoints drawn proportional to their social degree;
  matched  - each group gets its own reference: unlinked pairs with >= 1 common neighbour for triangle ties,
             unlinked pairs without one for bridges;
  cn3      - three groups by the number of common neighbours (0, 1, >= 2), each with its own reference.

Graphs:
  synthetic - synth_sweep_f{0,20,50} seeds 0-2: the clean world (f0) with injected noise of each type, plus the
              existing uniform-noise worlds f20 and f50 as a check; truth known per tie;
  real      - <ds> seed 0 (default douban_book) with +50% injected ties of each type; only injected ties are false,
              so we report the estimated share of injected ties (as in the paper) and how many are removed.

Writes results/<out>.csv and results/<out>.md (default out: hard_noise_study).
Usage: python scripts/hard_noise_study.py [--real douban_book epinions yelp] [--skip-synthetic] [--out NAME]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mixture_prior import posterior  # noqa: E402
from src.pugib_prep import adjacency, cosine_on_pairs, isin_pairs, pair_keys  # noqa: E402

N_REF = 200000
QUANTILES = (0.05, 0.1, 0.2, 0.3)
VARIANTS = ("base", "pooled", "q10", "degree", "matched", "cn3")


# ----------------------------------------------------------------------------- graph helpers
def canon(p):
    p = np.sort(np.asarray(p, dtype=np.int64).reshape(-1, 2), 1)
    return p[p[:, 0] != p[:, 1]]


def common_neighbours(A, pairs):
    return np.asarray(A[pairs[:, 0]].multiply(A[pairs[:, 1]]).sum(1)).ravel().astype(np.int64)


def two_hop(A):
    """All unordered pairs with >= 1 common neighbour and their count (upper triangle of A @ A)."""
    C = sp.triu(A @ A, k=1).tocoo()
    m = C.data > 0
    return np.stack([C.row[m], C.col[m]], 1).astype(np.int64), C.data[m].astype(np.int64)


def noise_share(s_obs, s_ref, quantiles):
    r = [(s_obs <= np.quantile(s_ref, q)).mean() / q for q in quantiles]
    return float(np.clip(np.median(r), 0.0, 1.0))


# ----------------------------------------------------------------------------- noise injection
def inject(kind, e, forbidden, n, n_false, rng):
    """n_false new ties of the given kind; forbidden = canonical pairs that must not be drawn."""
    taken = set(pair_keys(forbidden, n).tolist()) | set(pair_keys(e, n).tolist())
    if kind == "uniform":
        out = set()
        while len(out) < n_false:
            p = canon(rng.integers(0, n, (2 * n_false, 2)))
            for k in pair_keys(p, n).tolist():
                if k not in taken and k not in out:
                    out.add(k)
                    if len(out) == n_false:
                        break
        keys = np.array(sorted(out), dtype=np.int64)
    elif kind == "fof":
        pairs, _ = two_hop(adjacency(e, n))
        k = pair_keys(pairs, n)
        k = k[~np.isin(k, np.fromiter(taken, np.int64))]
        if len(k) < n_false:
            raise ValueError(f"only {len(k)} free friend-of-friend pairs for {n_false} false ties")
        keys = np.sort(rng.choice(k, n_false, replace=False))
    elif kind == "triadic":
        A = adjacency(e, n).tocsr()
        deg = np.diff(A.indptr)
        users = np.flatnonzero(deg > 0)
        out = set()
        while len(out) < n_false:
            u = rng.choice(users, 4 * n_false)
            v = A.indices[A.indptr[u] + (rng.random(len(u)) * deg[u]).astype(np.int64)]
            w = A.indices[A.indptr[v] + (rng.random(len(v)) * deg[v]).astype(np.int64)]
            for kk in pair_keys(canon(np.stack([u, w], 1)), n).tolist():
                if kk not in taken and kk not in out:
                    out.add(kk)
                    if len(out) == n_false:
                        break
        keys = np.array(sorted(out), dtype=np.int64)
    else:
        raise ValueError(kind)
    return np.stack([keys // n, keys % n], 1)


# ----------------------------------------------------------------------------- estimator variants
class Refs:
    """Reference pairs on the observed graph (computed once per graph, shared by the variants)."""

    def __init__(self, e, n, emb, seed):
        self.A = adjacency(e, n).tocsr()
        rng = np.random.default_rng([seed, 7])
        r = canon(rng.integers(0, n, (N_REF, 2)))
        r = r[~isin_pairs(r, e, n)]
        self.uniform, self.uniform_cn = r, common_neighbours(self.A, r)
        deg = np.diff(self.A.indptr).astype(float)
        r = canon(rng.choice(n, (N_REF, 2), p=deg / deg.sum()))
        self.degree = r[~isin_pairs(r, e, n)]
        th, cn = two_hop(self.A)
        free = ~isin_pairs(th, e, n)
        th, cn = th[free], cn[free]
        self.cn_pairs = {}
        for name, m in (("ge1", cn >= 1), ("eq1", cn == 1), ("ge2", cn >= 2)):
            p = th[m]
            if len(p) > N_REF:
                p = p[np.sort(rng.choice(len(p), N_REF, replace=False))]
            self.cn_pairs[name] = p
        self.cn_pairs["eq0"] = self.uniform[self.uniform_cn == 0]
        self.emb = emb
        self.s = {}

    def sim(self, name):
        if name not in self.s:
            p = {"uniform": self.uniform, "degree": self.degree}.get(name)
            p = self.cn_pairs[name] if p is None else p
            self.s[name] = cosine_on_pairs(self.emb, p)
        return self.s[name]


def estimate(variant, s_obs, cn, refs):
    """Returns (rho, {group: (mask, q_hat)})."""
    tri = cn >= 1
    if variant == "pooled":
        groups = {"all": (np.ones(len(cn), bool), "uniform")}
    elif variant in ("base", "q10"):
        groups = {"tri": (tri, "uniform"), "bridge": (~tri, "uniform")}
    elif variant == "degree":
        groups = {"tri": (tri, "degree"), "bridge": (~tri, "degree")}
    elif variant == "matched":
        groups = {"tri": (tri, "ge1"), "bridge": (~tri, "eq0")}
    elif variant == "cn3":
        groups = {"cn1": (cn == 1, "eq1"), "cn2+": (cn >= 2, "ge2"), "bridge": (cn == 0, "eq0")}
    quantiles = (0.1,) if variant == "q10" else QUANTILES
    rho, out = np.zeros(len(cn)), {}
    for g, (m, ref) in groups.items():
        if m.sum() == 0:
            continue
        s_ref = refs.sim(ref)
        q = noise_share(s_obs[m], s_ref, quantiles)
        rho[m] = posterior(s_obs[m], s_ref, q)
        out[g] = (m, q)
    return rho, out


def summarize(rho, groups, cn, false=None):
    """q_hat overall and for triangle ties / bridges (weighted over sub-groups), MAP errors if truth known."""
    n_hat = sum(q * m.sum() for m, q in groups.values())
    tri = cn >= 1
    hat_tri = sum(q * (m & tri).sum() for m, q in groups.values()) / max(tri.sum(), 1) if "all" not in groups else np.nan
    hat_br = sum(q * (m & ~tri).sum() for m, q in groups.values()) / max((~tri).sum(), 1) if "all" not in groups else np.nan
    keep = rho >= 0.5
    r = {"q_hat": n_hat / len(cn), "q_hat_tri": hat_tri, "q_hat_bridge": hat_br, "kept": keep.mean()}
    if false is not None:
        r.update({"q_true": false.mean(), "q_true_tri": false[tri].mean() if tri.any() else np.nan,
                  "q_true_bridge": false[~tri].mean() if (~tri).any() else np.nan,
                  "auroc": roc_auc_score(~false, rho) if 0 < false.sum() < len(false) else np.nan,
                  "false_removed": (false & ~keep).sum() / max(false.sum(), 1),
                  "genuine_lost": (~false & ~keep).sum() / max((~false).sum(), 1),
                  "false_left": false[keep].mean() if keep.any() else np.nan})
    return r


def run_graph(e, n, emb, seed, false=None):
    """All variants on one observed graph e (canonical); false = known false ties (bool) or None."""
    refs = Refs(e, n, emb, seed)
    cn = common_neighbours(refs.A, e)
    s_obs = cosine_on_pairs(emb, e)
    rows = []
    for v in VARIANTS:
        rho, groups = estimate(v, s_obs, cn, refs)
        rows.append({"variant": v, **summarize(rho, groups, cn, false), "_rho": rho})
    return rows, cn


# ----------------------------------------------------------------------------- data
def load(ds, seed):
    d = os.path.join(ROOT, "data/processed", ds, f"seed_{seed}")
    n = json.load(open(os.path.join(d, "meta.json")))["n_users"]
    rd = lambda f: canon(np.loadtxt(os.path.join(d, f), dtype=np.int64, ndmin=2))
    e, val = rd("social_train.txt"), rd("social_val.txt")
    emb_path = os.path.join(d, "pugib/best/embeddings.pt")
    if not os.path.exists(emb_path):
        emb_path = os.path.join(ROOT, "results/checkpoints", f"lightgcn_{ds}_seed{seed}.pt")
    emb = torch.load(emb_path, map_location="cpu")["user_final"].numpy()
    z = os.path.join(d, "truth/Z.txt")
    Z = canon(np.loadtxt(z, dtype=np.int64)) if os.path.exists(z) else None
    return e, val, n, emb, Z


def synthetic(levels=(0.2, 0.5), seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        e, val, n, emb, Z = load("synth_sweep_f0", s)
        forbidden = np.concatenate([val, Z])
        for kind in ("uniform", "fof", "triadic"):
            for lvl in levels:
                n_false = int(round(lvl / (1 - lvl) * len(e)))
                f = inject(kind, e, forbidden, n, n_false, np.random.default_rng([s, 11, int(lvl * 100)]))
                g = np.concatenate([e, f])
                o = np.lexsort((g[:, 1], g[:, 0]))
                g, false = g[o], np.r_[np.zeros(len(e), bool), np.ones(len(f), bool)][o]
                res, _ = run_graph(g, n, emb, s, false)
                rows += [{"graph": "synthetic", "noise": kind, "level": lvl, "seed": s, **r} for r in res]
                print(f"synthetic seed {s} {kind} {lvl}: base q {res[0]['q_hat']:.3f} (true {res[0]['q_true']:.3f})",
                      flush=True)
        for lvl, ds in ((0.2, "synth_sweep_f20"), (0.5, "synth_sweep_f50")):  # generator noise, as in the paper
            e2, _, n2, emb2, Z2 = load(ds, s)
            res, _ = run_graph(e2, n2, emb2, s, ~isin_pairs(e2, Z2, n2))
            rows += [{"graph": "synthetic", "noise": "uniform (generator)", "level": lvl, "seed": s, **r} for r in res]
    return rows


def real(ds, frac=0.5, seed=0):
    e, val, n, emb, _ = load(ds, seed)
    base, _ = run_graph(e, n, emb, seed)
    base = {r["variant"]: r for r in base}
    rows = []
    for kind in ("uniform", "fof", "triadic"):
        f = inject(kind, e, val, n, int(round(frac * len(e))), np.random.default_rng([seed, 11, int(frac * 100)]))
        g = np.concatenate([e, f])
        o = np.lexsort((g[:, 1], g[:, 0]))
        g, inj = g[o], np.r_[np.zeros(len(e), bool), np.ones(len(f), bool)][o]
        res, cn = run_graph(g, n, emb, seed)
        for r in res:
            b = base[r["variant"]]
            rho = r.pop("_rho")
            keep = rho >= 0.5
            rows.append({"graph": ds, "noise": kind, "level": len(f) / len(g), "seed": seed,
                         "variant": r["variant"], "q_true": len(f) / len(g),
                         # estimated share of injected ties: estimated noisy ties after minus before
                         "q_hat": (r["q_hat"] * len(g) - b["q_hat"] * len(e)) / len(g),
                         "injected_in_triangles": (cn[inj] >= 1).mean(),
                         "false_removed": (inj & ~keep).sum() / inj.sum(),
                         "original_kept": keep[~inj].mean(), "original_kept_before": b["kept"],
                         "auroc": roc_auc_score(~inj, rho)})
        print(f"{ds} {kind}: base estimate of injected share {rows[-len(res)]['q_hat']:.3f} "
              f"(true {len(f) / len(g):.3f})", flush=True)
    return rows


def taste_check(seeds=(0, 1, 2)):
    """Synthetic only: true taste cosine (latent vectors) and s of genuine ties, random pairs and each noise type."""
    L = ["## Synthetic: how preference-related is each kind of false tie?", "",
         "True taste = cosine of the latent taste vectors; 'below q30' = share below the 30% quantile of random "
         "pairs (true taste). Mean over seeds 0-2, false ties = 50% of the clean graph size.", "",
         "| ties | true taste | s | below q30 of random pairs |", "|---|---|---|---|"]
    acc = {}
    for s in seeds:
        e, val, n, emb, Z = load("synth_sweep_f0", s)
        U = np.load(os.path.join(ROOT, "data/processed/synth_sweep_f0", f"seed_{s}", "truth/tastes.npz"))["users"]
        r = canon(np.random.default_rng([s, 99]).integers(0, n, (50000, 2)))
        q30 = np.quantile(cosine_on_pairs(U, r), 0.3)
        sets = {"genuine": e, "random pairs": r}
        for k in ("uniform", "fof", "triadic"):
            sets[k] = inject(k, e, np.concatenate([val, Z]), n, len(e) // 2, np.random.default_rng([s, 5]))
        for k, p in sets.items():
            tt = cosine_on_pairs(U, p)
            acc.setdefault(k, []).append((tt.mean(), cosine_on_pairs(emb, p).mean(), (tt <= q30).mean()))
    for k, v in acc.items():
        m = np.mean(v, 0)
        L.append(f"| {k} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- report
def report(df):
    L = ["# Harder noise: estimator variants (CPU, in-memory injection)", "",
         "Synthetic: clean world synth_sweep_f0 with injected false ties of each type (share 0.2 / 0.5), plus the "
         "generator's uniform-noise worlds f20 / f50; mean over seeds 0-2 (max |error| over seeds in brackets). "
         "Douban-Book seed 0: +50% injected ties (share 0.333); q̂ = estimated share of injected ties.", ""]
    syn = df[df.graph == "synthetic"].copy()
    if len(syn):
        for c in ("", "_tri", "_bridge"):
            syn["err" + c] = syn["q_hat" + c] - syn["q_true" + c]
        L += ["## Synthetic", "",
              "| noise | share | variant | q̂ − q | q̂ − q, triangles | q̂ − q, bridges | true share in triangles "
              "| AUROC | false removed | genuine lost |", "|---|---|---|---|---|---|---|---|---|---|"]
        for (kind, lvl), part in syn.groupby(["noise", "level"], sort=False):
            for v in VARIANTS:
                p = part[part.variant == v]
                f = lambda c: f"{p[c].mean():+.3f} ({p[c].abs().max():.3f})" if p[c].notna().all() else "–"
                L.append(f"| {kind} | {lvl:.1f} | {v} | {f('err')} | {f('err_tri')} | {f('err_bridge')} | "
                         f"{p.q_true_tri.mean():.2f} | {p.auroc.mean():.3f} | {p.false_removed.mean():.0%} | "
                         f"{p.genuine_lost.mean():.1%} |")
        L.append("")
    for ds in [g for g in df.graph.unique() if g != "synthetic"]:
        db = df[df.graph == ds]
        L += ["", f"## {ds}, +50% injected ties (seed 0)", "",
              "| noise | injected in triangles | variant | q̂ injected (true 0.333) | AUROC injected vs original "
              "| injected removed | original kept (before injection) |", "|---|---|---|---|---|---|---|"]
        for _, r in db.iterrows():
            L.append(f"| {r.noise} | {r.injected_in_triangles:.0%} | {r.variant} | {r.q_hat:.3f} | {r.auroc:.3f} | "
                     f"{r.false_removed:.0%} | {r.original_kept:.1%} ({r.original_kept_before:.1%}) |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=["douban_book"],
                    help="real datasets with local embeddings (data/processed/<ds>/seed_0/pugib/best)")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--out", default="hard_noise_study", help="name of results/<out>.{csv,md}")
    a = ap.parse_args()
    rows = ([] if a.skip_synthetic else synthetic()) + [r for ds in a.real for r in real(ds)]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_rho"} for r in rows])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, f"results/{a.out}.csv"), index=False)
    txt = report(df) + ("" if a.skip_synthetic else "\n" + taste_check())
    open(os.path.join(ROOT, f"results/{a.out}.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
