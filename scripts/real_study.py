"""Mixture-prior denoising on a real dataset (Douban-Book, Epinions, Yelp): GBSR on the MAP graph vs GBSR.

Needs results/checkpoints/lightgcn_<ds>_seed<k>.pt (interaction-only LightGCN, configs/<ds>_lightgcn.yaml).

  python scripts/real_study.py prep   --dataset epinions --seeds 0   # mixture prior + MAP graph, SHaRe graphs
  python scripts/real_study.py jobs   --dataset epinions --seeds 0   # writes configs/jobs/real_<ds>.txt
  python scripts/real_study.py report --dataset epinions --seeds 0   # table -> stdout, results/real_<ds>_report.md

Runs (omega 1, GBSR's HSIC weight and kernel width for the dataset, paper protocol):
  GBSR on A | GBSR on MAP(prior) | GBSR, SHaRe cut only | GBSR, SHaRe rewiring + weights;
  the interaction-only LightGCN (no social graph) is the reference run real_<ds>_lightgcn.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import adjacency  # noqa: E402

# GBSR README: beta (our hsic_lambda) and sigma per dataset
HSIC = {"douban_book": (40, 2.5), "epinions": (3, 0.25), "yelp": (2, 0.25)}
OMEGAS = [1]


def sdir(ds, s):
    return f"data/processed/{ds}/seed_{s}"


def share_graphs(ds, s, block=512):
    """SHaRe rewiring (Jiang et al., WWW 2024, Sec. 3.1) with a blockwise top-M search over all user pairs."""
    import torch
    d = os.path.join(ROOT, sdir(ds, s))
    e = np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64)
    ck = os.path.join(ROOT, "results/checkpoints", f"lightgcn_{ds}_seed{s}.pt")
    emb = torch.load(ck, map_location="cpu")["user_final"].numpy().astype(np.float32)
    n = emb.shape[0]
    z = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    c_obs = np.einsum("ij,ij->i", z[e[:, 0]], z[e[:, 1]])
    cut = c_obs <= 0
    remain = e[~cut]
    M = int(cut.sum())
    A = adjacency(e, n).tocsr()
    best_v, best_i, best_j = np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
    for b0 in range(0, n, block):
        rows = np.arange(b0, min(b0 + block, n))
        C = z[rows] @ z.T
        C[np.arange(len(rows))[:, None] >= (np.arange(n)[None, :] - b0 + 0)] = -np.inf  # keep j > i only
        sub = A[rows].tocoo()
        C[sub.row, sub.col] = -np.inf  # unlinked pairs only
        flat = C.ravel()
        k = min(M, flat.size)
        if k == 0:
            continue
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.isfinite(flat[top])]
        best_v = np.concatenate([best_v, flat[top]])
        best_i = np.concatenate([best_i, rows[top // n]])
        best_j = np.concatenate([best_j, top % n])
        if len(best_v) > 4 * M:
            keep = np.argpartition(-best_v, M - 1)[:M]
            best_v, best_i, best_j = best_v[keep], best_i[keep], best_j[keep]
    if M > 0:
        keep = np.argpartition(-best_v, M - 1)[:M]
        add = np.stack([best_i[keep], best_j[keep]], 1)
    else:
        add = np.zeros((0, 2), np.int64)
    rew = np.concatenate([remain, add])
    c = np.einsum("ij,ij->i", z[rew[:, 0]], z[rew[:, 1]])
    w = (c - c.min()) / max(c.max() - c.min(), 1e-12)
    out = os.path.join(d, "pugib")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "share_prune.txt"), remain, fmt="%d")
    np.savetxt(os.path.join(out, "share_edges.txt"), rew, fmt="%d")
    np.savetxt(os.path.join(out, "share_w.txt"), w, fmt="%.6f")
    return {"ties": int(len(e)), "share_cut": M, "share_added": int(len(add)), "share_mean_weight": float(w.mean())}


def targeted_add(ds, s, k=1, max_deg=2):
    """Smart addition for users with 1..max_deg training ties: the structure proposes, the taste prior verifies.
    For each such user the top-k two-hop candidates by Resource Allocation (computed on the training graph) are
    proposed; a candidate is added to the MAP graph only if its mixture posterior rho(s) >= 0.5, with the noise
    share and density ratio of the observed ties inside Omega (candidates share a friend by construction)."""
    import scipy.sparse as sp
    import torch
    from sklearn.isotonic import IsotonicRegression
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from mixture_prior import noise_share  # noqa: E402
    from src.pugib_prep import isin_pairs  # noqa: E402
    d = os.path.join(ROOT, sdir(ds, s))
    e = np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64)
    emb = torch.load(os.path.join(ROOT, "results/checkpoints", f"lightgcn_{ds}_seed{s}.pt"),
                     map_location="cpu")["user_final"].numpy()
    n = emb.shape[0]
    z = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    A = adjacency(e, n).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    W = (A @ sp.diags(1 / np.maximum(deg, 1)) @ A).tocsr()
    tgt = np.where((deg >= 1) & (deg <= max_deg))[0]
    cand = []
    for u in tgt:
        row = W[u].toarray().ravel()
        row[u] = 0
        row[A[u].indices] = 0
        nz = np.flatnonzero(row)
        if len(nz):
            cand += [(min(u, x), max(u, x)) for x in nz[np.argsort(-row[nz])[:k]]]
    cand = np.unique(np.array(cand, dtype=np.int64).reshape(-1, 2), axis=0)
    # posterior of the in-Omega group, fitted as in scripts/mixture_prior.py
    mp = np.load(os.path.join(d, "pugib/mixture_prior.npz"))
    inom = mp["in_omega"]
    s_obs = np.einsum("ij,ij->i", z[e[inom, 0]], z[e[inom, 1]])
    rng = np.random.default_rng([s, 7])
    r = np.sort(rng.integers(0, n, (200000, 2)), 1)
    r = r[r[:, 0] != r[:, 1]]
    r = r[~isin_pairs(r, e, n)]
    s_rand = np.einsum("ij,ij->i", z[r[:, 0]], z[r[:, 1]])
    q, _ = noise_share(s_obs, s_rand)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=1e-9, y_max=1 - 1e-9).fit(
        np.concatenate([s_obs, s_rand]), np.concatenate([np.ones(len(s_obs)), np.zeros(len(s_rand))]))
    s_c = np.einsum("ij,ij->i", z[cand[:, 0]], z[cand[:, 1]]) if len(cand) else np.zeros(0)
    h = iso.predict(s_c) if len(cand) else np.zeros(0)
    post = np.clip(1 - q * (1 - h) / h * len(s_obs) / len(s_rand), 0, 1)
    acc = cand[post >= 0.5]
    mapg = np.loadtxt(os.path.join(d, "pugib/map_prior.txt"), dtype=np.int64)
    np.savetxt(os.path.join(d, "pugib/map_add.txt"), np.concatenate([mapg, acc]), fmt="%d")
    v = np.loadtxt(os.path.join(d, "social_val.txt"), dtype=np.int64)
    hit = lambda x: int(isin_pairs(x, np.sort(v, 1), n).sum()) if len(x) else 0
    return {"target_users": int(len(tgt)), "proposed": int(len(cand)), "accepted": int(len(acc)),
            "proposed_heldout_hits": hit(cand), "accepted_heldout_hits": hit(acc)}


def prep(ds, seeds):
    for s in seeds:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts/mixture_prior.py"), "--dataset", ds, "--seeds", str(s)],
                       check=True, stdout=subprocess.DEVNULL)
        info = json.load(open(os.path.join(ROOT, sdir(ds, s), "pugib/mixture_prior.json")))
        sh = share_graphs(ds, s)
        ad = targeted_add(ds, s)
        json.dump(ad, open(os.path.join(ROOT, sdir(ds, s), "pugib/map_add.json"), "w"), indent=2)
        g = info["groups"]
        print(f"{ds} seed {s}: ties {info['n_ties']} | noise share q̂: in Ω {g.get('in_omega', {}).get('q_hat', float('nan')):.3f} "
              f"(n {g.get('in_omega', {}).get('n', 0)}), bridges {g.get('bridges', {}).get('q_hat', float('nan')):.3f} "
              f"(n {g.get('bridges', {}).get('n', 0)}) | MAP keeps {info['map_kept']} | SHaRe cuts {sh['share_cut']}, "
              f"adds {sh['share_added']}, mean weight {sh['share_mean_weight']:.3f} | targeted addition: "
              f"{ad['target_users']} users with 1-2 ties, proposed {ad['proposed']} (held-out hits "
              f"{ad['proposed_heldout_hits']}), accepted {ad['accepted']} (hits {ad['accepted_heldout_hits']})")


def line(cfg, name, ds, s, extra):
    return f"- configs/paper/{cfg}.yaml --no-tune --seeds {s} --set dataset={ds} data_dir=data/processed/{ds} name={name} {extra}"


def jobs(ds, seeds):
    lam, sig = HSIC[ds]
    h = f"hsic_lambda={lam} hsic_sigma={sig}"
    out = [f"# generated by scripts/real_study.py jobs --dataset {ds}"]
    for s in seeds:
        d = sdir(ds, s)
        graphs = {"A": "", "map": f"social_edges_file={d}/pugib/map_prior.txt",
                  "mapadd": f"social_edges_file={d}/pugib/map_add.txt",
                  "shp": f"social_edges_file={d}/pugib/share_prune.txt",
                  "sh": f"social_edges_file={d}/pugib/share_edges.txt social_weights_file={d}/pugib/share_w.txt"}
        for tag, g in graphs.items():
            for om in OMEGAS:
                wt = str(om).replace(".", "")
                name = f"real_{ds}_gbsr_{tag}_w{wt}"
                if os.path.exists(os.path.join(ROOT, f"results/peruser/{ds}/{name}/seed{s}.npz")):
                    continue  # already finished
                out.append(line("gbsr", name, ds, s, f"social_omega={om} {g} {h}".strip()))
    path = os.path.join(ROOT, f"configs/jobs/real_{ds}.txt")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote {path}: {len(out) - 1} jobs")


def peruser(ds, name, s):
    p = os.path.join(ROOT, f"results/peruser/{ds}/{name}/seed{s}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(zip(z["users"].tolist(), z["ndcg"].tolist()))


def report(ds, seeds):
    rows = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/jobs/local_real_*.csv"))) + [os.path.join(ROOT, "results/results.csv")]:
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                rows[(r["method"], int(r["seed"]))] = r
    ref = f"real_{ds}_gbsr_A_w1"
    lines = [f"# {ds}: GBSR on the MAP graph vs GBSR (test, best-validation checkpoint)", "",
             "Δ = paired per-user NDCG@20 difference vs GBSR on A, ± SE over users; 'users 1-2 ties' = test users with",
             "1-2 training ties (the group that receives targeted additions).", "",
             "| method | seed | R@20 | N@20 | Δ N@20 vs GBSR(A) | Δ, users 1-2 ties | best epoch |",
             "|---|---|---|---|---|---|---|"]
    low = {}
    for s in seeds:
        e = np.loadtxt(os.path.join(ROOT, sdir(ds, s), "social_train.txt"), dtype=np.int64)
        dg = np.bincount(e.ravel())
        low[s] = set(np.flatnonzero((dg >= 1) & (dg <= 2)).tolist())
    summ = {}
    for lab, nm in (("LightGCN (no social graph)", f"real_{ds}_lightgcn"), ("GBSR on A", ref),
                    ("GBSR on MAP(prior) — ours", f"real_{ds}_gbsr_map_w1"),
                    ("GBSR on MAP(prior) + targeted addition — ours", f"real_{ds}_gbsr_mapadd_w1"),
                    ("GBSR, SHaRe cut only (ablation)", f"real_{ds}_gbsr_shp_w1"),
                    ("GBSR, SHaRe rewiring + weights", f"real_{ds}_gbsr_sh_w1")):
        for s in seeds:
            r = rows.get((nm, s))
            A, R = peruser(ds, nm, s), peruser(ds, ref, s)
            delta, dlow = "–", "–"
            if A is not None and R is not None:
                u = sorted(set(A) & set(R))
                dd = np.array([A[x] - R[x] for x in u])
                delta = f"{dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.4f}"
                summ.setdefault(lab, []).append(dd.mean())
                ul = [x for x in u if x in low[s]]
                if len(ul) > 1:
                    dl = np.array([A[x] - R[x] for x in ul])
                    dlow = f"{dl.mean():+.4f} ± {dl.std(ddof=1) / np.sqrt(len(dl)):.4f} (n {len(dl)})"
            if r is None:
                lines.append(f"| {lab} | {s} | – | – | {delta} | {dlow} | – |")
            else:
                lines.append(f"| {lab} | {s} | {float(r['test_recall@20']):.4f} | {float(r['test_ndcg@20']):.4f} | "
                             f"{delta} | {dlow} | {r['best_epoch']} |")
    if len(seeds) > 1:
        lines += ["", "Mean Δ N@20 vs GBSR(A) over seeds (± std, t = mean / (std/√n)):", ""]
        for lab, ms in summ.items():
            if len(ms) == len(seeds) and np.std(ms) > 0:
                sd = np.std(ms, ddof=1)
                lines.append(f"- {lab}: {' / '.join(f'{m:+.4f}' for m in ms)} → {np.mean(ms):+.4f} ± {sd:.4f}, "
                             f"t {np.mean(ms) / (sd / np.sqrt(len(ms))):.1f}")
    for s in seeds:
        p = os.path.join(ROOT, sdir(ds, s), "pugib/mixture_prior.json")
        if os.path.exists(p):
            info = json.load(open(p))
            g = info["groups"]
            lines.append("")
            lines.append(f"Seed {s}: {info['n_ties']} training ties; noise share q̂ in Ω "
                         f"{g.get('in_omega', {}).get('q_hat', float('nan')):.3f}, bridges {g.get('bridges', {}).get('q_hat', float('nan')):.3f}; "
                         f"MAP graph keeps {info['map_kept']}.")
            pa = os.path.join(ROOT, sdir(ds, s), "pugib/map_add.json")
            if os.path.exists(pa):
                ad = json.load(open(pa))
                lines.append(f"Targeted addition: {ad['target_users']} users with 1-2 ties, proposed {ad['proposed']} "
                             f"(held-out hits {ad['proposed_heldout_hits']}), accepted {ad['accepted']} "
                             f"(held-out hits {ad['accepted_heldout_hits']}).")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(ROOT, f"results/real_{ds}_report.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "jobs", "report"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    {"prep": prep, "jobs": jobs, "report": report}[a.cmd](a.dataset, a.seeds)
