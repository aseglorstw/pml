"""CPU analyses of finished runs (seconds; run where results/peruser and data/processed/*/pugib are, i.e. the cluster).

1. Addition precision: share of proposed / accepted / rejected ties that are held-out ties (social_val), per proposer,
   against the chance level |held-out| / #unlinked pairs. Sums over seeds.
2. Users with 1-2 training ties: per-user NDCG@20 difference vs GBSR on A, averaged over the seeds in which the user is
   a test user with 1-2 ties, then mean ± SE over users (no-ties control = placebo for "the graph changed").
3. Who gains: per-user Δ(matched MAP − A) by interaction-count quartile and by share of the user's ties removed
   (bins fixed in advance: none, (0, 0.5], (0.5, 1), all).
4. Does the graph help: estimated noise share q̂ (all ties) vs graph utility (GBSR on A − GBSR without ties) and vs
   our gain (matched − A, matched − no ties). Noisy copies use the no-ties run of their base dataset (same split).
Writes results/cpu_analyses.md.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEEDS = [0, 1, 2]
REAL = ["douban_book", "epinions", "yelp"]
NOISY = ["epinions_n50", "epinions_fof50", "yelp_n50", "yelp_fof50"]


def base_of(ds):
    m = re.match(r"^(.*)_(n|fof|tri)\d+$", ds)
    return m.group(1) if m else ds


def sdir(ds, s):
    return os.path.join(ROOT, "data/processed", ds, f"seed_{s}")


def peruser(ds, name, s):
    p = os.path.join(ROOT, f"results/peruser/{ds}/{name}/seed{s}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(zip(z["users"].tolist(), z["ndcg"].tolist()))


def delta(ds, name, s, ref_ds=None, ref=None):
    A, R = peruser(ds, name, s), peruser(ref_ds or ds, ref or f"real_{ds}_gbsr_A_w1", s)
    if A is None or R is None:
        return None
    return {u: A[u] - R[u] for u in set(A) & set(R)}


def fmt(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return "–"
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:+.4f} ± {se:.4f} (t {m / se:.1f}, n {len(x)})" if se > 0 else f"{m:+.4f}"


def addition_precision(L):
    L += ["## 1. Addition precision (held-out ties among proposals; sums over seeds 0-2)", "",
          "| dataset | proposer | proposed | hits | precision | accepted | precision | rejected | precision | chance |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tags = {"": "RA top-1, check ρ ≥ 0.5 (paper)", "_ra3t50": "RA top-3, check ρ ≥ 0.5",
            "_ra3t00": "RA top-3, no check", "_s3t50": "taste s top-3, check ρ ≥ 0.5",
            "_s3t50m": "taste s top-3 on matched MAP, matched check"}
    for ds in REAL:
        chance = []
        for s in SEEDS:
            d = sdir(ds, s)
            if not os.path.exists(os.path.join(d, "social_val.txt")):
                continue
            n = json.load(open(os.path.join(d, "meta.json")))["n_users"]
            n_tr = len(np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64, ndmin=2))
            n_va = len(np.loadtxt(os.path.join(d, "social_val.txt"), dtype=np.int64, ndmin=2))
            chance.append(n_va / (n * (n - 1) / 2 - n_tr))
        for tag, lab in tags.items():
            P = H = Ac = HA = 0
            for s in SEEDS:
                f = os.path.join(sdir(ds, s), f"pugib/map_add{tag}.json")
                if not os.path.exists(f):
                    continue
                j = json.load(open(f))
                P, H, Ac, HA = P + j["proposed"], H + j["proposed_heldout_hits"], Ac + j["accepted"], \
                    HA + j["accepted_heldout_hits"]
            if P == 0:
                continue
            Rj, HR = P - Ac, H - HA
            L.append(f"| {ds} | {lab} | {P} | {H} | {H / P:.2%} | {Ac} | {HA / max(Ac, 1):.2%} | {Rj} | "
                     f"{HR / max(Rj, 1):.2%} | {np.mean(chance):.4%} |")
    L.append("")


def low_tie_users(L):
    L += ["## 2. Users with 1-2 training ties: Δ NDCG@20 vs GBSR on A (per user averaged over seeds, then over users)",
          "", "| dataset | run | Δ ± SE (t, users) |", "|---|---|---|"]
    runs = [("no social ties (placebo for any graph change)", "empty"), ("MAP", "map"), ("matched MAP", "mapm"),
            ("MAP + RA top-1 addition (paper)", "mapadd"), ("MAP + s top-3 addition", "add_s3t50"), ("matched MAP + s top-3 addition", "add_s3t50m"),
            ("A + s top-3 addition, no pruning", "add_s3t50A"), ("MAP + RA top-3 with check", "add_ra3t50"),
            ("MAP + RA top-3 without check", "add_ra3t00"), ("random removal, MAP count", "rand")]
    for ds in REAL:
        low = {}
        for s in SEEDS:
            f = os.path.join(sdir(ds, s), "social_train.txt")
            if os.path.exists(f):
                e = np.loadtxt(f, dtype=np.int64)
                dg = np.bincount(e.ravel())
                low[s] = set(np.flatnonzero((dg >= 1) & (dg <= 2)).tolist())
        for lab, tag in runs:
            acc = {}
            for s in SEEDS:
                dd = delta(ds, f"real_{ds}_gbsr_{tag}_w1", s)
                if dd is None or s not in low:
                    continue
                for u, v in dd.items():
                    if u in low[s]:
                        acc.setdefault(u, []).append(v)
            if acc:
                L.append(f"| {ds} | {lab} | {fmt([np.mean(v) for v in acc.values()])} |")
    L.append("")


def who_gains(L):
    L += ["## 3. Who gains: Δ(matched MAP − A) per user, averaged over seeds", "",
          "| dataset | group | Δ ± SE (t, users) |", "|---|---|---|"]
    for ds in ["epinions", "yelp", "douban_book"]:
        by_act, by_rem = {}, {}
        for s in SEEDS:
            dd = delta(ds, f"real_{ds}_gbsr_mapm_w1", s)
            d = sdir(ds, s)
            if dd is None or not os.path.exists(os.path.join(d, "pugib/map_prior_matched.txt")):
                continue
            tr = np.loadtxt(os.path.join(d, "train.txt"), dtype=np.int64)
            n = json.load(open(os.path.join(d, "meta.json")))["n_users"]
            act = np.bincount(tr[:, 0], minlength=n)
            qs = np.quantile(act[act > 0], [0.25, 0.5, 0.75])
            e = np.loadtxt(os.path.join(d, "social_train.txt"), dtype=np.int64)
            k = np.loadtxt(os.path.join(d, "pugib/map_prior_matched.txt"), dtype=np.int64, ndmin=2).reshape(-1, 2)
            dg, dk = np.bincount(e.ravel(), minlength=n), np.bincount(k.ravel(), minlength=n)
            for u, v in dd.items():
                qa = int(np.searchsorted(qs, act[u], side="right"))
                by_act.setdefault(f"interactions Q{qa + 1}", {}).setdefault(u, []).append(v)
                if dg[u] == 0:
                    g = "no ties"
                else:
                    r = 1 - dk[u] / dg[u]
                    g = "removed none" if r == 0 else ("removed ≤ 50%" if r <= 0.5 else
                                                       ("removed > 50%" if r < 1 else "removed all"))
                by_rem.setdefault(g, {}).setdefault(u, []).append(v)
        for grp in (by_act, by_rem):
            for g in sorted(grp):
                L.append(f"| {ds} | {g} | {fmt([np.mean(v) for v in grp[g].values()])} |")
    L.append("")


def graph_utility(L):
    L += ["## 4. Does the graph help? Estimated noise share vs graph utility (mean over seeds)", "",
          "q̂ = tie-weighted noise share over groups (uniform / matched reference). Utility = GBSR on A − GBSR without "
          "ties; gains are paired per-user differences averaged over users and seeds.", "",
          "| dataset | q̂ uniform | q̂ matched | MAP keeps | matched keeps | A − no ties | matched − A | "
          "matched − no ties |", "|---|---|---|---|---|---|---|---|"]
    for ds in REAL + NOISY:
        b = base_of(ds)
        qu, qm, ku, km, ut, ga, gb = [], [], [], [], [], [], []
        for s in SEEDS:
            d = sdir(ds, s)
            for tag, q, kk in (("", qu, ku), ("_matched", qm, km)):
                f = os.path.join(d, f"pugib/mixture_prior{tag}.json")
                if os.path.exists(f):
                    j = json.load(open(f))
                    g = j["groups"]
                    q.append(sum(v["q_hat"] * v["n"] for v in g.values()) / sum(v["n"] for v in g.values()))
                    kk.append(j["map_kept"] / j["n_ties"])
            ref_empty = (b, f"real_{b}_gbsr_empty_w1")
            A, E, M = (peruser(ds, f"real_{ds}_gbsr_A_w1", s), peruser(*ref_empty, s),
                       peruser(ds, f"real_{ds}_gbsr_mapm_w1", s))
            if A and E:
                ut.append(np.mean([A[u] - E[u] for u in set(A) & set(E)]))
            if A and M:
                ga.append(np.mean([M[u] - A[u] for u in set(A) & set(M)]))
            if M and E:
                gb.append(np.mean([M[u] - E[u] for u in set(M) & set(E)]))
        m = lambda x, f="{:.3f}": f.format(np.mean(x)) if x else "–"  # noqa: E731
        L.append(f"| {ds} | {m(qu)} | {m(qm)} | {m(ku, '{:.0%}')} | {m(km, '{:.0%}')} | {m(ut, '{:+.4f}')} | "
                 f"{m(ga, '{:+.4f}')} | {m(gb, '{:+.4f}')} |")
    L.append("")


def main():
    L = ["# CPU analyses (scripts/cpu_analyses.py)", ""]
    for f in (addition_precision, low_tie_users, who_gains, graph_utility):
        try:
            f(L)
        except Exception as ex:  # keep the other sections
            L += [f"{f.__name__} failed: {ex!r}", ""]
    out = os.path.join(ROOT, "results/cpu_analyses.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
