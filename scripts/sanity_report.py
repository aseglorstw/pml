"""Evaluate the sanity suite and write results/sanity_report.md (explicit PASS/FAIL per check).

Pre-registered rules (test NDCG@20 at the best val checkpoint, synthetic seed 0):
  a  S(Z) - L >= 0.005  and  L < S(A) < S(Z)          (L = LightGCN warm, S = LightGCN-S warm)
  b  S(A + 50% random Omega edges) < S(A)
  c  for every kappa <= kappa* = eps/p_A (true, on Omega): c_hat <= c, pi_hat >= pi, rho_obs_hat >= rho_obs,
     rho_un_hat >= rho_un  (noise-corrected Elkan-Noto, Proposition 3.1)
  d  AUROC(alpha: true vs false observed edges) >= 0.6 for PU-GIB (beta=0.001), and recall of hidden true
     edges (U_rank ∩ Z) in the s pool >= 2 x the recall of a random pool of the same size
  e  PU-GIB (beta=0.001) > S(A)  and  GBSR > S(A)
  LastFM regression: S(oracle test-train) - L >= 0.005
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import anchor_estimates, channel_estimates, isin_pairs  # noqa: E402

SYN = os.path.join(ROOT, "data/processed/synth")
lines, verdicts = [], []


def runs():
    fs = glob.glob(os.path.join(ROOT, "results/jobs/local_sanity_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).sort_values("timestamp")
    return df.groupby("method").tail(1).set_index("method")


def verdict(name, ok, detail):
    verdicts.append((name, "PASS" if ok else "FAIL", detail))


def table(rows, cols):
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(f(r.get(c, "–")) for c in cols) + " |" for r in rows])


R = runs()
N = lambda m: float(R.loc[m, "test_ndcg@20"]) if m in R.index else float("nan")
truth = json.load(open(os.path.join(SYN, "truth/truth.json")))
T = truth["omega"]
Z = np.loadtxt(os.path.join(SYN, "truth/Z.txt"), dtype=np.int64)
M = truth["params"]["users"]

# ---- run table
rows = []
for m in R.index:
    h = json.loads(R.loc[m, "hparams"])
    rows.append({"run": m, "dataset": R.loc[m, "dataset"], "best ep": int(R.loc[m, "best_epoch"]),
                 "val N@20": float(R.loc[m, "val_ndcg@20"]), "R@20": float(R.loc[m, "test_recall@20"]),
                 "N@20": float(R.loc[m, "test_ndcg@20"]),
                 "last-ep N@20": (h.get("final_test") or {}).get("ndcg@20", float("nan")),
                 "s/epoch": h.get("sec_per_epoch", float("nan"))})

# ---- a / b
L, SZ, SA, SAp = N("san_syn_lightgcn_warm"), N("san_syn_lgcns_Z"), N("san_syn_lgcns_A"), N("san_syn_lgcns_Aplus50")
verdict("a  social pathway: S(Z) beats L clearly", SZ - L >= 0.005, f"S(Z) {SZ:.4f} vs L {L:.4f} (Δ {SZ - L:+.4f}, need ≥ +0.005)")
verdict("a  S(A) between L and S(Z)", L < SA < SZ, f"L {L:.4f} < S(A) {SA:.4f} < S(Z) {SZ:.4f}")
verdict("b  noise sensitivity: S(A+50% random) < S(A)", SAp < SA, f"S(A+50%) {SAp:.4f} vs S(A) {SA:.4f} (Δ {SAp - SA:+.4f})")

# ---- c estimator vs truth on Omega
pri = json.load(open(os.path.join(SYN, "seed_0/pugib/best/priors.json")))
p_a, g_val, ks = pri["p_A"], pri["g_val"], T["kappa_star"]
g_top = pri.get("anchor_g_top", {}).get("g_top")
kappas = sorted({0.02, 0.05, 0.1, round(ks / 2, 5), round(ks, 5)})
crow, ok_c, n_checked = [], True, 0
for k in kappas:
    e = channel_estimates(p_a, g_val, k)
    an = anchor_estimates(p_a, g_val, g_top, k) if g_top is not None else {}
    cond = k <= ks + 1e-12
    dirs = (e["c_hat"] <= T["c"] + 1e-12, e["pi_hat"] >= T["pi"] - 1e-12,
            e["rho_obs"] >= T["rho_obs"] - 1e-12, e["rho_un"] >= T["rho_un"] - 1e-12)
    if cond:
        n_checked += 1
        ok_c &= all(dirs)
    crow.append({"kappa": k, "κ ≤ κ*": cond, "c_hat EN": e["c_hat"], "pi_hat EN": e["pi_hat"],
                 "rho_obs EN": e["rho_obs"], "rho_un EN": e["rho_un"],
                 "directions (c≤, π≥, ρo≥, ρu≥)": "".join("✓" if d else "✗" for d in dirs),
                 "c_hat anchor": an.get("c_hat", float("nan")), "rho_un anchor": an.get("rho_un", float("nan"))})
verdict("c  estimator error direction (Prop. 3.1, κ ≤ κ*)", ok_c and n_checked > 0,
        f"{n_checked} κ values ≤ κ* = {ks:.4f}; true c {T['c']:.3f}, π {T['pi']:.3f}, ρ_obs {T['rho_obs']:.3f}, ρ_un {T['rho_un']:.4f}")

# ---- d masker AUROC and pool recall
def masker_auc(run):
    p = os.path.join(ROOT, "results/pugib_runs", run, "alpha_seed0.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    e, a, obs = np.sort(z["edges"].astype(np.int64), 1), z["alpha"], z["is_obs"].astype(bool)
    y = isin_pairs(e[obs], Z, M)
    return {"auc": float(roc_auc_score(y, a[obs])), "alpha_true": float(a[obs][y].mean()),
            "alpha_false": float(a[obs][~y].mean()), "n_true": int(y.sum()), "n_false": int((~y).sum()),
            "alpha_cand": float(a[~obs].mean()) if (~obs).any() else float("nan")}


mrows = []
for run in ["san_syn_pugib_b0001", "san_syn_pugib_b001", "san_syn_gbsr"]:
    r = masker_auc(run)
    if r:
        mrows.append({"run": run, "AUROC α (true vs false obs.)": r["auc"], "mean α true obs.": r["alpha_true"],
                      "mean α false obs.": r["alpha_false"], "mean α candidates": r["alpha_cand"],
                      "#true / #false obs.": f"{r['n_true']} / {r['n_false']}"})
main = masker_auc("san_syn_pugib_b0001")
om = np.load(os.path.join(SYN, "seed_0/pugib/omega.npz"))
pairs, yom = om["pairs"].astype(np.int64), om["y"]
u_rank = pairs[~yom]
hidden = u_rank[isin_pairs(u_rank, Z, M)]
prow = {}
for name, f in [("s", "best/pool_s.txt"), ("RA", "pool_ra.txt"), ("random", "best/pool_random.txt")]:
    pool = np.loadtxt(os.path.join(SYN, "seed_0/pugib", f), dtype=np.int64, ndmin=2)
    prow[name] = {"pool": name, "size": len(pool), "recall hidden true": float(isin_pairs(hidden, pool, M).mean()),
                  "random-pool expectation": len(pool) / len(u_rank)}
lift = prow["s"]["recall hidden true"] / prow["s"]["random-pool expectation"]
verdict("d  masker: α(false obs.) < α(true obs.)", main is not None and main["auc"] >= 0.6,
        f"AUROC {main['auc']:.3f} (need ≥ 0.6)" if main else "no alpha snapshot")
verdict("d  pool recall of hidden true edges ≫ random", lift >= 2,
        f"s pool recall {prow['s']['recall hidden true']:.3f} vs random {prow['s']['random-pool expectation']:.3f} (×{lift:.1f}, need ≥ 2)")

# ---- e payoff
PG, GB = N("san_syn_pugib_b0001"), N("san_syn_gbsr")
verdict("e  PU-GIB on A beats S(A)", PG > SA, f"PU-GIB {PG:.4f} vs S(A) {SA:.4f} (Δ {PG - SA:+.4f})")
verdict("e  GBSR on A beats S(A)", GB > SA, f"GBSR {GB:.4f} vs S(A) {SA:.4f} (Δ {GB - SA:+.4f})")

# ---- LastFM
LL, LS, LO, LG = (N(m) for m in ["san_lf_lightgcn_warm", "san_lf_lgcns_A", "san_lf_lgcns_oracle", "san_lf_gbsr"])
verdict("LastFM regression: oracle test-train beats L", LO - LL >= 0.005, f"oracle {LO:.4f} vs L {LL:.4f} (Δ {LO - LL:+.4f})")

out = ["# Sanity report (social_form = gbsr everywhere)", "",
       "| check | result | numbers |", "|---|---|---|"] + [f"| {n} | **{v}** | {d} |" for n, v, d in verdicts]
out += ["", f"LastFM (info): GBSR {LG:.4f}, LightGCN-S {LS:.4f}, LightGCN warm {LL:.4f} → "
        f"GBSR − LightGCN-S {LG - LS:+.4f}, GBSR − LightGCN {LG - LL:+.4f}", "",
        "## Synthetic world", "", "```", json.dumps({k: truth[k] for k in
        ["n_Z", "n_A", "n_A_true", "n_A_false", "false_frac_observed", "c_global", "eps_global"]}), json.dumps(T), "```", "",
        "## Runs (test metrics at the best val checkpoint)", "",
        table(rows, ["run", "dataset", "best ep", "val N@20", "R@20", "N@20", "last-ep N@20", "s/epoch"]), "",
        "## c. Estimator vs truth on Omega", "", f"p_A {p_a:.4f} (truth {T['p_A']:.4f}), g_val {g_val:.4f} ± {pri['g_val_se']:.4f}, "
        f"g_top {g_top if g_top is None else round(g_top, 4)}, κ* = ε/p_A = {ks:.4f}", "",
        table(crow, list(crow[0].keys())), "",
        "## d. Masker and candidate pools", "", table(mrows, list(mrows[0].keys())) if mrows else "no snapshots", "",
        f"Hidden true edges in U_rank: {len(hidden)} of {len(u_rank)}", "",
        table(list(prow.values()), ["pool", "size", "recall hidden true", "random-pool expectation"])]
text = "\n".join(out)
open(os.path.join(ROOT, "results/sanity_report.md"), "w").write(text + "\n")
print(text)
