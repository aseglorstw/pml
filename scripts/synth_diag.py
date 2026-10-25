"""Synthetic diagnostics: decomposition table (seed 0), selection of best K / beta by VALIDATION NDCG@20,
job list for the 3-seed paper table, and the final report (results/synth_diag_report.md).

  python scripts/synth_diag.py select   # -> configs/jobs/synth_diag4.txt (seeds 1, 2; seed 0 runs are reused)
  python scripts/synth_diag.py report   # -> results/synth_diag_report.md
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
KAPPA = 0.02
PART1 = ["san_syn_lightgcn_warm", "san_syn_lgcns_A", "san_syn_gbsr",
         "dg_pu_k2", "dg_pu_k5", "dg_pu_k10",
         "dg_pugib_nocand_b0001", "dg_pugib_nocand_b001",
         "dg_pugib_k2_b0001", "dg_pugib_k5_b0001", "dg_pugib_k10_b0001", "san_syn_pugib_b001"]


def runs():
    fs = glob.glob(os.path.join(ROOT, "results/jobs/local_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[df["dataset"] == "synth"].sort_values("timestamp")
    df = df[df["hparams"].map(lambda h: json.loads(h).get("stage", "final") == "final")]
    return df.groupby(["method", "seed"]).tail(1).set_index(["method", "seed"])


def truth(seed):
    t = json.load(open(os.path.join(SYN, f"seed_{seed}", "truth", "truth.json")))
    Z = np.loadtxt(os.path.join(SYN, f"seed_{seed}", "truth", "Z.txt"), dtype=np.int64)
    return t, Z


def pool_of(h, seed):
    """Candidate pool used by a run (None if no candidates)."""
    if h.get("method") not in ("pugib", "lightgcn_pu") or h.get("pool") == "none":
        return None
    k = h.get("pool_k")
    base = os.path.join(SYN, f"seed_{seed}", "pugib")
    f = {"s": f"best/pool_s_k{k}.txt" if k else "best/pool_s.txt", "ra": f"pool_ra_k{k}.txt" if k else "pool_ra.txt"}[h["pool"]]
    return np.loadtxt(os.path.join(base, f), dtype=np.int64, ndmin=2)


def alpha_stats(run, seed, Z, M):
    p = os.path.join(ROOT, "results/pugib_runs", run, f"alpha_seed{seed}.npz")
    if not os.path.exists(p):
        return {}
    z = np.load(p)
    e, a, obs = np.sort(z["edges"].astype(np.int64), 1), z["alpha"], z["is_obs"].astype(bool)
    tz = isin_pairs(e, Z, M)
    o, c = obs, ~obs
    out = {"AUROC obs": float(roc_auc_score(tz[o], a[o])) if 0 < tz[o].sum() < o.sum() else float("nan"),
           "α true obs": float(a[o & tz].mean()), "α false obs": float(a[o & ~tz].mean())}
    if c.any():
        out.update({"α true cand": float(a[c & tz].mean()) if (c & tz).any() else float("nan"),
                    "α false cand": float(a[c & ~tz].mean()) if (c & ~tz).any() else float("nan")})
    return out


def part1_rows(R):
    t, Z = truth(0)
    M = t["params"]["users"]
    rows = []
    for m in PART1:
        if (m, 0) not in R.index:
            rows.append({"run": m, "N@20": "missing"})
            continue
        r = R.loc[(m, 0)]
        h = json.loads(r["hparams"])
        pool = pool_of(h, 0)
        row = {"run": m, "K": h.get("pool_k", 10 if pool is not None else "–") if h.get("method") in ("pugib", "lightgcn_pu") else "–",
               "beta": h.get("beta", "–"), "val N@20": float(r["val_ndcg@20"]), "test N@20": float(r["test_ndcg@20"]),
               "pool size": len(pool) if pool is not None else "–",
               "pool precision": float(isin_pairs(pool, Z, M).mean()) if pool is not None and len(pool) else "–"}
        row.update(alpha_stats(m, 0, Z, M))
        rows.append(row)
    return rows


def select(R):
    def best(names):
        avail = [n for n in names if (n, 0) in R.index]
        return max(avail, key=lambda n: float(R.loc[(n, 0), "val_ndcg@20"]))
    pu = best(["dg_pu_k2", "dg_pu_k5", "dg_pu_k10"])
    nc = best(["dg_pugib_nocand_b0001", "dg_pugib_nocand_b001"])
    pg = best(["dg_pugib_k2_b0001", "dg_pugib_k5_b0001", "dg_pugib_k10_b0001", "san_syn_pugib_b001"])
    return {"lightgcn_pu": pu, "pugib_nocand": nc, "pugib": pg}


def overrides(run, R):
    h = json.loads(R.loc[(run, 0), "hparams"])
    keys = ["pool", "pool_k", "beta", "kappa", "norm", "prior"]
    return " ".join(f"{k}={h[k]}" for k in keys if k in h and h[k] is not None)


ROWS = [  # (label, seed-0 run, config for seeds 1-2, extra overrides)
    ("LightGCN warm", "san_syn_lightgcn_warm", "configs/pugib/lightgcn_warm.yaml", ""),
    ("LightGCN-S on A", "san_syn_lgcns_A", "configs/pugib/lightgcn_s_warm.yaml", ""),
    ("LightGCN-S on Z (oracle)", "san_syn_lgcns_Z", "configs/pugib/lightgcn_s_warm.yaml", "Z"),
    ("GBSR", "san_syn_gbsr", "configs/gbsr/lastfm_gbsr.yaml", "--no-tune beta=5 sigma=0.25 detach_mask=false"),
]


def paper_rows(sel, R):
    rows = list(ROWS)
    rows.append((f"LightGCN-PU (best K)", sel["lightgcn_pu"], "configs/pugib/lightgcn_pu_s.yaml", overrides(sel["lightgcn_pu"], R)))
    rows.append((f"PU-GIB, no candidates (best β)", sel["pugib_nocand"], "configs/pugib/pugib_s_anchor.yaml",
                 "--no-tune " + overrides(sel["pugib_nocand"], R)))
    rows.append((f"PU-GIB (best K, β)", sel["pugib"], "configs/pugib/pugib_s_anchor.yaml", "--no-tune " + overrides(sel["pugib"], R)))
    return rows


def cmd_select():
    R = runs()
    sel = select(R)
    lines = ["# 3-seed synthetic table, seeds 1 and 2 (seed 0 runs are reused). Generated by synth_diag.py select.",
             f"# selection by seed-0 val NDCG@20: {json.dumps(sel)}"]
    for label, run0, cfg, extra in paper_rows(sel, R):
        for s in (1, 2):
            tune = "--no-tune " if extra.startswith("--no-tune") else ""
            ex = extra.replace("--no-tune ", "").replace("--no-tune", "")
            if ex == "Z":
                ex = f"social_edges_file=data/processed/synth/seed_{s}/truth/Z.txt"
            lines.append(f"-  {cfg} {tune}--seeds {s} --set dataset=synth data_dir=data/processed/synth reg=0.0001 "
                         f"social_form=gbsr name={run0} {ex}".rstrip())
    open(os.path.join(ROOT, "configs/jobs/synth_diag4.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)


def table(rows, cols):
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(r.get(c, "–")) for c in cols) + " |" for r in rows])


def cmd_report():
    R = runs()
    out = ["# Synthetic diagnostics (social_form = gbsr; anchor prior, κ = 0.02, expected-degree normalization)", ""]
    p1 = part1_rows(R)
    cols = ["run", "K", "beta", "val N@20", "test N@20", "AUROC obs", "α true obs", "α false obs",
            "α true cand", "α false cand", "pool size", "pool precision"]
    out += ["## 1. Decomposition (seed 0)", "", table(p1, cols), "",
            "AUROC obs = AUROC of α for true vs false observed ties. GBSR row: noise-free gate (without +0.5 bias), "
            "averaged over both directions.", ""]
    sel = select(R)
    out += [f"Selected by seed-0 validation NDCG@20: {sel}", ""]
    # paper table
    trows = []
    for label, run0, *_ in paper_rows(sel, R):
        vals = {s: R.loc[(run0, s)] for s in (0, 1, 2) if (run0, s) in R.index}
        row = {"model": label, "seeds": len(vals)}
        for k, col in [("R@20", "test_recall@20"), ("N@20", "test_ndcg@20")]:
            x = np.array([float(v[col]) for v in vals.values()])
            row[k] = f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else (fmt(x[0]) if len(x) else "–")
        aucs = []
        for s in vals:
            t, Z = truth(s)
            st = alpha_stats(run0, s, Z, t["params"]["users"])
            if "AUROC obs" in st:
                aucs.append(st["AUROC obs"])
        row["mask AUROC obs"] = f"{np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}" if len(aucs) > 1 else (fmt(aucs[0]) if aucs else "–")
        trows.append(row)
    out += ["## 2. Paper table: synthetic seeds 0, 1, 2 (test, mean ± std)", "",
            table(trows, ["model", "seeds", "R@20", "N@20", "mask AUROC obs"]), ""]
    # estimator table
    erows, keys = [], ["c", "pi", "rho_obs", "rho_un"]
    vals = {k: {"true": [], "EN": [], "anchor": []} for k in keys}
    for s in (0, 1, 2):
        try:
            t, _ = truth(s)
            pri = json.load(open(os.path.join(SYN, f"seed_{s}", "pugib", "best", "priors.json")))
        except FileNotFoundError:
            continue
        T = t["omega"]
        en = channel_estimates(pri["p_A"], pri["g_val"], KAPPA)
        an = anchor_estimates(pri["p_A"], pri["g_val"], pri["anchor_g_top"]["g_top"], KAPPA)
        for k, ek in zip(keys, ["c_hat", "pi_hat", "rho_obs", "rho_un"]):
            vals[k]["true"].append(T[k])
            vals[k]["EN"].append(en[ek])
            vals[k]["anchor"].append(an[ek])
        erows.append({"seed": s, "κ*": T["kappa_star"], "p_A": T["p_A"], "g_val": pri["g_val"], "g_top": pri["anchor_g_top"]["g_top"]})
    est = [{"quantity": k, **{w: f"{np.mean(v[w]):.4f} ± {np.std(v[w], ddof=1):.4f}" for w in ("true", "EN", "anchor")}}
           for k, v in vals.items()]
    out += [f"## 3. Estimator vs truth on Ω (κ = {KAPPA}, mean ± std over seeds)", "",
            table(est, ["quantity", "true", "EN", "anchor"]), "", table(erows, ["seed", "κ*", "p_A", "g_val", "g_top"])]
    text = "\n".join(out)
    open(os.path.join(ROOT, "results/synth_diag_report.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    {"select": cmd_select, "report": cmd_report}[sys.argv[1]]()
