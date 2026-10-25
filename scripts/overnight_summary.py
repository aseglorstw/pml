"""Write results/overnight_summary.md (Douban warm comparison, Douban preprocessing, tonight's runs,
alpha diagnostics, LastFM diagnostics 1-3, failed jobs, GBSR code citations).

Usage: python scripts/overnight_summary.py [--since 2026-10-25T00:00]
"""
import argparse
import glob
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
M = ["test_recall@10", "test_recall@20", "test_ndcg@10", "test_ndcg@20"]


def j(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, n=4):
    return "–" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def section_douban_warm(df):
    d = df[(df["dataset"] == "douban_book") & (df["stage"] == "final") & (df["seed"] == 0)]
    d = d.groupby("method").tail(1).set_index("method")
    rows = []
    for m in ["lightgcn", "lightgcn_warm", "lightgcn_s_warm"]:
        if m in d.index:
            r = d.loc[m]
            rows.append({"method": m, "best_epoch": int(r["best_epoch"]), "val_ndcg@20": r["val_ndcg@20"],
                         **{k: r[k] for k in M}})
    s = ["## 1. Douban-Book warm-start comparison (seed 0, L2 = 1e-4)", "", table(rows, ["method", "best_epoch", "val_ndcg@20"] + M)]
    if {"lightgcn_warm", "lightgcn_s_warm"} <= set(d.index):
        diff = d.loc["lightgcn_s_warm", "test_ndcg@20"] - d.loc["lightgcn_warm", "test_ndcg@20"]
        dv = d.loc["lightgcn_s_warm", "val_ndcg@20"] - d.loc["lightgcn_warm", "val_ndcg@20"]
        verdict = "YES, clearly (> 0.002)" if diff > 0.002 else ("slightly" if diff > 0 else "NO")
        s += ["", f"LightGCN-S warm − LightGCN warm: test NDCG@20 {diff:+.4f}, val NDCG@20 {dv:+.4f}. "
                  f"Does LightGCN-S beat LightGCN under the same warm start? **{verdict}**."]
    else:
        s += ["", "Warm runs missing (see failed jobs)."]
    return "\n".join(s)


def prep_numbers(rep, seed="0"):
    if not rep or seed not in rep or "best" not in rep[seed]["variants"]:
        return None
    g, v = rep[seed]["graph"], rep[seed]["variants"]["best"]
    p = v["priors"]
    out = {"|Omega|": g["omega"], "p_A": g["p_A"], "train bridges": g["train_bridges"],
           "held-out bridges": g["heldout_bridges"], "|P_val|": g["p_val"], "base rate": g["base_rate"],
           "g_val": p["g_val"], "g_val SE": p["g_val_se"], "informative": p["informative"],
           "AUROC s (P_val vs U_rank)": v["similarity"]["auroc_s_pval_vs_urank"],
           "AUROC RA (P_val vs U_rank)": g.get("ra_auroc_pval_vs_urank"),
           "g_top": p.get("anchor_g_top", {}).get("g_top")}
    for k in ["0.02", "0.05", "0.1"]:
        e, a, u = p["estimated"][k], p.get("anchor", {}).get(k, {}), p["uninformed"][k]
        out[f"κ={k}: ĉ EN / anchor"] = f"{e['c_hat']:.4f} / {a.get('c_hat', float('nan')):.4f}"
        out[f"κ={k}: π̂ EN / anchor / unif"] = f"{e['pi_hat']:.3f} / {a.get('pi_hat', float('nan')):.3f} / {u['pi_hat']:.3f}"
        out[f"κ={k}: ρ_obs EN / anchor / unif"] = f"{e['rho_obs']:.3f} / {a.get('rho_obs', float('nan')):.3f} / {u['rho_obs']:.3f}"
        out[f"κ={k}: ρ_un EN / anchor / unif"] = f"{e['rho_un']:.4f} / {a.get('rho_un', float('nan')):.4f} / {u['rho_un']:.4f}"
    for pool in ["s", "ra", "random"]:
        st = v["pools"][pool]
        out[f"pool {pool}: size / recall / lift"] = f"{st['size']} / {st['recall']:.4f} / {st['lift']:.2f}"
        out[f"pool {pool}: |C| / C degree median, max"] = (f"{st['support_size']} / {st['support_degree']['median']:.0f}, "
                                                           f"{st['support_degree']['max']}")
    return out


def section_prep():
    db = prep_numbers(j("results/pugib_prep_summary_douban_book.json"))
    lf = prep_numbers(j("results/pugib_prep_summary.json"))
    s = ["## 2. PU-GIB preprocessing, seed 0, best-checkpoint embeddings: Douban-Book vs LastFM", ""]
    if db is None:
        return "\n".join(s + ["Douban-Book preprocessing not available (see failed jobs)."])
    s += ["| quantity | Douban-Book | LastFM |", "|---|---|---|"]
    s += [f"| {k} | {fmt(v)} | {fmt((lf or {}).get(k))} |" for k, v in db.items()]
    return "\n".join(s)


def section_runs(df, since):
    t = df[(df["timestamp"] >= since) & (df["stage"] == "final")].copy()
    t["s/epoch"] = t["hp"].map(lambda h: h.get("sec_per_epoch"))
    t["device"] = t["hp"].map(lambda h: h.get("device"))
    rows = [{"method": r["method"], "dataset": r["dataset"], "seed": r["seed"], "best_epoch": r["best_epoch"],
             "val_ndcg@20": r["val_ndcg@20"], **{k: r[k] for k in M}, "s/epoch": r["s/epoch"], "device": r["device"],
             "runtime_s": r["runtime_s"]} for _, r in t.iterrows()]
    return "\n".join(["## 3. Test metrics of all runs since " + since, "",
                      table(rows, ["method", "dataset", "seed", "best_epoch", "val_ndcg@20"] + M + ["s/epoch", "device", "runtime_s"])])


def section_alpha(since_ts):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/pugib_runs/*/final_seed*.json"))):
        if os.path.getmtime(f) < since_ts:
            continue
        r = json.load(open(f))
        d = r.get("best_diag") or {}
        ex = r.get("extra") or {}
        rows.append({"run": os.path.basename(os.path.dirname(f)), "dataset": r["cfg"].get("dataset"), "seed": r["seed"],
                     "rho_obs": ex.get("rho_obs") if r["cfg"].get("method") == "pugib" else None,
                     "rho_un": ex.get("rho_un") if r["cfg"].get("method") == "pugib" else None,
                     "α obs mean": d.get("alpha_mean_obs", d.get("gate_mean")), "α obs std": d.get("alpha_std_obs", d.get("gate_std")),
                     "α cand mean": d.get("alpha_mean_cand"), "α cand std": d.get("alpha_std_cand"),
                     "α min": d.get("alpha_min", d.get("gate_min")), "α max": d.get("alpha_max", d.get("gate_max")),
                     "cand α>0.5": d.get("frac_cand_gt05"), "P_val post. recall": d.get("posterior_recall_pval")})
    cols = ["run", "dataset", "seed", "rho_obs", "rho_un", "α obs mean", "α obs std", "α cand mean", "α cand std",
            "α min", "α max", "cand α>0.5", "P_val post. recall"]
    return "\n".join(["## 4. Alpha diagnostics at the best epoch (runs since tonight)", "",
                      "GBSR rows: noise-free gate without the +0.5 edge bias, averaged over both directions, all observed edges.", "",
                      table(rows, cols) if rows else "none"])


def section_lastfm_diag(df):
    s = ["## 5. LastFM diagnostics (seed 0)", ""]
    for name, title in [("grad_beta0", "5.1 Gradient check, β = 0 (50 epochs)"),
                        ("grad_best", "5.2 Gradient scale at the best config (β = 0.01, κ = 0.1, prior norm)")]:
        g = j(f"results/diagnostics/{name}.json")
        s += [f"### {title}", ""]
        if not g:
            s += ["missing", ""]
            continue
        s += [f"Connectivity checks: `{g['checks']}`", ""]
        rows = []
        for h in g["history"]:
            if h["epoch"] in (1, 2, 5, 10, 20, 30, 40, 50):
                rows.append({k: h.get(k) for k in ["epoch", "tau", "bpr_alpha_meanabs", "bpr_alpha_l2", "kl_alpha_meanabs",
                                                   "bpr_phi_l2", "kl_phi_l2", "bpr_alpha_snr_median", "alpha_std",
                                                   "alpha_min", "alpha_max", "alpha_mean_obs", "alpha_mean_cand"]})
        cols = list(rows[0].keys()) if rows else []
        s += [table([{k: (f"{v:.3e}" if isinstance(v, float) and abs(v) < 1e-2 and v != 0 else v) for k, v in r.items()}
                     for r in rows], cols), "", f"val at end: {g['val_end']}", ""]
    s += ["### 5.3 Social channel under 40% injected random edges (warm LightGCN-S)", ""]
    c = j("results/diagnostics/corruption_check.json")
    lf = df[(df["dataset"] == "lastfm") & (df["stage"] == "final")].groupby(["method", "seed"]).tail(1)
    rows = []
    for m in ["lightgcn_warm", "lightgcn_s_warm", "lightgcn_s_warm_inject40", "lightgcn_s_warm_replace100"]:
        for _, r in lf[lf["method"] == m].sort_values("seed").iterrows():
            rows.append({"method": m, "seed": r["seed"], "val_ndcg@20": r["val_ndcg@20"], "test_ndcg@20": r["test_ndcg@20"],
                         "test_recall@20": r["test_recall@20"]})
    s += [table(rows, ["method", "seed", "val_ndcg@20", "test_ndcg@20", "test_recall@20"]), ""]
    if c:
        s += [f"Mean drop clean − inject40 over seeds {c['seeds']}: test NDCG@20 {c['mean_drop_test']:+.4f}, "
              f"val {c['mean_drop_val']:+.4f}. Exceeds 0.002: **{c['drop_exceeds_threshold']}**."]
    au = os.path.join(ROOT, "results/diagnostics/auroc_injected.jsonl")
    if os.path.exists(au):
        s += ["", "AUROC of 1 − α for injected vs genuine observed edges:", ""]
        s += ["- " + line.strip() for line in open(au)]
    return "\n".join(s)


def section_failures():
    p = os.path.join(ROOT, "results/overnight_status.log")
    if not os.path.exists(p):
        return "## 6. Queue status\n\nno status log"
    lines = open(p).read().splitlines()
    bad = [x for x in lines if "FAILED" in x or "SKIPPED" in x]
    return "\n".join(["## 6. Failed or skipped jobs", ""] + (["- " + x for x in bad] or ["none"]) +
                     ["", "Full log: results/overnight_status.log"])


GBSR_CITES = """## 7. Official GBSR code: exact locations (github.com/yimutianyang/KDD24-GBSR, commit 881d712)

(a) Mask weights detached (masker receives no gradient), PyTorch version:
    `torch_version/GBSR.py`, line 68: `weights = weights.detach()` (in `graph_learner`, lines 51-72).
    TensorFlow version: `models/GBSR.py`, line 74: masked values written with
    `tf.scatter_nd_update(tf.Variable(self.adj_values, trainable=False), ...)` (in `graph_reconstruction`);
    TF1 treats ScatterNdUpdate as non-differentiable (not verified by running TF).
(b) Test set used as validation:
    `rec_dataset.py`, line 71: `self.valdata = np.load(self.data_path + 'testdata.npy', ...)` (line 72 loads the same file as testdata);
    per-epoch model selection evaluates on the test data: `torch_version/run_GBSR.py`, lines 134-136
    (`num_faiss_evaluate(rec_data.testdata, ...)`), best checkpoint chosen by that NDCG@20 (lines 137-153)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-25T00:00")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "results/results.csv"))
    df["hp"] = df["hparams"].map(json.loads)
    df["stage"] = df["hp"].map(lambda h: h.get("stage", "final"))
    since_ts = pd.Timestamp(a.since).timestamp()
    parts = ["# Overnight summary", "", section_douban_warm(df), "", section_prep(), "", section_runs(df, a.since), "",
             section_alpha(since_ts), "", section_lastfm_diag(df), "", section_failures(), "", GBSR_CITES]
    out = "\n".join(parts)
    with open(os.path.join(ROOT, "results/overnight_summary.md"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
