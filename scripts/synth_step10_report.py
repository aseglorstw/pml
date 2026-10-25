"""Separate candidate channel and targeted addition (synthetic seeds 0-2, ω = 1.5) + subgroup tables.
Paired Δ test NDCG@20 vs LightGCN-S on A (run sg_lgcns_A, same config as om_lgcnsA_w15, with per-user metrics).
Writes results/synth_step10_report.md"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from synth_diag import runs  # noqa: E402

R = runs()
SEEDS = (0, 1, 2)
REF = "sg_lgcns_A"
base = {s: float(R.loc[(REF, s), "test_ndcg@20"]) for s in SEEDS if (REF, s) in R.index}


def row(label, name):
    d = [float(R.loc[(name, s), "test_ndcg@20"]) - base[s] if (name, s) in R.index and s in base else np.nan for s in SEEDS]
    v = np.array(d)
    m = f"{np.nanmean(v):+.4f} ± {np.nanstd(v, ddof=1):.4f}" if np.isfinite(v).sum() > 1 else "–"
    return f"| {label} | " + " | ".join("–" if np.isnan(x) else f"{x:+.4f}" for x in d) + f" | {m} |"


L = ["# Separate candidate channel and targeted addition (synthetic, ω = 1.5, gbsr form)", "",
     f"Reference LightGCN-S on A ({REF}) test NDCG@20 per seed: {[round(base.get(s, float('nan')), 4) for s in SEEDS]}", "",
     "| variant | Δ seed 0 | Δ seed 1 | Δ seed 2 | mean ± std |", "|---|---|---|---|---|",
     row("joint form, ceiling: A + true global K20 ties (earlier run)", "gp_glob_ceil_k20"),
     row("joint form, hard: A + whole global K20 pool (earlier run)", "gp_glob_hard_k20")]
for w in ("0.25", "0.5", "1"):
    L.append(row(f"separate channel ω_c={w}, ceiling (true global K20 ties)", f"sep_ceil_g20_wc{w}"))
for w in ("0.25", "0.5", "1"):
    L.append(row(f"separate channel ω_c={w}, hard (whole global K20 pool)", f"sep_hard_g20_wc{w}"))
L += [row("targeted (deg ≤ 2, global K10), joint, ceiling", "tgt_joint_ceil"),
      row("targeted (deg ≤ 2, global K10), joint, hard", "tgt_joint_hard")]
for w in ("0.25", "0.5", "1"):
    L.append(row(f"targeted, separate ω_c={w}, ceiling", f"tgt_sep_ceil_wc{w}"))
for w in ("0.25", "0.5", "1"):
    L.append(row(f"targeted, separate ω_c={w}, hard", f"tgt_sep_hard_wc{w}"))
L += ["", "Note: earlier gp_* rows were computed against om_lgcnsA_w15, here against sg_lgcns_A (same configuration).", ""]
for args in (["--dataset", "synth", "--ref", REF], ["--dataset", "lastfm", "--ref", "sg_lf_lightgcn_warm"]):
    L.append(subprocess.run([sys.executable, os.path.join(ROOT, "scripts/subgroups.py"), *args],
                            capture_output=True, text=True).stdout)
text = "\n".join(L)
open(os.path.join(ROOT, "results/synth_step10_report.md"), "w").write(text + "\n")
print(text)
