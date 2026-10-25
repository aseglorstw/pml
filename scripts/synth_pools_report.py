"""Global vs two-hop candidate pools on synthetic seeds 0-2 -> results/synth_pools_report.md.
Paired differences of test NDCG@20 vs LightGCN-S on A with the same omega (run om_lgcnsA_w15)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from synth_diag import runs  # noqa: E402

R = runs()
SEEDS = (0, 1, 2)
st = {s: json.load(open(os.path.join(ROOT, f"data/processed/synth/seed_{s}/ceiling/pool_stats.json"))) for s in SEEDS}
ms = lambda x: f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"
lines = ["# Candidates beyond two hops (synthetic seeds 0-2, social_form gbsr, ω = 1.5)", "",
         "## Pool statistics (mean ± std over seeds)", "",
         "| pool | K | size | precision (Z) | outside Ω | recall of Z \\ A_train |", "|---|---|---|---|---|---|"]
for k in (5, 10, 20):
    for kind, lab in (("hop", "two-hop"), ("glob", "global")):
        v = [st[s][f"{kind}_k{k}"] for s in SEEDS]
        lines.append(f"| {lab} | {k} | {np.mean([x['size'] for x in v]):.0f} | {ms([x['precision'] for x in v])} | "
                     f"{ms([x['frac_outside_omega'] for x in v])} | {ms([x['recall_missing_true'] for x in v])} |")
base = {s: float(R.loc[("om_lgcnsA_w15", s), "test_ndcg@20"]) for s in SEEDS}
lines += ["", f"Baseline LightGCN-S on A (ω = 1.5): test NDCG@20 per seed {[round(base[s], 4) for s in SEEDS]}", "",
          "## Paired Δ test NDCG@20 vs LightGCN-S on A (ω = 1.5)", "",
          "| graph | pool | K | Δ seed 0 | Δ seed 1 | Δ seed 2 | mean ± std |", "|---|---|---|---|---|---|---|"]
for typ, tl in (("ceil", "ceiling: A + true pool ties"), ("hard", "hard addition: A + whole pool")):
    for k in (5, 10, 20):
        for kind, lab in (("hop", "two-hop"), ("glob", "global")):
            n = f"gp_{kind}_{typ}_k{k}"
            d = [float(R.loc[(n, s), "test_ndcg@20"]) - base[s] if (n, s) in R.index else np.nan for s in SEEDS]
            lines.append(f"| {tl} | {lab} | {k} | " + " | ".join(f"{x:+.4f}" for x in d) +
                         f" | {np.nanmean(d):+.4f} ± {np.nanstd(d, ddof=1):.4f} |")
text = "\n".join(lines)
open(os.path.join(ROOT, "results/synth_pools_report.md"), "w").write(text + "\n")
print(text)
