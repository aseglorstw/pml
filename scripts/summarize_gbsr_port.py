"""GBSR port check: one table with R@10, N@10, R@20, N@20 and best epoch for every gbsrport_* run, the official
code runs (if their logs exist) and the GBSR paper. Writes results/gbsr_port_report.md."""
import glob
import json
import os
import re

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER = ["gbsrport_official", "gbsrport_beta0", "gbsrport_a_mask_grad", "gbsrport_b_eval_mean",
         "gbsrport_c_gate_fixed1", "gbsrport_d_hsic_users", "gbsrport_e_carved_val", "gbsrport_f_warm",
         "gbsrport_lightgcn"]
LABEL = {"gbsrport_official": "port, official settings (β=40)", "gbsrport_beta0": "port, β=0",
         "gbsrport_a_mask_grad": "(a) mask with gradients", "gbsrport_b_eval_mean": "(b) deterministic gate at eval",
         "gbsrport_c_gate_fixed1": "(c) gate fixed to 1, HSIC kept", "gbsrport_d_hsic_users": "(d) HSIC on users only",
         "gbsrport_e_carved_val": "(e) selection on carved-out validation",
         "gbsrport_f_warm": "(f) warm start, lr_emb 1e-4", "gbsrport_lightgcn": "LightGCN (for f), selection on test"}
fs = glob.glob(os.path.join(ROOT, "results/jobs/*.csv")) + [os.path.join(ROOT, "results/results.csv")]
df = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)
df = df[df["method"].isin(ORDER)].sort_values("timestamp").groupby("method").tail(1).set_index("method")
rows = ["| run | R@10 | N@10 | R@20 | N@20 | best epoch | epochs run | s/epoch |", "|---|---|---|---|---|---|---|---|"]
for m in ORDER:
    if m in df.index:
        r = df.loc[m]
        h = json.loads(r["hparams"])
        rows.append(f"| {LABEL[m]} | {r['test_recall@10']:.4f} | {r['test_ndcg@10']:.4f} | {r['test_recall@20']:.4f} | "
                    f"{r['test_ndcg@20']:.4f} | {int(r['best_epoch'])} | {h.get('epochs_run')} | {h.get('sec_per_epoch')} |")
    else:
        rows.append(f"| {LABEL[m]} | not run yet | | | | | | |")
for log in sorted(glob.glob(os.path.join(ROOT, "external/KDD24-GBSR/saved/douban_book/GBSR/*/training.txt"))):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log).read())
    m = {int(k): (float(r), float(n)) for k, r, n in re.findall(r"Topk:\s*(\d+), HR:[\d.]+, Recall:([\d.]+), NDCG:([\d.]+)", txt)}
    best = re.findall(r"Best Evaluation: Epoch:(\d+)", txt)
    if 10 in m and 20 in m:
        rows.append(f"| official code `{os.path.basename(os.path.dirname(log))}` | {m[10][0]:.4f} | {m[10][1]:.4f} | "
                    f"{m[20][0]:.4f} | {m[20][1]:.4f} | {best[-1] if best else '?'} (0-based) | | |")
rows += ["| paper GBSR (Table 2) | 0.1189 | 0.1451 | 0.1694 | 0.1523 | | | |",
         "| paper LightGCN-S (Table 2) | 0.1021 | 0.1187 | 0.1506 | 0.1281 | | | |"]
text = "\n".join(["# GBSR port check (authors' Douban-Book split, 1 seed; selection on their test file except (e))", "",
                  "Target for the port: N@20 ≈ 0.153 (β=40) and ≈ 0.125 (β=0); stop and diff if β=40 misses by > 0.003.", ""] + rows)
open(os.path.join(ROOT, "results/gbsr_port_report.md"), "w").write(text + "\n")
print(text)
