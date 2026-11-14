"""GBSR port check: per dataset (authors' Douban-Book, Epinions, Yelp splits) one table with R@10, N@10, R@20, N@20
and best epoch for every gbsrport_* run, the official code runs (if their logs exist) and the GBSR paper (Table 2,
arXiv 2406.08214v2). Writes results/gbsr_port_report.md."""
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
df = df[df["method"].str.startswith("gbsrport_", na=False)].sort_values("timestamp").groupby("method").tail(1)
df = df.set_index("method")
# GBSR paper, Table 2: R@10, N@10, R@20, N@20
PAPER = {"douban_book": {"GBSR": (0.1189, 0.1451, 0.1694, 0.1523), "LightGCN-S": (0.1021, 0.1187, 0.1506, 0.1281),
                         "LightGCN": (0.1039, 0.1195, 0.1526, 0.1283)},
         "yelp": {"GBSR": (0.0805, 0.0592, 0.1243, 0.0724), "LightGCN-S": (0.0714, 0.0529, 0.1126, 0.0651),
                  "LightGCN": (0.0698, 0.0507, 0.1081, 0.0623)},
         "epinions": {"GBSR": (0.0529, 0.0385, 0.0793, 0.0464), "LightGCN-S": (0.0477, 0.0347, 0.0716, 0.0417),
                      "LightGCN": (0.0432, 0.0314, 0.0675, 0.0385)}}
SECTIONS = {"douban_book": (ORDER, LABEL)}
for ds, (b, s) in {"epinions": (3, 0.25), "yelp": (2, 0.25)}.items():
    SECTIONS[ds] = ([f"gbsrport_{ds}_official", f"gbsrport_{ds}_beta0", f"gbsrport_{ds}_lightgcn"],
                    {f"gbsrport_{ds}_official": f"port, official settings (β={b}, σ={s})",
                     f"gbsrport_{ds}_beta0": "port, β=0 (no HSIC)", f"gbsrport_{ds}_lightgcn": "LightGCN, no social ties"})
text = ["# GBSR port check (authors' splits, 1 seed; selection on their test file except Douban (e))", "",
        "Port = our reimplementation with the official settings. Official code = github.com/yimutianyang/KDD24-GBSR "
        "at 881d712, unmodified (scripts/run_official_gbsr.sh).", ""]
for ds, (order, label) in SECTIONS.items():
    rows = [f"## {ds}", "", "| run | R@10 | N@10 | R@20 | N@20 | best epoch | epochs run | s/epoch |",
            "|---|---|---|---|---|---|---|---|"]
    for m in order:
        if m in df.index:
            r = df.loc[m]
            h = json.loads(r["hparams"])
            rows.append(f"| {label[m]} | {r['test_recall@10']:.4f} | {r['test_ndcg@10']:.4f} | {r['test_recall@20']:.4f} | "
                        f"{r['test_ndcg@20']:.4f} | {int(r['best_epoch'])} | {h.get('epochs_run')} | {h.get('sec_per_epoch')} |")
        else:
            rows.append(f"| {label[m]} | not run yet | | | | | | |")
    for log in sorted(glob.glob(os.path.join(ROOT, f"external/KDD24-GBSR/saved/{ds}/GBSR/*/training.txt"))):
        txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log).read())
        m = {int(k): (float(r), float(n)) for k, r, n in re.findall(r"Topk:\s*(\d+), HR:[\d.]+, Recall:([\d.]+), NDCG:([\d.]+)", txt)}
        best = re.findall(r"Best Evaluation: Epoch:(\d+)", txt)
        if 10 in m and 20 in m:
            rows.append(f"| official code `{os.path.basename(os.path.dirname(log))}` | {m[10][0]:.4f} | {m[10][1]:.4f} | "
                        f"{m[20][0]:.4f} | {m[20][1]:.4f} | {best[-1] if best else '?'} (0-based) | | |")
    for k, v in PAPER[ds].items():
        rows.append(f"| paper {k} (Table 2) | " + " | ".join(f"{x:.4f}" for x in v) + " | | | |")
    text += rows + [""]
text = "\n".join(text)
open(os.path.join(ROOT, "results/gbsr_port_report.md"), "w").write(text + "\n")
print(text)
