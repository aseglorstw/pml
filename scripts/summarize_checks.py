"""results/douban_checks_summary.md: frozen oracle check, trained checks, official GBSR vs paper."""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = {"LightGCN": (0.1039, 0.1195, 0.1526, 0.1283), "LightGCN-S": (0.1021, 0.1187, 0.1506, 0.1281),
         "GBSR": (0.1189, 0.1451, 0.1694, 0.1523)}  # Douban-Book, Table 2 (R@10, N@10, R@20, N@20)
parts = ["# Douban-Book implementation checks (seed 0)", ""]
f = os.path.join(ROOT, "results/oracle_check_douban_book.txt")
parts += ["## 1. Frozen check (warm LightGCN embeddings, no training)", "", "```", open(f).read().strip() if os.path.exists(f) else "missing", "```", ""]
t = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/summarize_runs.py"), "--prefix", "chk_"],
                   capture_output=True, text=True).stdout
parts += ["## 2. Trained (new schedule; oracle = deliberate test leak)", "", t.strip(), ""]
rows = []
for log in sorted(glob.glob(os.path.join(ROOT, "external/KDD24-GBSR/saved/douban_book/GBSR/*/training.txt"))):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log).read())
    m = {int(k): (float(r), float(n)) for k, r, n in re.findall(r"Topk:\s*(\d+), HR:[\d.]+, Recall:([\d.]+), NDCG:([\d.]+)", txt)}
    best = re.findall(r"Best Evaluation: Epoch:(\d+)", txt)
    beta = re.search(r"beta=([\d.]+)", txt)
    if 10 in m and 20 in m:
        rows.append(f"| official run `{os.path.basename(os.path.dirname(log))}` (beta={beta.group(1) if beta else '?'}) | "
                    f"{m[10][0]:.4f} | {m[10][1]:.4f} | {m[20][0]:.4f} | {m[20][1]:.4f} | {best[-1] if best else '?'} |")
    else:
        rows.append(f"| official run `{os.path.basename(os.path.dirname(log))}` | not finished | | | | {best[-1] if best else '?'} |")
rows += [f"| paper {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | |" for k, v in PAPER.items()]
parts += ["## 3. Official GBSR code on its own Douban-Book split vs paper (Table 2)", "",
          "| run | R@10 | N@10 | R@20 | N@20 | best epoch |", "|---|---|---|---|---|---|"] + rows
parts += ["", "Official code selects the best epoch on its TEST file (rec_dataset.py:71); the paper averages 5 runs."]
out = "\n".join(parts)
print(out)
open(os.path.join(ROOT, "results/douban_checks_summary.md"), "w").write(out + "\n")
