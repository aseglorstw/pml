"""Clean timing (efficiency section): one run at a time on an otherwise idle GPU.

For each dataset (seed 0): time of the mixture prior + MAP graph (scripts/mixture_prior.py) and of the SHaRe
rewiring (scripts/real_study.py share_graphs), then GBSR on A / on the MAP graph / on the SHaRe graph for a fixed
number of epochs without evaluation during training (training seconds per epoch and peak GPU memory).
Needs the graph files from scripts/real_study.py prep. Writes results/timing.csv and results/timing_report.md.

Usage (cluster, nothing else running on the GPU):  python scripts/timing_bench.py [--datasets douban_book epinions]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import real_study  # noqa: E402

CSV = "results/timing.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["douban_book", "epinions"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    prep = {}
    for ds in a.datasets:
        d = f"data/processed/{ds}/seed_0/pugib"
        t0 = time.perf_counter()
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts/mixture_prior.py"), "--dataset", ds, "--seeds", "0"],
                       check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
        t1 = time.perf_counter()
        real_study.share_graphs(ds, 0)
        t2 = time.perf_counter()
        prep[ds] = {"mixture_prior_s": t1 - t0, "share_rewiring_s": t2 - t1}
        lam, sig = real_study.hsic_of(ds)
        graphs = {"A": [], "map": [f"social_edges_file={d}/map_prior.txt"],
                  "sh": [f"social_edges_file={d}/share_edges.txt", f"social_weights_file={d}/share_w.txt"]}
        for v, g in graphs.items():
            cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts/run_experiment.py"), "configs/paper/gbsr.yaml",
                   "--no-tune", "--seeds", "0", "--device", a.device, "--quiet", "--set", f"dataset={ds}",
                   f"data_dir=data/processed/{ds}", f"name=timing_{ds}_{v}", "social_omega=1", *g,
                   f"hsic_lambda={lam}", f"hsic_sigma={sig}", f"max_epochs={a.epochs}", "eval_every=100000",
                   f"results_csv={CSV}"]
            print(f"{ds} {v}: {a.epochs} epochs ...", flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
    report(a.datasets, prep)


def report(datasets, prep):
    rows = {r["method"]: r for r in csv.DictReader(open(os.path.join(ROOT, CSV)))}
    lines = ["# Timing (one run at a time, seed 0; training seconds per epoch, peak GPU memory during training)", "",
             "| dataset | graph | social ties | s/epoch | peak GPU MB | preprocessing |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        d = os.path.join(ROOT, f"data/processed/{ds}/seed_0")
        files = {"A": os.path.join(d, "social_train.txt"), "map": os.path.join(d, "pugib/map_prior.txt"),
                 "sh": os.path.join(d, "pugib/share_edges.txt")}
        for v, lab in (("A", "GBSR on A"), ("map", "GBSR on MAP graph (ours)"), ("sh", "GBSR on SHaRe graph")):
            r = rows.get(f"timing_{ds}_{v}")
            if r is None:
                continue
            h = json.loads(r["hparams"])
            ties = len(np.loadtxt(files[v], dtype=np.int64, ndmin=2))
            pre = {"A": "–", "map": f"{prep.get(ds, {}).get('mixture_prior_s', float('nan')):.1f} s mixture prior",
                   "sh": f"{prep.get(ds, {}).get('share_rewiring_s', float('nan')):.1f} s rewiring"}[v]
            lines.append(f"| {ds} | {lab} | {ties} | {h.get('sec_per_epoch', float('nan')):.2f} | "
                         f"{h.get('max_mem_mb', float('nan'))} | {pre} |")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(ROOT, "results/timing_report.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
