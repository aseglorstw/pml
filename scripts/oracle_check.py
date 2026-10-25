"""Frozen-embedding check of the social pathway (no training, seconds per row).

Warm LightGCN embeddings are propagated with: no social graph, the real training graph, and the
oracle graphs (scripts/make_oracle_social.py, deliberate test leak), for both social forms.
Usage: python scripts/oracle_check.py --dataset douban_book --seed 0 [--device auto]
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.splits import load_split  # noqa: E402
from src.train import fit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="douban_book")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--omega", type=float, default=0.5)
a = ap.parse_args()
dd = f"data/processed/{a.dataset}"
split = load_split(os.path.join(ROOT, dd, f"seed_{a.seed}"))
base = yaml.safe_load(open(os.path.join(ROOT, "configs/pugib/lightgcn_s_warm.yaml")))
base.update({"dataset": a.dataset, "data_dir": dd, "device": a.device, "freeze_backbone": True,
             "omega_s": a.omega, "max_epochs": 0})
graphs = [("none", None), ("real (train)", None)] + [
    (f"oracle {k}", f"{dd}/seed_{a.seed}/oracle/{k}_k10.txt") for k in ("test_test", "test_train")]
lines = [f"{a.dataset} seed {a.seed}, frozen warm LightGCN embeddings (no training), omega_S={a.omega} for 'ours'",
         f"{'graph':22s} {'form':5s} {'val N@20':>9s} {'test R@20':>9s} {'test N@20':>9s}"]
print("\n".join(lines), flush=True)
for gname, path in graphs:
    for form in (["-"] if gname == "none" else ["ours", "gbsr"]):
        cfg = {**base, "method": "lightgcn" if gname == "none" else "lightgcn_s", "social_form": "ours" if form == "-" else form,
               "init": "e_tilde"}
        if path:
            cfg["social_edges_file"] = path
        r = fit(split, cfg, a.seed, ROOT, log=lambda *_: None)
        lines.append(f"{gname:22s} {form:5s} {r['val_ndcg@20']:9.4f} {r['test']['recall@20']:9.4f} {r['test']['ndcg@20']:9.4f}")
        print(lines[-1], flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
open(os.path.join(ROOT, f"results/oracle_check_{a.dataset}.txt"), "w").write("\n".join(lines) + "\n")
