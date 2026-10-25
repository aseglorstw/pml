"""Does injecting 40% random edges lower warm LightGCN-S test NDCG@20 on LastFM by more than 0.002?

Compares lightgcn_s_warm_inject40 with lightgcn_s_warm (LastFM, final rows) on the seeds present in both.
Exit 0 = yes (drop > threshold), 4 = no, 5 = missing results. Writes results/diagnostics/corruption_check.json.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THR = 0.002
df = pd.read_csv(os.path.join(ROOT, "results/results.csv"))
df = df[(df["dataset"] == "lastfm") & df["hparams"].map(lambda h: json.loads(h).get("stage") == "final")]
last = df.groupby(["method", "seed"]).tail(1).set_index(["method", "seed"])
out = {"threshold": THR, "per_seed": {}}
try:
    seeds = sorted(set(last.loc["lightgcn_s_warm_inject40"].index) & set(last.loc["lightgcn_s_warm"].index))
except KeyError:
    seeds = []
if not seeds:
    print("missing results")
    sys.exit(5)
for s in seeds:
    c, i = last.loc[("lightgcn_s_warm", s)], last.loc[("lightgcn_s_warm_inject40", s)]
    out["per_seed"][int(s)] = {"clean_test_ndcg20": c["test_ndcg@20"], "inject_test_ndcg20": i["test_ndcg@20"],
                               "clean_val_ndcg20": c["val_ndcg@20"], "inject_val_ndcg20": i["val_ndcg@20"]}
ps = out["per_seed"].values()
out["mean_drop_test"] = sum(p["clean_test_ndcg20"] - p["inject_test_ndcg20"] for p in ps) / len(seeds)
out["mean_drop_val"] = sum(p["clean_val_ndcg20"] - p["inject_val_ndcg20"] for p in ps) / len(seeds)
out["seeds"] = [int(s) for s in seeds]
out["drop_exceeds_threshold"] = bool(out["mean_drop_test"] > THR)
os.makedirs(os.path.join(ROOT, "results/diagnostics"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/diagnostics/corruption_check.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
sys.exit(0 if out["drop_exceeds_threshold"] else 4)
