"""Print the best tuning combination of a method as key=value pairs (for --set).

Usage: python scripts/pick_best.py <method name> key [key ...]
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
name, keys = sys.argv[1], sys.argv[2:]
df = pd.read_csv(os.path.join(ROOT, "results/results.csv"))
df = df[df["method"] == name]
hp = df["hparams"].map(json.loads)
df = df[hp.map(lambda h: h.get("stage")) == "tune"]
best = json.loads(df.loc[df["val_ndcg@20"].idxmax(), "hparams"])
print(" ".join(f"{k}={best[k]}" for k in keys))
