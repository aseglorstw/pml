"""Merge per-job CSVs (results/jobs/*.csv) into results/results.csv.

Duplicate rows are dropped, rows are sorted by timestamp, and the previous results.csv is backed up.
Usage: python scripts/merge_results.py [--jobs results/jobs] [--out results/results.csv] [--dry-run]
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.train import RESULT_COLUMNS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", default=os.path.join(ROOT, "results/jobs"))
ap.add_argument("--out", default=os.path.join(ROOT, "results/results.csv"))
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.jobs, "*.csv")))
parts = [pd.read_csv(f, dtype=str) for f in files]
new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RESULT_COLUMNS)
old = pd.read_csv(a.out, dtype=str) if os.path.exists(a.out) else pd.DataFrame(columns=RESULT_COLUMNS)
merged = pd.concat([old, new], ignore_index=True)[RESULT_COLUMNS].drop_duplicates().sort_values("timestamp", kind="stable")
added = len(merged) - len(old.drop_duplicates())
print(f"{len(files)} job file(s), {len(new)} job row(s); results.csv {len(old)} -> {len(merged)} rows (+{added})")
if not a.dry_run:
    if os.path.exists(a.out):
        shutil.copy(a.out, a.out + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    merged.to_csv(a.out, index=False)
