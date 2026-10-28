"""Download LastFM, preprocess, print/save statistics, and write splits for seeds 0,1,2.

Usage: python scripts/prepare_data.py [--seeds 0 1 2]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_douban_book, load_gbsr_dataset, load_lastfm, save_stats  # noqa: E402
from src.splits import make_split, save_split  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dataset", default="lastfm", choices=["lastfm", "douban_book", "epinions", "yelp"])
    args = ap.parse_args()

    loader = {"lastfm": load_lastfm, "douban_book": load_douban_book}.get(args.dataset)
    ds = loader(os.path.join(ROOT, "data/raw")) if loader else load_gbsr_dataset(os.path.join(ROOT, "data/raw"), args.dataset)
    out = os.path.join(ROOT, f"data/processed/{args.dataset}")
    stats = save_stats(ds, os.path.join(out, "stats.json"))
    print(json.dumps(stats, indent=2))
    for s in args.seeds:
        sp = make_split(ds, s)
        save_split(sp, os.path.join(out, f"seed_{s}"), s)
        print(f"seed {s}: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)} "
              f"social_train={len(sp.social_train)} social_val={len(sp.social_val)}")
