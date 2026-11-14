"""Filter a job list: drop jobs whose run has already finished (results/peruser/<dataset>/<name>/seed<k>.npz exists).

A run writes its per-user file only at the very end, so interrupted runs are kept and start again from scratch.
Usage: python scripts/skip_finished.py <list.txt>   (prints the remaining jobs; comments are dropped)
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def finished(line):
    name = re.search(r"\bname=(\S+)", line)
    ds = re.search(r"\bdataset=(\S+)", line)
    seed = re.search(r"--seeds\s+(\d+)", line)
    if not (name and ds and seed):
        return False
    return os.path.exists(os.path.join(ROOT, "results/peruser", ds.group(1), name.group(1), f"seed{seed.group(1)}.npz"))


if __name__ == "__main__":
    for ln in open(sys.argv[1]):
        ln = ln.rstrip("\n")
        if ln.strip() and not ln.lstrip().startswith("#") and not finished(ln):
            print(ln)
