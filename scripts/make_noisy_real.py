"""Inject random false ties into the TRAINING social graph of a real dataset (RQ4), with a known ground truth.

For every seed, data/processed/<ds>_n<pct>/seed_<k>/ gets the same interaction split and held-out social ties as
<ds>, and social_train = original training ties + round(frac * |E_train|) uniformly random user pairs that are not
linked in the full observed graph (train or held-out). The injected ties are listed in truth/injected.txt.
The interaction-only LightGCN does not depend on the social graph, so its checkpoint is reused
(results/checkpoints/lightgcn_<ds>_seed<k>.pt is copied to lightgcn_<ds>_n<pct>_seed<k>.pt).

Usage (where the checkpoints are, i.e. on the cluster):
  python scripts/make_noisy_real.py --dataset epinions --fracs 0.5 1.0 --seeds 0
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pugib_prep import isin_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    for frac in a.fracs:
        name = f"{a.dataset}_n{int(round(frac * 100))}"
        for s in a.seeds:
            src = os.path.join(ROOT, "data/processed", a.dataset, f"seed_{s}")
            dst = os.path.join(ROOT, "data/processed", name, f"seed_{s}")
            os.makedirs(os.path.join(dst, "truth"), exist_ok=True)
            for f in ("train.txt", "val.txt", "test.txt", "social_val.txt", "meta.json"):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            meta = json.load(open(os.path.join(src, "meta.json")))
            n = meta["n_users"]
            tr = np.loadtxt(os.path.join(src, "social_train.txt"), dtype=np.int64)
            va = np.loadtxt(os.path.join(src, "social_val.txt"), dtype=np.int64)
            observed = np.concatenate([tr, va])
            k = int(round(frac * len(tr)))
            rng = np.random.default_rng([s, 13, int(round(frac * 100))])
            inj = np.zeros((0, 2), np.int64)
            while len(inj) < k:
                p = np.sort(rng.integers(0, n, (2 * (k - len(inj)) + 1000, 2)), 1)
                p = p[p[:, 0] != p[:, 1]]
                p = p[~isin_pairs(p, observed, n)]
                inj = np.unique(np.concatenate([inj, p]), axis=0)
            inj = inj[rng.permutation(len(inj))[:k]]
            st = np.concatenate([tr, inj])
            st = st[np.lexsort((st[:, 1], st[:, 0]))]
            np.savetxt(os.path.join(dst, "social_train.txt"), st, fmt="%d")
            np.savetxt(os.path.join(dst, "truth", "injected.txt"), inj, fmt="%d")
            meta.update({"n_social_train": int(len(st)), "injected": k, "injected_frac": frac, "base_dataset": a.dataset})
            json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
            ck = os.path.join(ROOT, "results/checkpoints", f"lightgcn_{a.dataset}_seed{s}.pt")
            if os.path.exists(ck):
                shutil.copy(ck, os.path.join(ROOT, "results/checkpoints", f"lightgcn_{name}_seed{s}.pt"))
            else:
                print(f"warning: {ck} missing (train the interaction-only LightGCN first)")
            print(f"{name} seed {s}: {len(tr)} original + {k} injected = {len(st)} training ties "
                  f"(true noise from injection {k / len(st):.3f})")


if __name__ == "__main__":
    main()
