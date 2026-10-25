"""Compare CPU and MPS for sparse matmul and LightGCN(-S) training epochs on seed 0 of a dataset.

Usage: python scripts/benchmark_device.py [--dataset douban_book] [--epochs 1]
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.lightgcn import LightGCN, norm_bipartite_adj, sparse_tensor  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.train import NegativeSampler  # noqa: E402


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def bench_spmm(sp_, dev, reps=200):
    A = sparse_tensor(norm_bipartite_adj(sp_.train, sp_.n_users, sp_.n_items), dev)
    X = torch.randn(sp_.n_users + sp_.n_items, 64, device=dev, requires_grad=True)
    ref = torch.sparse.mm(A.cpu(), X.detach().cpu())
    out = torch.sparse.mm(A, X)
    err = (out.detach().cpu() - ref).abs().max().item()
    out.sum().backward()  # backward must work too
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = torch.sparse.mm(A, X)
    sync(dev)
    return (time.perf_counter() - t0) / reps * 1e3, err


def bench_epoch(sp_, dev, social, n_epochs=3):
    torch.manual_seed(0)
    m = LightGCN(sp_.n_users, sp_.n_items, sp_.train, social_edges=sp_.social_train if social else None,
                 device=dev).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    smp = NegativeSampler(sp_.train, sp_.n_users, sp_.n_items, rng)
    times = []
    for ep in range(n_epochs + 1):  # first epoch is warm-up
        t0 = time.perf_counter()
        perm = rng.permutation(len(sp_.train))
        for s in range(0, len(perm), 2048):
            b = sp_.train[perm[s:s + 2048]]
            neg = smp.sample(b[:, 0])
            loss, _ = m.bpr_loss(torch.as_tensor(b[:, 0], device=dev), torch.as_tensor(b[:, 1], device=dev),
                                 torch.as_tensor(neg, device=dev), 1e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync(dev)
        if ep:
            times.append(time.perf_counter() - t0)
    return float(np.mean(times))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lastfm")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--devices", nargs="+", default=None, help="subset of cpu/cuda/mps to time (default: all available)")
    args = ap.parse_args()
    sp_ = load_split(os.path.join(ROOT, f"data/processed/{args.dataset}/seed_0"))
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
    if args.devices:
        devs = [d for d in devs if d in args.devices]
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, devices {devs}", flush=True)
    for dev in devs:
        try:
            ms, err = bench_spmm(sp_, dev)
            print(f"{dev}: sparse.mm fwd {ms:.3f} ms (max abs diff vs CPU {err:.2e})", flush=True)
            for social in (False, True):
                t = bench_epoch(sp_, dev, social, n_epochs=args.epochs)
                print(f"{dev}: {'LightGCN-S' if social else 'LightGCN  '} epoch {t:.3f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{dev}: FAILED: {type(e).__name__}: {e}")
