#!/bin/bash
# Minimal Douban-Book experiment (seed 0, L2 = 1e-4, val every 10 epochs):
#   a. LightGCN (cold) -> checkpoint;  b. LightGCN warm and LightGCN-S warm from that checkpoint.
#   nohup bash scripts/run_douban_min.sh > results/logs/douban_min.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
RUN="$PY scripts/run_experiment.py"
L=results/logs
D="dataset=douban_book data_dir=data/processed/douban_book reg=0.0001 eval_every=10"
echo "[douban-min] a. LightGCN start $(date)"
$RUN configs/douban_book_lightgcn.yaml --no-tune --seeds 0 --device mps --set $D > $L/douban_min_lightgcn.log 2>&1
grep "\[run\].*best_epoch" $L/douban_min_lightgcn.log
echo "[douban-min] b. warm runs start $(date)"
$RUN configs/pugib/lightgcn_warm.yaml   --seeds 0 --device mps --set $D > $L/douban_min_lightgcn_warm.log 2>&1 &
$RUN configs/pugib/lightgcn_s_warm.yaml --seeds 0 --device mps --set $D > $L/douban_min_lightgcn_s_warm.log 2>&1 &
wait
grep -h "\[run\].*best_epoch" $L/douban_min_lightgcn_warm.log $L/douban_min_lightgcn_s_warm.log
echo "[douban-min] done $(date)"
