#!/bin/bash
# Douban-Book baselines + PU-GIB preprocessing, queued behind the LastFM step-3 jobs.
#   nohup bash scripts/run_douban_step2.sh [PID_TO_WAIT_FOR] > results/logs/douban_step2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
L=results/logs
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "[douban] waiting for pid $WAIT_PID (LastFM step 3) $(date)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo "[douban] start $(date)"
$PY scripts/benchmark_device.py --dataset douban_book --epochs 1 > results/device_benchmark_douban_book.txt 2>&1
cat results/device_benchmark_douban_book.txt
$PY scripts/run_experiment.py configs/douban_book_lightgcn.yaml   --quiet --device mps > $L/douban_lightgcn.log 2>&1 &
P1=$!
$PY scripts/run_experiment.py configs/douban_book_lightgcn_s.yaml --quiet --device mps > $L/douban_lightgcn_s.log 2>&1 &
P2=$!
wait $P1
echo "[douban] LightGCN done $(date); running PU-GIB preprocessing"
$PY scripts/prepare_pugib.py --dataset douban_book --variants best > $L/douban_prepare_pugib.log 2>&1
echo "[douban] preprocessing exit $? $(date)"
wait $P2
echo "[douban] all done $(date)"
