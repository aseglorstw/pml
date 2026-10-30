#!/bin/bash
# Whole real-data study on THIS machine (interactive GPU session), one dataset:
#   1. interaction-only LightGCN per seed (skipped if results/checkpoints/lightgcn_<ds>_seed<k>.pt exists)
#   2. mixture prior + MAP graph, SHaRe graphs   3. GBSR on A / MAP / SHaRe cut / SHaRe full (4 in parallel)
#   4. report -> results/real_<ds>_report.md
# Usage (in the project folder):  nohup bash scripts/real_pipeline.sh epinions 0 > results/logs/real_epinions.log 2>&1 &
# NPAR=<n> sets the number of parallel GBSR runs (default 4).
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?dataset}; shift
SEEDS=${*:-0}
DEVICE=${DEVICE:-cuda}
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1 || true; fi
source .venv/bin/activate
mkdir -p results/logs results/jobs
echo "$(date +%FT%T) start $DS seeds $SEEDS"
for s in $SEEDS; do
  if [ ! -f "results/checkpoints/lightgcn_${DS}_seed${s}.pt" ]; then
    echo "$(date +%FT%T) LightGCN seed $s"
    python -u scripts/run_experiment.py "configs/${DS}_lightgcn.yaml" --no-tune --seeds "$s" --device "$DEVICE" --quiet \
      --set name="real_${DS}_lightgcn" results_csv="results/jobs/local_real_${DS}_lightgcn.csv"
  fi
done
echo "$(date +%FT%T) prep"
python scripts/real_study.py prep --dataset "$DS" --seeds $SEEDS
python scripts/real_study.py jobs --dataset "$DS" --seeds $SEEDS
echo "$(date +%FT%T) GBSR runs (status: results/logs/local_runner_status.log)"
DEVICE=$DEVICE bash scripts/run_jobs_local.sh "configs/jobs/real_${DS}_s$(echo $SEEDS | tr -d ' ').txt" "${NPAR:-4}"
python scripts/real_study.py report --dataset "$DS" --seeds $SEEDS
echo "$(date +%FT%T) done"
