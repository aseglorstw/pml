#!/bin/bash
# Douban-Book pipeline as ordinary processes on the GPU of THIS machine (e.g. the interactive session).
#   nohup bash scripts/run_douban_local.sh [--stage1-seeds "0 1"] [--from 1|2|3] [--par 5] \
#        > results/logs/douban_local.log 2>&1 &
#   tail -f results/logs/douban_local_status.log          # stage progress
#   tail -f results/logs/local_runner_status.log          # individual runs
# Stage 1: cold LightGCN for the given seeds (checkpoints for warm start), all in parallel
# Stage 2: PU-GIB preprocessing for seeds 0 1 2
# Stage 3: configs/jobs/douban_stage3.txt, --par runs at a time
# Stage 4: merge per-run CSVs + results/douban_stage3_summary.md
# Everything stops if the session ends.
set -uo pipefail
cd "$(dirname "$0")/.."
S1SEEDS="0 1 2"; FROM=1; PAR=5
while [ $# -gt 0 ]; do
  case $1 in --stage1-seeds) S1SEEDS=$2; shift;; --from) FROM=$2; shift;; --par) PAR=$2; shift;;
             *) echo "unknown $1"; exit 1;; esac; shift
done
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1; fi
source .venv/bin/activate
mkdir -p results/logs results/jobs
ST=results/logs/douban_local_status.log
say() { echo "$(date +%FT%T) $*" | tee -a "$ST"; }
DB="--set dataset=douban_book data_dir=data/processed/douban_book reg=0.0001 eval_every=10"

if [ "$FROM" -le 1 ]; then
  L=results/jobs/douban_stage1.txt
  : > "$L"
  for s in $S1SEEDS; do echo "- configs/douban_book_lightgcn.yaml --no-tune --seeds $s $DB" >> "$L"; done
  say "stage 1 start: cold LightGCN seeds $S1SEEDS"
  bash scripts/run_jobs_local.sh "$L" "$(echo $S1SEEDS | wc -w)"
  for s in 0 1 2; do
    [ -f results/checkpoints/lightgcn_douban_book_seed$s.pt ] || { say "stage 1 FAILED: no checkpoint for seed $s"; exit 1; }
  done
  say "stage 1 done"
fi
if [ "$FROM" -le 2 ]; then
  say "stage 2 start: PU-GIB preprocessing seeds 0 1 2"
  if python -u scripts/prepare_pugib.py --dataset douban_book --variants best --seeds 0 1 2 > results/logs/douban_prep.log 2>&1 \
     && python scripts/summarize_prep.py --dataset douban_book >> results/logs/douban_prep.log 2>&1; then
    say "stage 2 done: results/douban_book_prep_summary.md"
  else
    say "stage 2 FAILED (see results/logs/douban_prep.log)"; exit 1
  fi
fi
if [ "$FROM" -le 3 ]; then
  say "stage 3 start: $(grep -c '^-' configs/jobs/douban_stage3.txt) runs, $PAR in parallel"
  bash scripts/run_jobs_local.sh configs/jobs/douban_stage3.txt "$PAR"
  say "stage 3 done ($(grep -c FAILED results/logs/local_runner_status.log) FAILED lines in runner log)"
fi
python scripts/merge_results.py > /dev/null 2>&1
python scripts/summarize_runs.py --prefix db3_ --out results/douban_stage3_summary.md > /dev/null 2>&1
say "ALL DONE: results/douban_book_prep_summary.md, results/douban_stage3_summary.md"
