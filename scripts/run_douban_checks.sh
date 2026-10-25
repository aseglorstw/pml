#!/bin/bash
# Implementation checks on Douban-Book seed 0 in the interactive GPU session (ordinary processes):
#   nohup bash scripts/run_douban_checks.sh [--par 4] [--skip-official] > results/logs/douban_checks.log 2>&1 &
#   1. oracle graphs + frozen check (minutes) -> results/oracle_check_douban_book.txt
#   2. 7 trained runs (configs/jobs/douban_checks.txt), --par at a time
#   3. official GBSR code (2 runs, sequential; needs ~20 GB RAM each while building its dense adjacency)
#   4. results/douban_checks_summary.md
set -uo pipefail
cd "$(dirname "$0")/.."
PAR=4; OFFICIAL=1
while [ $# -gt 0 ]; do case $1 in --par) PAR=$2; shift;; --skip-official) OFFICIAL=0;; *) exit 1;; esac; shift; done
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1; fi
source .venv/bin/activate
mkdir -p results/logs
ST=results/logs/douban_checks_status.log
say() { echo "$(date +%FT%T) $*" | tee -a "$ST"; }
say "1. oracle graphs + frozen check"
python scripts/make_oracle_social.py --dataset douban_book --seed 0 >> "$ST" 2>&1
python scripts/oracle_check.py --dataset douban_book --seed 0 --device cuda >> "$ST" 2>&1
say "2. trained runs ($PAR in parallel)"
bash scripts/run_jobs_local.sh configs/jobs/douban_checks.txt "$PAR"
say "2. done"
if [ $OFFICIAL = 1 ]; then say "3. official GBSR"; bash scripts/run_official_gbsr.sh; say "3. done"; fi
python scripts/merge_results.py > /dev/null 2>&1
python scripts/summarize_checks.py >> "$ST" 2>&1
say "ALL DONE: results/douban_checks_summary.md"
