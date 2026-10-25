#!/bin/bash
# Run a job list on THIS machine (e.g. an interactive MetaCentrum GPU session), N jobs in parallel
# on the same GPU. Same list format as submit_grid.sh (the walltime column is ignored here):
#   <walltime> <config.yaml> [run_experiment.py arguments]
# Usage:
#   nohup bash scripts/run_jobs_local.sh configs/jobs/example.txt 3 > results/logs/local_runner.log 2>&1 &
# Each job writes results/jobs/local_<listname>_<n>.csv (merge with scripts/merge_results.py) and
# results/logs/local_<listname>_<n>.log. Status lines: results/logs/local_runner_status.log
# Note: everything stops when the interactive session ends; use submit_grid.sh for long runs.
set -uo pipefail
cd "$(dirname "$0")/.."
LIST=${1:?job list file}
NPAR=${2:-3}
DEVICE=${DEVICE:-cuda}
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1; fi
source .venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p results/jobs results/logs
TAG=$(basename "$LIST" .txt)
STATUS=results/logs/local_runner_status.log

run_one() {  # $1 = job number, $2... = config and arguments
  n=$1; shift
  log="results/logs/local_${TAG}_${n}.log"
  echo "$(date +%FT%T) STARTED  #$n $*" >> "$STATUS"
  if python -u scripts/run_experiment.py "$@" --device "$DEVICE" \
       --set results_csv=results/jobs/local_${TAG}_${n}.csv > "$log" 2>&1; then
    echo "$(date +%FT%T) FINISHED #$n $(grep -h '\[run\].*best_epoch' "$log" | tail -1 | cut -c1-160)" >> "$STATUS"
  else
    echo "$(date +%FT%T) FAILED   #$n (see $log): $(tail -1 "$log" | cut -c1-200)" >> "$STATUS"
  fi
}

n=0
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [ -z "${line// }" ] && continue
  read -r _WALL REST <<< "$line"
  n=$((n + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 5; done
  eval "set -- $REST"
  run_one "$n" "$@" &
done < "$LIST"
wait
echo "$(date +%FT%T) ALL DONE ($n jobs from $LIST)" >> "$STATUS"
