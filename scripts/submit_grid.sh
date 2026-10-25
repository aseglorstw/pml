#!/bin/bash
# Submit one PBS job per line of a job list.
#   bash scripts/submit_grid.sh configs/jobs/example.txt [--dry-run]
# Job list format (one job per line, '#' = comment):
#   <walltime> <config.yaml> [run_experiment.py arguments]
#   04:00:00 configs/pugib/pugib_s_anchor.yaml --no-tune --seeds 1 --set kappa=0.05 beta=0.01
# Queue/resources are in scripts/job.pbs (CHECK them against the MetaCentrum documentation).
set -euo pipefail
cd "$(dirname "$0")/.."
LIST=${1:?job list file}
DRY=${2:-}
mkdir -p results/jobs/specs results/pbs
n=0
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [ -z "${line// }" ] && continue
  read -r WALL CFG REST <<< "$line"
  n=$((n + 1))
  NAME="$(basename "$CFG" .yaml)_$(date +%m%d%H%M%S)_$n"
  SPEC="results/jobs/specs/${NAME}.args"
  echo "$CFG $REST" > "$SPEC"
  CMD=(qsub -N "pg_${n}_$(basename "$CFG" .yaml | cut -c1-8)" -l "walltime=$WALL" -o "results/pbs/${NAME}.out"
       -v "SPEC=$SPEC,PROJECT=$(pwd)" scripts/job.pbs)
  if [ "$DRY" = "--dry-run" ]; then echo "${CMD[*]}   # $(cat "$SPEC")"; else echo "$("${CMD[@]}")  $NAME"; fi
done < "$LIST"
echo "$n job(s) $( [ "$DRY" = "--dry-run" ] && echo prepared || echo submitted )"
