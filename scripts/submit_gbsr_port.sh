#!/bin/bash
# GBSR port check on the authors' Douban-Book split as PBS jobs (one GPU each).
#   bash scripts/submit_gbsr_port.sh 1 [--dry-run]   # run 1 (beta 40 and beta 0) + LightGCN for run (f)
#   bash scripts/submit_gbsr_port.sh 2 [--dry-run]   # ablations (a)-(f); (f) waits for the LightGCN job
# Report: python scripts/summarize_gbsr_port.py   (reads results/jobs/*.csv and the official logs)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage 1 or 2}; DRY=${2:-}
source .venv/bin/activate
[ -f data/processed/douban_gbsr/seed_0/meta.json ] || python scripts/make_gbsr_split.py
mkdir -p results/jobs/specs results/pbs
LIST=configs/jobs/gbsr_port_stage$STAGE.txt
LGJOB=results/jobs/gbsr_port_lightgcn.jobid
n=0
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [ -z "${line// }" ] && continue
  read -r WALL CFG REST <<< "$line"
  n=$((n + 1))
  NAME=$(echo "$REST" | grep -o 'name=[a-z0-9_]*' | head -1 | cut -d= -f2 || true); NAME=${NAME:-gbsrport_official}
  SPEC="results/jobs/specs/${NAME}.args"; echo "$CFG $REST" > "$SPEC"
  ARGS=(-N "${NAME:0:15}" -l "walltime=$WALL" -o "results/pbs/${NAME}.out" -v "SPEC=$SPEC,PROJECT=$(pwd)")
  if [ "$NAME" = "gbsrport_f_warm" ] && [ ! -f results/checkpoints/lightgcn_douban_gbsr_seed0.pt ] && [ -f "$LGJOB" ]; then
    ARGS+=(-W "depend=afterok:$(cat $LGJOB)")
  fi
  if [ "$DRY" = "--dry-run" ]; then echo "qsub ${ARGS[*]} scripts/job.pbs   # $CFG $REST"; continue; fi
  JID=$(qsub "${ARGS[@]}" scripts/job.pbs); echo "$JID  $NAME"
  [ "$NAME" = "gbsrport_lightgcn" ] && echo "$JID" > "$LGJOB"
done < "$LIST"
echo "$n job(s) $( [ "$DRY" = "--dry-run" ] && echo prepared || echo submitted ). Monitor: qstat -u \$USER"
