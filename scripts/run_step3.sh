#!/bin/bash
# Step 3: PU-GIB variants on LastFM. Run detached:
#   nohup bash scripts/run_step3.sh > results/logs/step3_main.log 2>&1 &
# Phase 1 tunes the main variant (s pool, anchor prior) for both normalizations in parallel
# and runs the warm-started controls. Phase 2 fixes the winning normalization for everything else.
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
RUN="$PY scripts/run_experiment.py"
L=results/logs
mkdir -p $L

echo "[step3] phase 1 start $(date)"
$RUN configs/pugib/pugib_s_anchor.yaml --tune-only --quiet --set norm=static --device mps > $L/step3_main_static.log 2>&1 &
$RUN configs/pugib/pugib_s_anchor.yaml --tune-only --quiet --set norm=prior  --device cpu > $L/step3_main_prior.log 2>&1 &
( for c in lightgcn_s_warm lightgcn_warm lightgcn_pu_s lightgcn_pu_ra; do
    $RUN configs/pugib/$c.yaml --quiet --device cpu
  done ) > $L/step3_controls.log 2>&1 &
wait

BEST=$($PY scripts/pick_best.py pugib_s_anchor beta kappa norm)
NORM=$($PY scripts/pick_best.py pugib_s_anchor norm | cut -d= -f2)
echo "[step3] main best: $BEST  -> norm=$NORM  $(date)"

( $RUN configs/pugib/pugib_s_anchor.yaml --no-tune --quiet --set $BEST --device mps
  $RUN configs/pugib/pugib_s_grid.yaml --quiet --set norm=$NORM --device mps ) > $L/step3_w1.log 2>&1 &
( for c in pugib_s_en pugib_s_unif; do
    $RUN configs/pugib/$c.yaml --quiet --set norm=$NORM --device cpu
  done ) > $L/step3_w2.log 2>&1 &
( for c in pugib_ra_anchor pugib_ra_en pugib_ra_unif; do
    $RUN configs/pugib/$c.yaml --quiet --set norm=$NORM --device cpu
  done ) > $L/step3_w3.log 2>&1 &
wait
echo "[step3] done $(date)"
