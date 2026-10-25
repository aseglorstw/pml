#!/bin/bash
# Synthetic sanity suite + LastFM regression; must pass before expensive experiments.
#   bash scripts/sanity.sh [--par 4] [--device auto]      -> results/sanity_report.md
# Steps: generate synthetic world -> cold LightGCN (warm-start checkpoint) -> PU-GIB preprocessing ->
#        +50% random edges graph -> LastFM oracle graph -> runs (configs/jobs/sanity.txt) -> report.
set -uo pipefail
cd "$(dirname "$0")/.."
PAR=4; export DEVICE=auto
while [ $# -gt 0 ]; do case $1 in --par) PAR=$2; shift;; --device) DEVICE=$2; shift;; *) exit 1;; esac; shift; done
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1; fi
source .venv/bin/activate
mkdir -p results/logs results/jobs
rm -f results/jobs/local_sanity_*.csv; rm -rf results/pugib_runs/san_*
L=results/logs/sanity.log; : > "$L"
step() { echo "$(date +%T) $*" | tee -a "$L"; }
fail() { step "ERROR: $*"; exit 1; }

step "1 synthetic world";        python scripts/make_synthetic.py >> "$L" 2>&1 || fail "generator"
step "2 cold LightGCN (synth)"
python -u scripts/run_experiment.py configs/lastfm_lightgcn.yaml --quiet --no-tune --seeds 0 --device "$DEVICE" \
  --set dataset=synth data_dir=data/processed/synth reg=0.0001 name=san_syn_lightgcn_cold \
        results_csv=results/jobs/local_sanity_cold.csv >> "$L" 2>&1 || fail "cold LightGCN"
step "3 PU-GIB preprocessing (synth)"
python -u scripts/prepare_pugib.py --dataset synth --variants best --seeds 0 >> "$L" 2>&1 || fail "preprocessing"
step "4 noisy graphs (+50% uniform random edges; +50% random Omega edges)"
python scripts/make_corrupted_social.py --dataset synth --seed 0 --frac 0.5 --mode inject --universe all >> "$L" 2>&1 || fail "noise graph"
python scripts/make_corrupted_social.py --dataset synth --seed 0 --frac 0.5 --mode inject >> "$L" 2>&1 || fail "noise graph (omega)"
step "5 LastFM oracle graph"
[ -f data/processed/lastfm/seed_0/oracle/test_train_k10.txt ] || python scripts/make_oracle_social.py --dataset lastfm --seed 0 >> "$L" 2>&1
step "6 runs ($PAR in parallel, device $DEVICE)"
bash scripts/run_jobs_local.sh configs/jobs/sanity.txt "$PAR"
step "7 report"
python scripts/sanity_report.py | tee -a "$L"
