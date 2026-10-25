#!/bin/bash
# Build synthetic world <seed> end to end: generator -> cold LightGCN (warm-start checkpoint) ->
# PU-GIB preprocessing -> K-pools.   bash scripts/prep_synth_seed.sh 1
set -euo pipefail
cd "$(dirname "$0")/.."
s=$1; DEV=${DEVICE:-auto}
source .venv/bin/activate
python scripts/make_synthetic.py --seed "$s" > /dev/null
python -u scripts/run_experiment.py configs/lastfm_lightgcn.yaml --quiet --no-tune --seeds "$s" --device "$DEV" \
  --set dataset=synth data_dir=data/processed/synth reg=0.0001 name=syn_lightgcn_cold \
        results_csv=results/jobs/local_synth_cold_s$s.csv | grep "best_epoch"
python -u scripts/prepare_pugib.py --dataset synth --variants best --seeds "$s" | grep -E "g_val|anchor:"
python scripts/make_pools.py --dataset synth --seed "$s" --K 2 5 10 > /dev/null
echo "synth seed $s ready"
