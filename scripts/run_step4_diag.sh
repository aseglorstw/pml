#!/bin/bash
# Step-4 LastFM diagnostics on CPU, niced and limited to 2 threads (Douban jobs use MPS).
#   nohup bash scripts/run_step4_diag.sh > results/logs/step4_diag.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
PY="nice -n 10 .venv/bin/python -u"
$PY scripts/diag_gradients.py --beta 0 --epochs 50 --out results/diagnostics/grad_beta0.json > results/logs/step4_grad_beta0.log 2>&1
$PY scripts/diag_gradients.py --beta 0.01 --kappa 0.1 --norm prior --epochs 50 --out results/diagnostics/grad_best.json > results/logs/step4_grad_best.log 2>&1
C=configs/pugib/lightgcn_s_warm.yaml
for s in 0 1 2; do
  $PY scripts/run_experiment.py $C --quiet --device cpu --seeds $s \
    --set name=lightgcn_s_warm_inject40 social_edges_file=data/processed/lastfm/seed_$s/corrupt/inject40.txt
done
$PY scripts/run_experiment.py $C --quiet --device cpu --seeds 0 \
  --set name=lightgcn_s_warm_replace100 social_edges_file=data/processed/lastfm/seed_0/corrupt/replace100.txt
echo "[step4] done $(date)"
