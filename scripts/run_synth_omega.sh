#!/bin/bash
# omega selection on synthetic seed 0 -> seeds 1,2 for selected omega -> report (after synth_diag4 is done)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export DEVICE=${DEVICE:-auto}
python scripts/synth_diag.py omega_jobs > /dev/null
bash scripts/run_jobs_local.sh configs/jobs/synth_omega1.txt 3
python scripts/synth_diag.py omega_select
bash scripts/run_jobs_local.sh configs/jobs/synth_omega2.txt 4
while ! grep -q "ALL DONE (18 jobs from configs/jobs/synth_diag4.txt)" results/logs/local_runner_status.log; do sleep 30; done
python scripts/synth_diag.py report > /dev/null
echo "OMEGA CHAIN DONE"
