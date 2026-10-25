#!/bin/bash
# GPU check on MetaCentrum, inside a GPU job, from the project folder:
#   qsub -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=8gb -l walltime=0:30:00 -I
#   cd <project> && bash scripts/cluster_test_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY_MODULE=${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}
if command -v module >/dev/null 2>&1; then module add "$PY_MODULE"; fi
if [ -n "${SCRATCHDIR:-}" ]; then export TMPDIR=$SCRATCHDIR; trap 'clean_scratch' TERM EXIT; fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible: run inside a GPU job (ngpus=1)"
print("GPU:", torch.cuda.get_device_name(0))
PY
# tests/test_device.py compares CUDA with CPU results
.venv/bin/python -m pytest -q -rs tests/test_device.py tests/test_lightgcn.py tests/test_pugib_model.py tests/test_gbsr.py
# timing (needs the unpacked data archive)
if [ -d data/processed/lastfm/seed_0 ]; then .venv/bin/python scripts/benchmark_device.py --dataset lastfm --epochs 2; fi
if [ -d data/processed/douban_book/seed_0 ]; then .venv/bin/python scripts/benchmark_device.py --dataset douban_book --epochs 1; fi
echo "GPU test done"
