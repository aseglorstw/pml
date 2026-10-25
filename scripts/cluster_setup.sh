#!/bin/bash
# One-time setup on MetaCentrum: venv in the project folder, CUDA torch, requirements, tests on the GPU.
# Run it inside an interactive GPU job (see CLUSTER.md), so the GPU test actually runs:
#   qsub -I -q gpu -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=1:00:00     # CHECK syntax in MetaCentrum docs
#   cd <project> && bash scripts/cluster_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# --- CHECK against MetaCentrum documentation -------------------------------------------------
# Python 3.11 may need a module, e.g. `module avail python` then `module add python/<3.11 version>`.
PYTHON=${PYTHON:-python3.11}
# Torch build: pick a CUDA version supported by the node driver (`nvidia-smi` shows the max CUDA).
TORCH_VERSION=${TORCH_VERSION:-2.14.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
# pip caches/builds in $TMPDIR; on MetaCentrum point it to scratch if home quota is small.
export TMPDIR=${SCRATCHDIR:-${TMPDIR:-/tmp}}
# ----------------------------------------------------------------------------------------------

command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not found (load a Python 3.11 module first)"; exit 1; }
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX" \
  || { echo "CUDA index failed, trying PyPI (Linux wheels on PyPI are CUDA builds)"; python -m pip install "torch==${TORCH_VERSION}"; }
python -m pip install -r requirements-cluster.txt

python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU visible - run this script inside a GPU job to test CUDA")
PY
# Full test suite; tests/test_device.py compares CUDA against CPU when a GPU is visible.
python -m pytest -q -rs
# Timing on the GPU (needs the unpacked data archive)
if [ -d data/processed/lastfm/seed_0 ]; then python scripts/benchmark_device.py --dataset lastfm --epochs 2; fi
echo "setup done"
