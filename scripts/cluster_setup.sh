#!/bin/bash
# One-time install on MetaCentrum (follows the MetaCentrum/course recipe):
# run inside an interactive CPU job with scratch space, from the project folder:
#   qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I
#   cd <project> && bash scripts/cluster_setup.sh
# Then test the GPU once:  bash scripts/cluster_test_gpu.sh  (inside a GPU job, see CLUSTER.md)
set -euo pipefail
cd "$(dirname "$0")/.."

PY_MODULE=${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}
TORCH_VERSION=${TORCH_VERSION:-2.14.1}
# CUDA builds tried in order; plain PyPI Linux wheels are CUDA builds too (last fallback).
TORCH_INDEXES=${TORCH_INDEXES:-"https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu118"}

if command -v module >/dev/null 2>&1; then module add "$PY_MODULE"; fi
if [ -n "${SCRATCHDIR:-}" ]; then
  export TMPDIR=$SCRATCHDIR                      # pip builds/unpacks in scratch, not in the home quota
  trap 'clean_scratch' TERM EXIT
fi

python3 --version
python3 -m venv .venv
PIP=".venv/bin/pip --no-cache-dir"
$PIP install --upgrade pip
ok=0
for idx in $TORCH_INDEXES; do
  if $PIP install "torch==${TORCH_VERSION}" --extra-index-url "$idx"; then ok=1; break; fi
  echo "torch ${TORCH_VERSION} not available from $idx, trying next"
done
[ $ok = 1 ] || $PIP install "torch==${TORCH_VERSION}"
$PIP install -r requirements-cluster.txt

.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
# CPU part of the test suite (GPU comparison is skipped here; run cluster_test_gpu.sh in a GPU job)
.venv/bin/python -m pytest -q -rs
echo "install done; next: GPU test (see CLUSTER.md)"
