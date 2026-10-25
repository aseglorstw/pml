#!/bin/bash
# Run the OFFICIAL GBSR code (PyTorch version), unmodified, on its own Douban-Book data and split.
#   bash scripts/run_official_gbsr.sh [beta sigma runid] ...   (default: "40 2.5 paper" then "0 2.5 nohsic")
# Clones github.com/yimutianyang/KDD24-GBSR at commit 881d712 into external/ (git-ignored), creates its own
# venv (.venv_gbsr: same torch build + numba, faiss-cpu, tqdm), runs from torch_version/ as in its README.
# Their code: 1000 max epochs, evaluation on the TEST file every epoch, early stop after 10 non-improving
# epochs (after epoch 50), test metrics of that best epoch. Logs: external/KDD24-GBSR/saved/douban_book/GBSR/<runid>/training.txt
set -uo pipefail
cd "$(dirname "$0")/.."
ROOTDIR=$(pwd)
if command -v module >/dev/null 2>&1; then module add "${PY_MODULE:-python/3.11.11-gcc-10.2.1-555dlyc}" >/dev/null 2>&1; fi
if [ -n "${SCRATCHDIR:-}" ]; then export TMPDIR=$SCRATCHDIR; fi
mkdir -p external results/logs
if [ ! -d external/KDD24-GBSR ]; then
  git clone -q https://github.com/yimutianyang/KDD24-GBSR.git external/KDD24-GBSR
  git -C external/KDD24-GBSR checkout -q 881d712313f6b5536dfdb67554675b060f68a618
fi
if [ ! -x .venv_gbsr/bin/python ]; then
  python3 -m venv .venv_gbsr
  .venv_gbsr/bin/pip install -q --no-cache-dir --upgrade pip
  .venv_gbsr/bin/pip install -q --no-cache-dir "torch==${TORCH_VERSION:-2.14.1}" \
      --extra-index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}" \
    || .venv_gbsr/bin/pip install -q --no-cache-dir "torch==${TORCH_VERSION:-2.14.1}"
  .venv_gbsr/bin/pip install -q --no-cache-dir numpy scipy numba faiss-cpu tqdm scikit-learn
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} KMP_DUPLICATE_LIB_OK=TRUE   # faiss + torch OpenMP; no code change
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=("40 2.5 paper" "0 2.5 nohsic")
cd external/KDD24-GBSR/torch_version
for r in "${RUNS[@]}"; do
  read -r BETA SIGMA RID <<< "$r"
  echo "$(date +%FT%T) official GBSR start beta=$BETA sigma=$SIGMA runid=$RID" | tee -a "$ROOTDIR/results/logs/official_gbsr_status.log"
  "$ROOTDIR/.venv_gbsr/bin/python" -u run_GBSR.py --dataset douban_book --runid "$RID" --beta "$BETA" --sigma "$SIGMA" \
     > "$ROOTDIR/results/logs/official_gbsr_${RID}.log" 2>&1
  echo "$(date +%FT%T) official GBSR end runid=$RID exit $?" | tee -a "$ROOTDIR/results/logs/official_gbsr_status.log"
done
