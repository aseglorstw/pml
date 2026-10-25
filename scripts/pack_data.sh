#!/bin/bash
# Pack everything the cluster needs that is not in git: processed splits (LastFM and Douban-Book,
# all seeds, including pugib/ artifacts) and the LightGCN checkpoints used for warm start.
# Run locally after the overnight queue has finished:  bash scripts/pack_data.sh
# Unpack on the cluster in the project folder:        tar -xzf pugib_data.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-pugib_data.tar.gz}
for d in data/processed/lastfm data/processed/douban_book results/checkpoints; do
  [ -d "$d" ] || { echo "missing $d"; exit 1; }
done
COPYFILE_DISABLE=1 tar -czf "$OUT" data/processed results/checkpoints
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT" | awk -F/ '{print $1"/"$2"/"$3}' | sort | uniq -c
