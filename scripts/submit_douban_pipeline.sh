#!/bin/bash
# Douban-Book pipeline on MetaCentrum as PBS jobs chained with dependencies:
#   stage 0: GPU epoch-time benchmark (no CPU part)
#   stage 1: cold LightGCN, seeds 0 1 2 (3 parallel GPU jobs; saves warm-start checkpoints)
#   stage 2: PU-GIB preprocessing for seeds 0 1 2 (CPU job, after ALL stage-1 jobs succeed)
#   stage 3: seed-0 comparison, 10 GPU jobs (after stage 2), new schedule from the configs
#   stage 4: merge per-job CSVs + summary table (after all stage-3 jobs, even failed ones)
# Usage:  bash scripts/submit_douban_pipeline.sh [--dry-run] [--from 1|2|3]
#   --from 2 / --from 3 skips the earlier stages (their outputs must already exist).
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; FROM=1
while [ $# -gt 0 ]; do
  case $1 in --dry-run) DRY=1;; --from) FROM=$2; shift;; *) echo "unknown $1"; exit 1;; esac; shift
done
P=$(pwd)
mkdir -p results/jobs/specs results/pbs results/logs
TS=$(date +%m%d%H%M)

sub() {  # name walltime script spec_content [depend] [select]
  local name=$1 wall=$2 script=$3 content=$4 dep=${5:-} sel=${6:-}
  local ext=args; [ "$script" = scripts/job_cmd.pbs ] && ext=cmd
  local spec="results/jobs/specs/${name}_${TS}.${ext}"
  echo "$content" > "$spec"
  local args=(-N "$name" -l "walltime=$wall" -o "results/pbs/${name}_${TS}.out" -v "SPEC=$spec,PROJECT=$P")
  [ -n "$dep" ] && args+=(-W "depend=$dep")
  [ -n "$sel" ] && args+=(-l "select=$sel")
  if [ $DRY = 1 ]; then echo "qsub ${args[*]} $script    # $content" >&2; echo "DRY_${name}"
  else qsub "${args[@]}" "$script"; fi
}

DB="--set dataset=douban_book data_dir=data/processed/douban_book reg=0.0001 eval_every=10"
S2=""
if [ "$FROM" -le 1 ]; then
  sub db_bench 00:30:00 scripts/job_cmd.pbs \
    "python -u scripts/benchmark_device.py --dataset douban_book --epochs 2 --devices cuda | tee results/douban_benchmark_cuda.txt" \
    "" "1:ncpus=2:ngpus=1:mem=16gb:scratch_local=10gb" >/dev/null
  IDS=()
  for s in 0 1 2; do
    IDS+=("$(sub db1_lightgcn_s$s 08:00:00 scripts/job.pbs "configs/douban_book_lightgcn.yaml --no-tune --seeds $s $DB")")
  done
  DEP1="afterok:$(IFS=:; echo "${IDS[*]}")"
fi
if [ "$FROM" -le 2 ]; then
  S2=$(sub db2_prep 03:00:00 scripts/job_cmd.pbs \
    "python -u scripts/prepare_pugib.py --dataset douban_book --variants best --seeds 0 1 2 && python scripts/summarize_prep.py --dataset douban_book" \
    "${DEP1:-}")
fi
DEP3=""; [ -n "$S2" ] && DEP3="afterok:$S2"
P3="--no-tune --seeds 0 $DB"
PG="kappa=0.05 norm=prior"
IDS3=()
add3() { IDS3+=("$(sub "$1" 12:00:00 scripts/job.pbs "$2" "$DEP3")"); }
add3 db3_lightgcn_warm          "configs/pugib/lightgcn_warm.yaml $P3 name=db3_lightgcn_warm"
add3 db3_lightgcn_s_warm        "configs/pugib/lightgcn_s_warm.yaml $P3 name=db3_lightgcn_s_warm"
add3 db3_lightgcn_s_omega025    "configs/pugib/lightgcn_s_warm.yaml $P3 name=db3_lightgcn_s_warm_omega025 omega_s=0.25"
add3 db3_gbsr                   "configs/gbsr/lastfm_gbsr.yaml $P3 name=db3_gbsr beta=40 sigma=2.5 detach_mask=false"
add3 db3_lightgcn_pu_s          "configs/pugib/lightgcn_pu_s.yaml $P3 name=db3_lightgcn_pu_s"
add3 db3_pugib_anchor_b0        "configs/pugib/pugib_s_anchor.yaml $P3 name=db3_pugib_s_anchor_b0 $PG beta=0"
add3 db3_pugib_anchor_b0001     "configs/pugib/pugib_s_anchor.yaml $P3 name=db3_pugib_s_anchor_b0001 $PG beta=0.001"
add3 db3_pugib_anchor_b001      "configs/pugib/pugib_s_anchor.yaml $P3 name=db3_pugib_s_anchor_b001 $PG beta=0.01"
add3 db3_pugib_unif_b0001       "configs/pugib/pugib_s_unif.yaml $P3 name=db3_pugib_s_unif_b0001 $PG beta=0.001"
add3 db3_pugib_ra_anchor_b0001  "configs/pugib/pugib_ra_anchor.yaml $P3 name=db3_pugib_ra_anchor_b0001 $PG beta=0.001"
sub db4_summary 00:30:00 scripts/job_cmd.pbs \
  "python scripts/merge_results.py && python scripts/summarize_runs.py --prefix db3_ --out results/douban_stage3_summary.md" \
  "afterany:$(IFS=:; echo "${IDS3[*]}")" "1:ncpus=1:mem=8gb:scratch_local=1gb" >/dev/null
echo "submitted (dry=$DRY). Monitor: qstat -u \$USER ; outputs: results/douban_*summary.md" >&2
