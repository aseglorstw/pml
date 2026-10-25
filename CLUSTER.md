# Running on MetaCentrum (A40 GPU, PBS)

Python module, resource syntax and scratch handling follow the MetaCentrum recipe. Lines marked **CHECK**
(frontend host, optional A40 property) still need a look at the MetaCentrum documentation.

## 1. Locally (Mac)

```bash
cd ~/Projects/CTU/PML/pu_gib
# code -> private GitHub repo (create an empty private repo on GitHub first)
git remote add origin git@github.com:<you>/pu_gib.git
git push -u origin main

# data + warm-start checkpoints -> one archive (after the overnight queue has finished)
bash scripts/pack_data.sh                      # writes pugib_data.tar.gz
scp pugib_data.tar.gz <user>@<frontend>.metacentrum.cz:~/      # CHECK frontend name
```

## 2. On the cluster: one-time setup

```bash
ssh <user>@<frontend>.metacentrum.cz           # CHECK frontend name
git clone git@github.com:<you>/pu_gib.git      # needs a GitHub SSH key (or a token) on the cluster
cd pu_gib                                      # keep this directory private (chmod 700)
tar -xzf ~/pugib_data.tar.gz                   # creates data/processed/ and results/checkpoints/

# (a) install: interactive CPU job with scratch (MetaCentrum recipe)
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I
cd ~/pu_gib && bash scripts/cluster_setup.sh   # module add python/3.11.11-gcc-10.2.1-555dlyc, venv in .venv,
exit                                           # torch 2.14.1 CUDA build, requirements, CPU tests

# (b) GPU check: short interactive GPU job
qsub -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=8gb -l walltime=0:30:00 -I
cd ~/pu_gib && bash scripts/cluster_test_gpu.sh   # nvidia-smi, CUDA-vs-CPU tests, epoch timings
exit
```

The scripts load the Python module, use `$SCRATCHDIR` as `TMPDIR` and call `clean_scratch` on exit.
`cluster_setup.sh` tries the cu128 build of torch 2.14.1, then cu118, then plain PyPI (override with
`TORCH_VERSION=... TORCH_INDEXES="..."`).

## 3. Submitting runs

One job per line of a job list (`<walltime> <config> [arguments]`), see `configs/jobs/example.txt`:

```bash
bash scripts/submit_grid.sh configs/jobs/example.txt --dry-run   # print the qsub commands
bash scripts/submit_grid.sh configs/jobs/example.txt             # submit
qstat -u $USER                                                    # monitor
```

- Resources are at the top of `scripts/job.pbs`: `select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=10gb`,
  walltime per line, no explicit queue (MetaCentrum routes GPU jobs). To force an A40, add the GPU-cluster
  property to the `select` line (**CHECK** the "GPU clusters" page).
- Each job writes its own CSV `results/jobs/<jobid>.csv` and log `results/logs/job_<jobid>.log`;
  PBS stdout goes to `results/pbs/`.
- Single run without a list: `qsub -l walltime=02:00:00 -v CONFIG=configs/pugib/lightgcn_s_warm.yaml,ARGS="--seeds 0" scripts/job.pbs`
  (no commas inside `ARGS`; use a job list for anything longer).
- Devices: configs say `device: auto` (cuda > mps > cpu); `job.pbs` passes `--device cuda`, which falls back to CPU if no GPU is visible.
- Warm start reads `results/checkpoints/lightgcn_<dataset>_seed<k>.pt` from the archive. PU-GIB and LightGCN-PU
  read `data/processed/<dataset>/seed_<k>/pugib/`.

## 4. Collecting results

```bash
python scripts/merge_results.py            # results/jobs/*.csv -> results/results.csv (dedup, backup kept)
python scripts/summarize.py                # or summarize_step3.py / overnight_summary.py
# back to the Mac
scp <user>@<frontend>.metacentrum.cz:~/pu_gib/results/results.csv results/results_cluster.csv
```

## 5. Running inside an interactive (OnDemand / GUI) GPU session

One A40 easily holds several of our runs at once. Run a job list with N runs in parallel on the
session's GPU (survives closing the terminal thanks to nohup, but NOT the end of the session):

```bash
cd ~/pml
mkdir -p results/logs                                 # needed once: the shell opens the log before the script runs
nohup bash scripts/run_jobs_local.sh configs/jobs/example.txt 3 > results/logs/local_runner.log 2>&1 &
tail -f results/logs/local_runner_status.log          # STARTED / FINISHED / FAILED per run
nvidia-smi                                            # GPU memory / utilisation
python scripts/merge_results.py                       # afterwards: per-run CSVs -> results/results.csv
```

Keep N ≤ the number of CPUs of the session (`echo $PBS_NCPUS`), since sampling and evaluation use the CPU
(each run gets `OMP_NUM_THREADS=2` by default). For runs longer than the session, submit batch jobs from the
same terminal with `bash scripts/submit_grid.sh <list>`; they run independently of the GUI session.

## 6. Douban-Book pipeline (PBS jobs with dependencies)

```bash
cd ~/pml && git pull && source .venv/bin/activate
bash scripts/submit_douban_pipeline.sh --dry-run    # print the qsub commands
bash scripts/submit_douban_pipeline.sh              # submit all stages
qstat -u $USER                                      # 'H' = held, waiting for its dependency
```
Stages: 0 GPU benchmark → 1 cold LightGCN seeds 0-2 (parallel) → 2 PU-GIB preprocessing seeds 0-2 (CPU,
after all of stage 1 succeeded) → 3 ten seed-0 runs (parallel, after stage 2) → 4 merge + summary (after stage 3).
Results: `results/douban_benchmark_cuda.txt`, `results/douban_book_prep_summary.md`,
`results/douban_stage3_summary.md`. If a stage fails, its dependents stay held: delete them with
`qdel`, fix, and resubmit from that stage with `--from 2` or `--from 3`.

Schedule used by every warm-started run (configs/pugib, configs/gbsr): warm start from the LightGCN checkpoint,
`lr_emb = 1e-4` for embeddings and `lr = 1e-3` for the masker, fixed 300-epoch budget with τ annealed
1.0 → 0.2 over it (`tau_epochs: auto`), best validation checkpoint kept, last-epoch metrics also logged.
