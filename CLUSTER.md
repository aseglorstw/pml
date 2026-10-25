# Running on MetaCentrum (A40 GPU, PBS)

Lines marked **CHECK** depend on MetaCentrum specifics (frontend host, storage path, queue,
GPU selection, Python module, CUDA build). Verify them in the MetaCentrum documentation first.

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
cd pu_gib
tar -xzf ~/pugib_data.tar.gz                   # creates data/processed/ and results/checkpoints/

# interactive GPU job for setup + GPU tests                         # CHECK queue/resource syntax
qsub -I -q gpu -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=1:00:00
cd ~/pu_gib                                   # or your storage path
module add python/3.11                        # CHECK exact module name (`module avail python`)
bash scripts/cluster_setup.sh                 # venv + CUDA torch + requirements + tests + timing
exit
```

`cluster_setup.sh` takes `PYTHON`, `TORCH_VERSION`, `TORCH_INDEX` (CUDA build, default cu128 — **CHECK**
with `nvidia-smi`) from the environment, e.g. `TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/cluster_setup.sh`.
The test suite includes `tests/test_device.py`, which compares CUDA results with CPU results.

## 3. Submitting runs

One job per line of a job list (`<walltime> <config> [arguments]`), see `configs/jobs/example.txt`:

```bash
bash scripts/submit_grid.sh configs/jobs/example.txt --dry-run   # print the qsub commands
bash scripts/submit_grid.sh configs/jobs/example.txt             # submit
qstat -u $USER                                                    # monitor
```

- Resources and queue are at the top of `scripts/job.pbs` (1 GPU, 4 CPUs, 16 GB, walltime per line). **CHECK**
  the queue name and how to request an A40 (property on the `select` line).
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
