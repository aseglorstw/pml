# PU-GIB: experimental foundation

Shared data pipeline, splits, metrics and baselines (Popularity, LightGCN, LightGCN-S)
for the PU-GIB project described in `bakhtrob.tex`. PU-GIB itself is not implemented yet.

## Setup

```bash
brew install uv                      # or any way to get uv
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

## Reproduce

```bash
# 1. Download LastFM (HetRec 2011), preprocess, write stats + splits for seeds 0,1,2
.venv/bin/python scripts/prepare_data.py

# 2. Tests (splits disjoint, social graph symmetric, metrics, eq. 8 propagation)
.venv/bin/python -m pytest -q

# 3. Optional: CPU vs MPS timing (writes nothing; output saved in results/device_benchmark.txt)
.venv/bin/python scripts/benchmark_device.py

# 4. Experiments (each appends rows to results/results.csv)
.venv/bin/python scripts/run_experiment.py configs/lastfm_popularity.yaml
.venv/bin/python scripts/run_experiment.py configs/lastfm_lightgcn.yaml
.venv/bin/python scripts/run_experiment.py configs/lastfm_lightgcn_s.yaml

# 5. Mean ± std over seeds -> results/summary.md
.venv/bin/python scripts/summarize.py

# 6. PU-GIB preprocessing (needs results/checkpoints/lightgcn_lastfm_seed*.pt from step 4)
.venv/bin/python scripts/prepare_pugib.py
```

Long runs can be detached from the terminal, e.g.
`nohup .venv/bin/python -u scripts/run_experiment.py configs/lastfm_lightgcn.yaml > results/logs/lightgcn.log 2>&1 &`.

Final LightGCN/LightGCN-S runs save their best checkpoint to `results/checkpoints/` and
LightGCN-S writes per-layer norms of the interaction and social terms to `results/diagnostics/`.
The tuning run of the selected L2 weight is reused as the seed-0 final run (identical config and seed).

## PU-GIB preprocessing outputs (`data/processed/lastfm/seed_k/pugib/`)

| File | Contents |
|---|---|
| `omega.npz` | `pairs` (i<j, sorted), `y` = A_train label, `in_pval` (held-out edge), `r` (Resource Allocation) |
| `graph_stats.json` | \|Ω\|, p_A, training / held-out bridges, base rate |
| `pool_ra.txt` | top-K candidate pool ranked by RA |
| `<variant>/` | `best` = best-val LightGCN checkpoint, `ep20` = LightGCN after 20 epochs |
| `<variant>/embeddings.pt` | ego and final embeddings used as ẽ |
| `<variant>/s.npy`, `g.npy` | cosine similarity and isotonic g on Ω (aligned with `omega.npz`) |
| `<variant>/priors.json` | p_A, g_val ± SE, informativeness test, estimated and uninformed priors per κ |
| `<variant>/pool_s.txt`, `pool_random.txt`, `support.txt` | candidate pools and support C = E_train ∪ pool_s |
| `<variant>/report.json` | pool recall/precision/lift, support degrees, similarity AUROCs |

Figures: `results/figures/lastfm_seed{k}_{variant}.png`. All seeds: `results/pugib_prep_summary.json`.

`--device cpu` overrides the config device. `configs/smoke.yaml` is a 10-epoch check that
writes to `results/smoke.csv`.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | download, implicit feedback, symmetric social graph, id remapping, statistics |
| `src/splits.py` | per-user 80/10/10 interaction split, 10% social hold-out, save/load |
| `src/metrics.py` | full-ranking Recall@K / NDCG@K with train (and val) exclusion |
| `src/models/lightgcn.py` | LightGCN, LightGCN-S (social weights over an edge list), Popularity |
| `src/train.py` | BPR training, early stopping, checkpoints, results logging |
| `src/pugib_prep.py` | Ω, RA, cosine, isotonic g, channel estimates and priors, candidate pools |
| `configs/` | one YAML per experiment |
| `data/processed/lastfm/seed_k/` | `train/val/test.txt` (user item), `social_train/social_val.txt` (u v, u<v), `meta.json` |
| `results/results.csv` | one row per run |

## Douban-Book

Source: the dataset files released with GBSR (Yang et al., KDD 2024),
<https://github.com/yimutianyang/KDD24-GBSR> (commit `881d712`), `datasets/douban_book/`.
Clone it into `data/raw/gbsr_repo/src`, then `python scripts/prepare_data.py --dataset douban_book`.

- The release is already filtered and binarized; the GBSR paper only says it follows a previously
  released version and gives no filtering rule. We take it as is.
- The files are pickled `.npy` dicts. They are read with a restricted unpickler
  (`src/data.py: safe_load_pickled_npy`) that only allows numpy/list/set/dict types.
- GBSR's own 80/20 train/test split is **not** used: we merge their train and test parts and
  re-split with our protocol (per-user 80/10/10, 10% social hold-out, seeds 0-2). Our numbers are
  therefore not directly comparable to the GBSR paper.
- The release has 598,420 unique interactions, while the GBSR paper's Table 1 reports 792,062;
  the 169,150 directed social rows match. Users without interactions are dropped.

```bash
nohup bash scripts/run_douban_step2.sh > results/logs/douban_step2.log 2>&1 &   # baselines + PU-GIB prep
```

## Warm start (default for social methods)

Every social method (LightGCN-S, LightGCN-PU, PU-GIB, GBSR, later SHaRe) starts its ego embeddings
from the best-validation LightGCN checkpoint of the same dataset and seed
(`results/checkpoints/lightgcn_<dataset>_seed<k>.pt`). Plain LightGCN starts from random init.
Override with `init: random` (the cold LightGCN-S baseline configs set this explicitly).
Reason: on LastFM a warm-started LightGCN-S gains as much as every PU-GIB variant, so
cold-started baselines would credit the warm start to the method.

## GBSR port (`src/models/gbsr.py`, `configs/gbsr/`)

Ported from the official code (<https://github.com/yimutianyang/KDD24-GBSR>, `torch_version/GBSR.py`)
onto our LightGCN-S backbone and protocol. Notes on the official code:
- The PyTorch masker gets **no gradient**: `graph_learner` calls `weights = weights.detach()`
  before building the masked graph. In the TensorFlow version the gates go through
  `tf.scatter_nd_update` on a non-trainable variable, which TF1 also treats as non-differentiable
  (not verified by running TF). `detach_mask: true` reproduces this; the default `false` lets the
  gradient reach the masker, as the paper describes.
- The official loader reads `testdata.npy` as validation data, so early stopping and model
  selection there use the test set. We use our validation split.
- Official evaluation re-samples the noisy mask; we evaluate with the noise-free gate.

## Protocol

- **Data.** `user_artists.dat` → any listening count = 1. `user_friends.dat` → symmetrized,
  self-loops and duplicates removed, only users with interactions. Ids remapped to 0..M-1, 0..N-1.
- **Splits.** Per user, random 80/10/10 (rounded; at least one train item is kept).
  Independently, a uniform 10% of undirected social edges is held out as `social_val` (E_val).
  Both use seeded generators (`[seed, 0]` and `[seed, 1]`), so they are reproducible and independent.
- **Metrics.** Ranking over all items. Validation excludes train items; test excludes train
  and val items. Averaged over users with ≥1 held-out item. K ∈ {10, 20}.
- **LightGCN.** d=64, L=3, mean over layers 0..L, BPR with one uniform negative per
  training interaction, batch 2048, Adam lr 1e-3. L2 term = `reg · ½‖ego embeddings of batch‖² / batch`
  (LightGCN reference convention), `reg` tuned in {1e-4, 1e-3} on seed-0 validation.
- **LightGCN-S** (eq. 8): user update adds `ω_S Σ_v w_uv / sqrt(d_u d_v) e_v` over training social
  edges, with ω_S = 0.5 and d_u the static number of social edges of u. `LightGCN.propagate(social_w)`
  takes a weight per undirected edge (shape `(E,)`, differentiable), `None` = all ones.
- **Training.** Validate every 5 epochs; early stop on val NDCG@20, patience 10 evaluations,
  max 1500 epochs; test metrics come from the best validation checkpoint only.
  L2 grid {1e-5, 1e-4, 1e-3}.
- **results.csv columns.** timestamp, method, dataset, seed, hparams (JSON, including device,
  s/epoch, and `stage` = tune/final), best_epoch, val_ndcg@20, test recall/ndcg @10/@20, runtime_s.

## Cluster (MetaCentrum)

See [CLUSTER.md](CLUSTER.md): `scripts/cluster_setup.sh`, `scripts/pack_data.sh`, `scripts/job.pbs`, `scripts/submit_grid.sh`, `scripts/merge_results.py`.
