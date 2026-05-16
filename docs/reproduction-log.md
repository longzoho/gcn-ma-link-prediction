# Reproduction Log

Living document of every choice that differs from or extends what the paper specifies.

## Plan 1: Foundation (GCN_MA on CollegeMsg)

### Hyperparameters not specified in the paper

| Param | Value | Source |
|---|---|---|
| `hidden_dim` | 128 | EvolveGCN convention; fits 12GB RTX 3060 |
| `num_heads` | 8 | Transformer convention; 128/8 = 16 head dim |
| `lr` | 1e-3 | Adam default for GCN family |
| `optimizer` | Adam | Standard |
| `weight_decay` | 1e-5 | Light regularization, standard |
| `epochs` | 200 with patience 20 | Standard for dynamic LP |
| `dropout` | 0.1 (attention + MLP) | Standard |
| `grad_clip_max_norm` | 5.0 | Anti-explode for LSTM cell |
| `β` | 0.8 (fixed for Plan 1) | Paper recommends [0.7, 0.9]; Plan 2 will grid-search |

### Choices beyond paper

- **Negative sampling:** Uniform random with rejection. 1:1 positive:negative ratio. Per-epoch resampling for training, fixed for validation and test (seed 999). Shared across all models.
- **Test pairs cached with seed 999** to keep evaluation identical when other models are added in later plans.
- **Node features for GCN_MA:** `[degree, CC, AS]` per snapshot, recomputed every snapshot. Baselines (later plans) will use one-hot identity.
- **Train/val/test split (spec §6.4):** training targets snapshots `[1, ⌊0.8T⌋)`; val target is snapshot `⌊0.8T⌋`; test targets are snapshots `[⌊0.8T⌋+1, T-1]`. Pooled AUC/AP across all test steps. Training never targets the val or test snapshots (no leakage).

### Resolved issues

(populate as bugs are found and fixed)

### Result — CollegeMsg, seed 42 (Plan 1 closing)

Full training (epochs ≤ 200, early stop patience 20, hidden_dim=128, num_heads=8, β=0.8) on RTX 3060:

| Metric | Our run | Paper Table 2 | Δ |
|---|---|---|---|
| Test AUC | **0.9024** | 0.9149 | -0.0125 (-1.4% rel.) |
| Test AP  | **0.9201** | 0.8926 | +0.0275 (+3.1% rel.) |
| Val AUC  | 0.9530 | — | — |
| Best epoch | 4 (early stop at 24) | — | — |
| Wall-clock | 61 s | — | — |

Within 1.5% of paper AUC on first run with a single seed; AP exceeds paper. Reasonable reproduction. Plan 2 will:
- Sweep seeds {42, 123, 2024} to report mean±std.
- Grid-search β ∈ {0.7, 0.8, 0.9} on validation.
- Extend to the remaining 5 datasets.

Pipeline notes from the closing run:
- Early stop fired at epoch 24 (no improvement past epoch 4). Model converges fast on CollegeMsg.
- `tee` to `results/logs/<file>.log` requires `mkdir -p results/logs` first — convenience-only, training itself only writes to `results/metrics.jsonl` and the checkpoint.

### Known limitations

- Hyperparameters tuned on CollegeMsg only; later plans will validate on additional datasets.
- TBPTT not yet implemented; will be added when reaching EUT (T=127) in Plan 2.
- `train_ratio` is hardcoded to 0.8 in `trainer.py` and `train.py` even though `configs/datasets/collegemsg.yaml` declares it. Fix in Plan 2 when adding more datasets.
- `pairwise_aggregation` is an O(N²·d_avg) Python loop; acceptable for CollegeMsg (N≈2k) but will need vectorization or sparsification before EUT/LastFM in Plan 2.
- Negative sampling treats edges as directed: for an undirected protocol the reverse `(v, u)` of a positive `(u, v)` is currently eligible as a negative. Conservative bias — flagged for Plan 2 if needed.

---

## Plan 2: Scale-out (6 datasets × 3 seeds, β tuning)

### Chosen hyperparameters (after β grid on Bitcoinotc, seed 42, 50 epochs)

| β   | hidden_dim | val_auc | val_ap  | test_auc | test_ap | best_epoch |
|-----|-----------|---------|---------|----------|---------|------------|
| 0.7 | 64        | 0.9066  | 0.9134  | 0.8543   | 0.8964  | 1          |
| 0.7 | 128       | 0.9343  | 0.9268  | 0.8785   | 0.9086  | 15         |
| **0.8** | **64** ⭐ | **0.9356** | **0.9331** | 0.8785   | 0.9086  | 1          |
| 0.8 | 128       | 0.9135  | 0.9123  | 0.8630   | 0.9006  | 15         |
| 0.9 | 64        | 0.9336  | 0.9299  | 0.8658   | 0.9033  | 17         |
| 0.9 | 128       | 0.9251  | 0.9242  | 0.8791   | 0.9092  | 26         |

**Best by val_auc:** β = 0.8, hidden_dim = 64 (val_auc = 0.9356).

Note: hidden_dim=64 combos converged at epoch 1, suggesting quick convergence (or potentially overfit to initial weights for this seed). Multi-seed run validated stability — all 3 seeds produced consistent test_auc on Bitcoinotc.

Updated `configs/models/gcn_ma.yaml`: hidden_dim 128 → 64, num_heads 8 → 4. All dataset configs kept β=0.8.

### Engineering changes in Plan 2

- **Vectorized `pairwise_aggregation`** via `A @ A` dense matmul (`src/data/preprocess.py`). CollegeMsg N=1899 preprocessing drops from ~30s (Plan 1 Python loop) to ~300ms. Plan 1 carry-forward fixed.
- **Wired `train_ratio`** from each dataset's YAML through `TrainConfig` to `temporal_split` (`src/training/trainer.py`, `scripts/train.py`). Plan 1 carry-forward fixed.
- **Disk cache** stores β-independent features and edge_index per snapshot, keyed by `(raw_file_md5, preprocess_version, cache_format_version)` (`src/data/cache.py`, `src/data/loaders/_base.py`). After format optimization, CollegeMsg cache is 2 MB (down from 648 MB).
- **SNAPTemporalLoader base** extracted from CollegeMsg (`src/data/loaders/_base.py`). 5 new loaders inherit it: Bitcoinotc, EUT, Mooc-actions, LastFM, Wikipedia.
- **Quantile binning option** added to base, applied only to EUT (`binning_strategy = "quantile"`). EUT email activity is bursty; equal-time bins at T=127 produced 24 consecutive empty snapshots in the test region (val_auc stuck at 0.5000). Quantile binning gives equal-event-count bins → all snapshots populated → AUC jumps from 0.535 to 0.90.
- **Empty-validation handling** in `trainer.py` and `evaluator.py`: walk forward to first non-empty val target if `val_step+1` is empty; defensive skip of empty pairs in evaluator. `sample_negative_edges` returns shape `(2, 0)` when `num_samples=0` (was returning `(0,)`).
- **Bipartite loaders** (Mooc, LastFM, Wikipedia) shift item IDs by `max(user_id) + 1` before dense remap to keep user/item ID ranges disjoint.
- **JODIE CSV parsing**: `usecols=[0, 1, 2]` reads only positional columns (the CSV header declares 5 columns but the data has 8/172 trailing feature columns that pandas otherwise misaligns).

### Final results — 6 datasets × 3 seeds {42, 123, 2024}

Best (β=0.8, hidden_dim=64, num_heads=4) applied to all datasets. Test AUC/AP pooled across test snapshots. Reported mean ± std over 3 seeds.

| Dataset      | n | AUC (mean ± std)  | AP (mean ± std)   | Paper AUC | Paper AP | Δ AUC   | Δ AP    |
|--------------|---|-------------------|-------------------|-----------|----------|---------|---------|
| collegemsg   | 3 | 0.9005 ± 0.0002   | 0.9181 ± 0.0009   | 0.9149    | 0.8926   | -0.0144 | +0.0255 |
| bitcoinotc   | 3 | 0.8560 ± 0.0054   | 0.8966 ± 0.0023   | 0.9120    | 0.8943   | -0.0560 | +0.0023 |
| eut          | 3 | 0.9008 ± 0.0016   | 0.9021 ± 0.0020   | 0.9222    | 0.9082   | -0.0214 | -0.0061 |
| mooc_actions | 3 | 0.9845 ± 0.0002   | 0.9751 ± 0.0008   | 0.9880    | 0.9863   | -0.0035 | -0.0112 |
| lastfm       | 3 | 0.8004 ± 0.0040   | 0.7836 ± 0.0071   | 0.8757    | 0.8704   | -0.0753 | -0.0868 |
| wikipedia    | 3 | 0.8696 ± 0.0007   | 0.8914 ± 0.0006   | 0.8742    | 0.8575   | -0.0046 | +0.0339 |

**Highlights:**

- **Strong matches** (Δ AUC within ±1.5%): Mooc-actions (-0.4%), Wikipedia (-0.5%), CollegeMsg (-1.4%).
- **Acceptable** (Δ AUC ~2%): EUT (-2.1%) — quantile binning was critical.
- **Notable gaps**: Bitcoinotc (-5.6% AUC), LastFM (-7.5% AUC).
- **AP beats paper** on CollegeMsg (+2.6%), Bitcoinotc (+0.2%), and Wikipedia (+3.4%).
- **Std consistently below 0.01** across all datasets → multi-seed convergence is stable. Bitcoinotc and LastFM have the largest std (~0.005), still small.

### Possible causes of remaining gaps

- **hidden_dim=64 chosen on Bitcoinotc** via β grid (best by val_auc). For datasets where 128 may have helped (LastFM in particular — it has 1.29M edges and might benefit from more capacity), we're under-parameterized.
- **Negative sampling protocol unknown in paper**. Paper does not specify ratio, rejection strategy, or fixed-vs-resample.
- **Bipartite ID convention** for Mooc / LastFM / Wikipedia (we shift items by max_user+1; paper might union without shift, which changes `num_nodes` and thus normalized adjacency magnitudes).
- **Quantile binning is unique to EUT** — paper likely uses a uniform strategy across all 6 datasets, but it isn't specified.

### Bugs caught and fixed in Plan 2

- **`pd.read_csv` column misalignment** for JODIE CSVs (Mooc, LastFM, Wikipedia): header declares 5 columns, data has 8/172. Fix: `usecols=[0, 1, 2]` (`b27bde7`).
- **Dense `S` matrices cached**: 14 GB cache for Mooc, intolerable I/O on WSL2 `/mnt/d/`. Fix: cache only edge_index + features, recompute `S = (A @ A) · AS` at load time (~50 ms / snapshot). Caches shrank 100-1000× (`27de8b8`).
- **Mooc parser misalignment** as above (`b27bde7`).
- **`sample_negative_edges` with `num_samples=0`** returned shape `(0,)` instead of `(2, 0)`, crashing decoder on empty validation snapshots (`6421868`).
- **Empty validation snapshot handling** in trainer + evaluator (`6421868`).
- **EUT empty-bin pathology** under equal-time binning. Fix: per-loader `binning_strategy = "quantile"` (`49c6819`).
- **RTK proxy corrupted `results/metrics.jsonl`** during a `grep | mv` pipeline — recovered all 15 prior records by re-parsing the per-run log files. Lesson: always work on a copy when rewriting an authoritative file.

### Operational notes from this plan

- β grid search on Bitcoinotc (6 combos × 50 epochs): **79 min** wall-clock on RTX 3060.
- Full 18-run experiment (6 datasets × 3 seeds): **~5.5 hours** wall-clock, dominated by Mooc-actions (~30-45 min per seed) and Bitcoinotc (~10-15 min per seed).
- Cache invalidation by `preprocess_version` bump worked cleanly when EUT switched to quantile binning.

### Carry-forwards to Plan 3 (baselines integration)

- Bitcoinotc and LastFM gaps justify tuning per-dataset (currently using one shared config). Could grid-search on each dataset, or expose `hidden_dim` per dataset YAML.
- Bipartite node-count discrepancy vs paper: our LastFM has 1980 nodes (980 users + 1000 items) but paper reports 1000. Likely a "users only" convention in the paper; document in thesis.
- LastFM AUC of 0.80 vs paper's 0.876 is the biggest gap. Worth a focused investigation in Plan 3 — maybe an ablation isolating the cause.
- Test pair caching (seed=999) currently happens on every CLI run. For Plan 3 baselines, cache `test_pairs.pt` so all models evaluate on the exact same pairs.

---

## Plan 3a: EvolveGCN-O baseline integration

### Approach

Vendored [`IBM/EvolveGCN`](https://github.com/IBM/EvolveGCN) as a git submodule at pinned commit `9086906` (see `.gitmodules`). Wrote a thin adapter `src/models/evolvegcn.py` (~165 LOC) extending `DynamicLinkPredictor`. Reuses Plan 1/2 trainer, evaluator, negative sampling, loaders.

### Hyperparameter policy (Hybrid)

Shared with GCN_MA: `lr=1e-3`, `weight_decay=1e-5`, Adam, `epochs=200`, patience 20, `dropout=0.1`, `grad_clip_max_norm=5.0`, `hidden_dim=64`.
EvolveGCN-specific: `num_layers=2` (upstream hardcoded), `activation=RReLU`, `feat_dim=64` (learnable embedding instead of one-hot identity).

### Deviation from spec §6.6

Spec called for one-hot identity `I_N` as baseline input feature, but RAM cost is prohibitive for Bitcoinotc / Mooc / Wikipedia (34-59 GB). Replaced with `nn.Embedding(N, feat_dim)` Xavier-initialized — same convention as IBM/EvolveGCN's own code for large-N cases.

### Upstream EGCN bugs patched (PyTorch 2.4 compat)

The upstream `EGCN.__init__` had two issues that broke standard `nn.Module` machinery under PyTorch 2.4:

1. `self._parameters` was overwritten with `nn.ParameterList`, breaking `Module._apply()` / `.to(device)`.
2. `GRCU_layers` was stored as a plain Python list, invisible to parameter traversal.

`_patch_upstream_egcn()` in `src/models/evolvegcn.py` fixes both by promoting `GRCU_layers` to `nn.ModuleList` and restoring `_parameters` to `{}`. 0 changes made to upstream code.

### Results — 3 of 6 datasets

EvolveGCN-O reproduced successfully on CollegeMsg, Bitcoinotc, EUT:

| Dataset      | GCN_MA AUC      | EvolveGCN-O AUC | Δ (GCN_MA − EvolveGCN-O) |
|--------------|-----------------|------------------|--------------------------|
| collegemsg   | 0.9005 ± 0.0002 | 0.8259 ± 0.0010  | **+0.075** ⭐ |
| bitcoinotc   | 0.8560 ± 0.0054 | 0.8139 ± 0.0204  | **+0.042** ⭐ |
| eut          | 0.9008 ± 0.0016 | 0.8620 ± 0.0017  | **+0.039** ⭐ |

GCN_MA wins on all 3 datasets — expected direction (paper's contribution: NRNAE + attention beats vanilla EvolveGCN).

### Known limitation: bipartite datasets

EvolveGCN-O produced **`Z = all zeros`** on the 3 bipartite datasets (LastFM, Mooc-actions, Wikipedia), yielding val_auc = 0.5000 every epoch and best_epoch = 0. Root cause investigation (see `results/logs/evolvegcn_o_lastfm_seed42_*.log` and ad-hoc diagnostics):

- Upstream EGCN forward returns zero output even at single-snapshot input (`time_step=0`).
- Gradient norms are exactly 0 on all `core` parameters and on `node_emb`. Only `decoder.bias` gets gradient — model can't distinguish positive from negative pairs because Z is constant.
- For comparable non-bipartite N (CollegeMsg N=1899 vs LastFM N=1980), CollegeMsg produces non-zero Z starting at `time_step=1`.
- Conjecture: upstream's directed sparse adjacency + Xavier-init scaling + bipartite block-diagonal structure (`A^T A` is block-diagonal) cause RReLU to saturate to zero. Items have zero in-degree in our adjacency construction, so their rows in `A @ X @ W` are exactly zero.

**Stale entries removed** from `results/metrics.jsonl` (9 records: lastfm, mooc_actions, wikipedia × 3 seeds). Cross-model table shows `—` for these.

**Fix candidates** (not pursued in Plan 3a, deferred to follow-up):
- Symmetrize adjacency: `A = A + A^T` so item rows have signal too.
- Larger node embedding init (e.g., `nn.init.uniform_(-1, 1)` instead of Xavier).
- Truncated window (process only last K snapshots).
- Use NRNAE features as input (departs from "baselines vanilla" but enables apples-to-apples comparison).

### Final 27 metric records

15 GCN_MA (Plan 2) + 9 EvolveGCN-O (Plan 3a partial success) = 27 in `results/metrics.jsonl`. Plan 1 archive (1 record at hidden_dim=128) remains in `results/metrics_plan1_hidden128.jsonl`.

Wait — actually 18 GCN_MA from Plan 2. Let me recount:
- 18 GCN_MA (6 datasets × 3 seeds, Plan 2)
- 9 EvolveGCN-O (3 working datasets × 3 seeds, Plan 3a)
- **27 total** ✓

### Carry-forwards to Plans 3b/c/d

- `_build_model` dispatch in `scripts/train.py` is extensible — same pattern for HTGN, DyGNN, DGCN.
- `aggregate_results.py --models` works for arbitrary model lists.
- Bipartite failure for EvolveGCN should be investigated before reporting Plan 4 final numbers — fix or document properly.
- The `_patch_upstream_egcn()` PyTorch 2.4 shim approach may be needed for other 2019-2020 upstream baselines.
