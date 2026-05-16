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
