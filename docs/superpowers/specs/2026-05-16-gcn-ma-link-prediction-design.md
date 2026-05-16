# Design Spec — Reproducing GCN_MA for Dynamic Network Link Prediction

**Date:** 2026-05-16
**Author:** long.huynh@siliconprime.com
**Status:** DRAFT (awaiting user review)
**Project:** `gcn-ma-link-prediction`
**Purpose:** Thesis / coursework reproduction with planned improvements (Phase 2)

---

## 1. Reference paper

- **Title:** *Dynamic network link prediction with node representation learning from graph convolutional networks*
- **Authors:** Peng Mei, Yu hong Zhao
- **Venue:** Scientific Reports (Nature), January 2024
- **DOI:** 10.1038/s41598-023-50977-6
- **Open access mirror:** PMC10766634
- **Code availability in paper:** None (no GitHub link). Reimplementation required.

## 2. Goals & non-goals

### Goals (Phase 1 — this spec)

1. Reproduce the **GCN_MA** model end-to-end in PyTorch + PyTorch Geometric.
2. Reproduce **4 baselines** from the paper: EvolveGCN, HTGN, DyGNN, DGCN (fork upstream where possible).
3. Evaluate on **6 datasets** the paper uses: CollegeMsg, Bitcoinotc, Email-EU-temporal (EUT), Mooc-actions, LastFM, Wikipedia.
4. Produce results comparable to Table 2 of the paper (AUC, AP), with 3 seeds, `mean ± std`.
5. Produce thesis-ready deliverables: bảng kết quả Markdown/LaTeX, plots PDF, reproduction log defending every deviation from paper.

### Non-goals (Phase 1)

- Improving the model. That is Phase 2, separate brainstorm.
- Hyperparameter tuning baselines (use authors' defaults — "as reported").
- Implementing additional datasets beyond the 6 in the paper.
- Realtime / streaming inference.

## 3. Constraints

- **Framework:** PyTorch 2.4 + PyTorch Geometric 2.6 (locked).
- **Hardware:** NVIDIA RTX 3060 12GB, local.
- **Python:** 3.11.
- **Timeline:** 8 weeks + 1 week buffer.
- **Code reuse policy:** Fork upstream baseline repos as git submodules under `third_party/`; write GCN_MA from paper. Document fork commit hash in `reproduction-log.md`.

## 4. Approach (selected: A — Monorepo unified)

Single repo, single env, plugin architecture for all models. Shared data loader, trainer, evaluator. Each model is a class implementing the `DynamicLinkPredictor` interface. Baselines wrapped as thin adapters around upstream code in `third_party/`. Configuration is YAML at three levels (dataset × model × experiment).

Rejected: Approach B (per-model envs — env hell, comparison unfair) and Approach C (TGB wrapper — split mismatch breaks reproducibility claim).

## 5. Architecture

### 5.1 Flow

```
CLI (scripts/train.py --config <yaml>)
  → DatasetLoader (per dataset) → DynamicGraph
  → Model (per model) → embeddings Z^t
  → Trainer (shared loop) → checkpoints + metrics.jsonl
  → Evaluator (shared AUC/AP) → results
```

### 5.2 Repo layout

```
gcn-ma-link-prediction/
├── pyproject.toml                # uv-managed deps, Python 3.11
├── .python-version
├── .gitignore                    # .venv/, data/raw/, data/processed/, results/, __pycache__/, *.pt
├── configs/
│   ├── datasets/                 # 1 YAML / dataset (raw path, time_steps, β candidates)
│   ├── models/                   # 1 YAML / model (hyperparams)
│   └── experiments/              # 1 YAML / (dataset, model, seed) tuple
├── src/
│   ├── data/
│   │   ├── base.py               # DynamicGraph dataclass
│   │   ├── loaders/              # 1 file / dataset
│   │   └── preprocess.py         # NRNAE features (CC, AS, S_ij, Ŝ)
│   ├── models/
│   │   ├── base.py               # abstract DynamicLinkPredictor
│   │   ├── gcn_ma/               # gcn_layer, lstm_weight, attention, link_decoder, model
│   │   ├── evolvegcn.py          # adapter wrapping third_party/EvolveGCN
│   │   ├── htgn.py
│   │   ├── dygnn.py
│   │   └── dgcn.py
│   ├── training/
│   │   ├── trainer.py            # shared loop
│   │   ├── losses.py             # BCEWithLogitsLoss wrapper
│   │   └── negative_sampling.py
│   ├── eval/
│   │   ├── evaluator.py
│   │   └── metrics.py            # AUC, AP via sklearn
│   └── utils/
│       ├── seed.py
│       └── logging.py
├── scripts/
│   ├── download_datasets.py
│   ├── train.py                  # CLI entry
│   ├── run_all.sh                # 5 models × 6 datasets × 3 seeds = 90 runs
│   ├── aggregate_results.py      # metrics.jsonl → Markdown/LaTeX tables, plots
│   └── generate_report_assets.py
├── tests/                        # tier 1 (unit), tier 2 (smoke), tier 3 (integration)
├── results/                      # gitignored
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics.jsonl             # append-only
│   ├── plots/
│   ├── tb/                       # optional TensorBoard
│   └── report/                   # auto-generated tables + plots
├── docs/
│   ├── superpowers/specs/        # this file
│   ├── paper-notes.md            # quoted formulas, section refs
│   ├── reproduction-log.md       # every deviation from paper + reason
│   ├── architecture.md
│   └── results.md                # filled at end
└── third_party/                  # git submodules
    ├── EvolveGCN/
    ├── HTGN/
    ├── DyGNN/
    └── DGCN/                     # fallback: reimplement if no upstream repo
```

## 6. Data layer

### 6.1 Datasets

| Dataset | Nodes | Edges | T (snapshots) | Source |
|---|---|---|---|---|
| CollegeMsg | 1,899 | 59,835 | 47 | SNAP `CollegeMsg.txt.gz` |
| Bitcoinotc | 6,005 | 35,592 | 62 | SNAP `soc-sign-bitcoin-otc.csv.gz` |
| EUT (Email-EU-temporal) | 1,005 | 332,334 | 127 | SNAP `email-Eu-core-temporal.txt.gz` |
| Mooc-actions | 7,047 | 411,749 | 72 | SNAP `mooc.zip` |
| LastFM | 1,000 | 1,293,103 | 76 | SNAP `lastfm_song.txt.gz` |
| Wikipedia | 5,684 | 87,931 | 42 | konect / jodie dump |

`scripts/download_datasets.py` downloads with retry and caches Parquet under `data/raw/<name>/`.

### 6.2 Snapshot construction

Equal-time-window binning over `[t_min, t_max]` into T bins. Each bin → `torch_geometric.data.Data` with `edge_index`. List of `Data` plus `num_nodes` and `node_features` becomes a `DynamicGraph` dataclass.

### 6.3 NRNAE preprocessing (paper's main contribution)

For each snapshot t:
- **Clustering coefficient:** `CC(i) = 2·R(i) / (K(i)·(K(i)-1))` via `networkx.clustering`.
- **Aggregation strength:** `AS(i) = degree(i) · CC(i)`.
- **Pairwise aggregation:** `S(i,j) = |N(i) ∩ N(j)| · AS(i)`, stored as sparse `[N×N]`.
- **Enhanced adjacency:** `Ŝ_t = A_t + β·S_t + I_N`.

Cached to `data/processed/<dataset>/S_t_{0..T-1}.pt`. Cache key = hash of `(raw_md5, num_time_steps, preprocess_version)`.

### 6.4 Train / val / test split (temporal)

- **Train:** snapshots `[0, …, ⌊0.8T⌋ - 1]`.
- **Val:** snapshot `⌊0.8T⌋`.
- **Test:** snapshots `[⌊0.8T⌋ + 1, …, T - 1]`.

Task at each evaluated step t: given inputs `[0..t]`, predict adjacency at `t+1`.

### 6.5 Negative sampling

Paper does not specify. We adopt:
- **Strategy:** uniform random rejection sampling. Reject if `(u,v) ∈ E_{t+1}` or `u == v`.
- **Ratio:** 1:1 with positives.
- **Shared across all models** for fairness.
- Documented in `reproduction-log.md`.

### 6.6 Node features into the model

- **GCN_MA input X^t:** `[degree, CC, AS]` per node (3 dims), **recomputed per snapshot t** since the underlying graph evolves. Computed alongside NRNAE preprocessing in §6.3 and cached together. Justification: paper emphasizes these as "rich features"; NRNAE is the contribution.
- **Baselines input X:** one-hot identity `I_N` (static across snapshots). Standard when features absent.

### 6.7 Reproducibility

- Seeds: `{42, 123, 2024}` for 3-run mean ± std.
- Test pairs (positive + negative) fixed once per dataset with `seed_test=999`, cached to `data/processed/<name>/test_pairs.pt`. All models evaluated on identical pairs.
- `torch.use_deterministic_algorithms(True)` when feasible; otherwise documented.

## 7. Model layer

### 7.1 Common interface (`src/models/base.py`)

```python
class DynamicLinkPredictor(nn.Module, ABC):
    @abstractmethod
    def forward(self, snapshots: list[Data], time_step: int) -> Tensor:
        """Returns Z^t ∈ R^{N×D}."""

    def predict_links(self, Z: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Default: MLP([Z[src] ⊕ Z[dst]]) → logit. Models may override."""
```

### 7.2 GCN_MA modules

- **GCN layer** — spectral form `H^t = ReLU(D̂^(-1/2) · Ŝ^t · D̂^(-1/2) · X^t · W^t)`, where `X^t` are the per-snapshot features from §6.6 and `W^t` is the LSTM-evolved weight matrix. Written by hand with `MessagePassing` (~30 LOC) for clarity in thesis writeup.
- **LSTM weight updater** — `W^t = LSTMCell(flatten(W^{t-1}), hidden^{t-1})`. Xavier init for `W^0`.
- **Multi-head self-attention** — `Z^t = MultiHeadAttn(H^t, H^t, H^t)` with `num_heads = 8`, residual + LayerNorm.
- **Link decoder** — `P^t[i,j] = σ(MLP_2-layer([Z^t[i] ⊕ Z^t[j]]))`. ReLU + sigmoid.
- **Composition** — at step t: scan `τ = 0..t` building `H_τ` and updating `W`; then attention on `H_t` → `Z_t`.

### 7.3 Hyperparameters (defaults inferred where paper is silent)

| Param | Value | Source |
|---|---|---|
| `hidden_dim` | 128 | EvolveGCN convention, fits 12GB VRAM |
| `num_heads` | 8 | Transformer convention |
| `β` | grid `{0.7, 0.8, 0.9}` on val | Paper recommends `[0.7, 0.9]` |
| `lr` | 1e-3 | Adam default for GCNs |
| `optimizer` | Adam | Standard |
| `weight_decay` | 1e-5 | Light regularization |
| `epochs` | 200, early stop patience 20 on val AUC | Standard |
| `dropout` | 0.1 (attention, MLP) | Standard |
| `grad_clip` | max_norm = 5.0 | Anti-explode for LSTM |
| `loss` | BCEWithLogitsLoss | Paper equation |

All choices logged in `reproduction-log.md`.

### 7.4 Baselines (fork strategy)

| Baseline | Upstream | Adapter effort | Fallback |
|---|---|---|---|
| EvolveGCN | `IBM/EvolveGCN` | Medium (likely PyTorch-1 API patches) | Reimplement EvolveGCN-O if needed |
| HTGN | `marlin-codes/HTGN` | High (hyperbolic ops, custom loader) | If fail >1 day → reimplement |
| DyGNN | `alge24/DyGNN` | Medium-high (edge-sequence paradigm) | Reimplement |
| DGCN | TBD (no canonical repo) | High → likely reimplement from Manessi et al. 2020 | Native impl in `src/models/dgcn.py` |

`third_party/` are git submodules — not modified. Adapter in `src/models/<name>.py` translates between `DynamicGraph` and the upstream input format and conforms to the `DynamicLinkPredictor` interface.

## 8. Training

### 8.1 Loop (`src/training/trainer.py`)

For each epoch, iterate `t` over training snapshots: compute `Z_t`, sample `|E_{t+1}|` negatives, build pos/neg logits, BCE loss, backward, Adam step. Val AUC at end of epoch drives `ReduceLROnPlateau` and early stop.

### 8.2 Backprop length

- Default: full backprop through time.
- Fallback if CUDA OOM (likely on EUT with T=127): **TBPTT length=10** (detach state every 10 snapshots). Documented per dataset if used.

### 8.3 Robustness

| Failure | Handling |
|---|---|
| CUDA OOM | Retry with `hidden_dim // 2`. If still OOM, log fail and skip. |
| NaN loss | Detect via `torch.isnan(loss)`; dump grad norms; abort run. |
| Dataset file missing | 3 retries in downloader; clear error if persistent. |
| Baseline import error | Try/except around adapter import; log "baseline X unavailable"; continue. |
| Process kill | Checkpoint every 10 epochs; `--resume <ckpt>` flag. |

### 8.4 Artifacts

- `results/checkpoints/<model>_<dataset>_<seed>_best.pt` (only best, intermediates deleted).
- `results/metrics.jsonl` — one line per run with `{date, model, dataset, seed, auc, ap, epoch_best, runtime_s, config_hash, git_sha}`.
- `results/logs/<model>_<dataset>_<seed>_<ts>.log` — full text log.
- `results/tb/<run>/` — TensorBoard (optional).

### 8.5 Bulk execution

`scripts/run_all.sh` iterates 5 models × 6 datasets × 3 seeds = **90 experiments**. Sequential, single GPU, estimated ~50 hours wall-clock. Failures appended to `failed.log` and re-runnable individually.

## 9. Evaluation

### 9.1 Metrics

- **AUC:** `sklearn.metrics.roc_auc_score(y_true, y_score)`.
- **AP:** `sklearn.metrics.average_precision_score(y_true, y_score)`.

### 9.2 Test pair construction

For each test step `t+1`:
- Positives = all edges in `E_{t+1}`.
- Negatives = `|E_{t+1}|` sampled with rejection (no positives, no self-loops). Edges that appeared in earlier snapshots are **not** filtered out — model must handle recurrence.

Fixed via `seed_test=999`, cached, identical across models.

### 9.3 Evaluator

Pooled across all test snapshots: collect `(y_score, y_true)` from every test step, then one AUC and one AP over the pooled arrays. Mirrors paper convention.

### 9.4 Hyperparameter tuning

- **GCN_MA:** grid `β ∈ {0.7, 0.8, 0.9}` × `hidden_dim ∈ {64, 128}` on Bitcoinotc only with seed 42. Best val AUC config applied to all datasets.
- **Baselines:** authors' defaults. Not tuned. ("As reported.")
- Limitation noted in thesis.

### 9.5 Aggregation & report assets

`scripts/aggregate_results.py` reads `metrics.jsonl` and produces:
- Markdown comparison table (paste into report).
- LaTeX table `results/report/table2_repro.tex`.
- Plots PDF: `auc_comparison.pdf`, `loss_curves_<dataset>.pdf`, `beta_sensitivity.pdf` (reproduces paper's β figure).
- `results/report/SUMMARY.md` — auto-summary "GCN_MA wins/loses vs paper, mean deviation X%".

## 10. Testing

Three tiers:

- **Tier 1 (unit, <30s):** `test_nrnae.py` (CC/AS/S correctness on 5-node graph), `test_metrics.py`, `test_negative_sampling.py`, `test_data_split.py`.
- **Tier 2 (smoke, <2min, CPU):** `test_models_smoke.py` — each model forwards on 50-node fake graph, output shape OK, backward without NaN.
- **Tier 3 (integration, 5–10min, manual):** `test_full_pipeline.py` — GCN_MA on CollegeMsg with `epochs=2`, verify `metrics.jsonl` append + checkpoint roundtrip.

Coverage target: ≥60% for `src/data/`, `src/eval/`. Models covered by smoke + integration.

Linting via `ruff`. Type hints on dataclasses and public APIs.

## 11. Dependency management

- `uv` for env + install.
- `pyproject.toml` pins `torch==2.4.0`, `torch-geometric==2.6.1`, plus `networkx`, `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `tqdm`, `matplotlib`, `seaborn`, `tensorboard` (optional). Dev extras: `pytest`, `ruff`, `mypy`.
- Setup:
  ```bash
  uv venv --python 3.11
  uv pip install -e ".[dev]"
  uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  ```
- Baseline env conflicts: attempt API patches in adapter first; subprocess+sidecar env only as last resort.

## 12. Timeline (8 weeks + 1 buffer)

| Week | Goal | Output |
|---|---|---|
| 1 | Repo, env, dataset download + parse, `DynamicGraph` + smoke test | All datasets loadable, stats plotted |
| 2 | NRNAE preprocessing + cache + tests | `S_t` cached for 6 datasets; tests pass |
| 3 | GCN_MA modules + smoke | Forward pass OK on 1 dataset |
| 4 | Trainer, evaluator, logging, checkpoint; first GCN_MA run on CollegeMsg | First AUC/AP, debug gap vs paper |
| 5 | Adapt EvolveGCN + DGCN (simpler baselines) | 2 baselines runnable, preliminary comparison table |
| 6 | Adapt HTGN + DyGNN | All 5 models runnable on ≥3 datasets |
| 7 | Full 5×6×3 = 90 runs (~50h background); debug failures | `metrics.jsonl` complete |
| 8 | Aggregate, plots, write thesis, polish | Tables + plots ready |
| 9 (buffer) | Final report fixes | — |

## 13. Git workflow

- `git init` immediately (done as part of repo setup).
- Branch: `main`.
- Commit prefixes: `[data]`, `[model]`, `[train]`, `[eval]`, `[docs]`, `[fix]`.
- `.gitignore`: `.venv/`, `data/raw/`, `data/processed/`, `results/`, `__pycache__/`, `.idea/`, `*.pt`, `wandb/`.
- Tags: `v0.1-data-ready`, `v0.2-gcn-ma-trained`, `v0.3-baselines-integrated`, `v1.0-thesis-ready`.
- Spec committed before implementation begins (gate to `writing-plans`).

## 14. Documentation

- `docs/paper-notes.md` — verbatim equations, section refs.
- `docs/reproduction-log.md` — every deviation from paper with rationale (critical for thesis defense).
- `docs/architecture.md` — diagrams.
- `docs/results.md` — final tables + plots + discussion.

## 15. Deliverables at thesis submission

- Git repo with runnable code and README.
- `results/metrics.jsonl` — raw per-run results.
- `results/report/` — tables (Markdown + LaTeX) and plots (PDF).
- `docs/reproduction-log.md` — defensible record of every non-paper choice.
- Thesis report (LaTeX / Word) using `results/report/` assets.
- Defense slides with auto-generated plots.

## 16. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Fork baseline incompatible with PyTorch 2.x | Try adapter API patches first; isolate env only if >1 day lost |
| No canonical DGCN repo | Reimplement from Manessi et al. 2020 in `src/models/dgcn.py` |
| LSTM W^t blows up at `F·D = 16k` | Fall back to GRU; log in reproduction log |
| EUT T=127 OOM full backprop | TBPTT length 10 |
| 90 experiments × time | Run in background overnight; partial-fail tolerant |
| Hyperparam tune only on Bitcoinotc | Disclosed as limitation in thesis |

## 17. Phase 2 (out of scope for this spec)

After Phase 1 completes, a separate brainstorm will design improvements: ablations (drop NRNAE / drop LSTM / drop attention), alternative attention mechanisms, alternative weight evolution (GRU, mamba), additional datasets, etc.

## 18. Open questions / pending

- DGCN upstream repo discovery (week 5 task).
- Confirm SNAP / konect URLs are still live at week 1 start.
- LastFM raw size on disk after parse (may exceed expected; budget disk).
