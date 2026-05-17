# Design Spec — HTGN Baseline Integration (Plan 3b)

**Date:** 2026-05-17
**Author:** long.huynh@siliconprime.com
**Status:** DRAFT (awaiting user review)
**Project:** `gcn-ma-link-prediction`
**Purpose:** Thesis / coursework reproduction — Plan 3b of 4 (baselines integration).

**Predecessor work:**
- Plan 1 (`v0.1-foundation`): GCN_MA end-to-end on CollegeMsg.
- Plan 2 (`v0.2-gcn-ma-full`): GCN_MA scaled to 6 datasets × 3 seeds.
- Plan 3a (`v0.3a-evolvegcn-o`): EvolveGCN-O integrated. Symmetrize fix unlocked all 6 datasets.

**Successor plans:**
- Plan 3c: DyGNN integration.
- Plan 3d: DGCN reimplementation.
- Plan 4: Full aggregation, plots, thesis assets.

---

## 1. Goals & non-goals

### Goals (Plan 3b only)

1. Integrate HTGN (Hyperbolic Temporal Graph Network, Yang et al., CIKM 2021) as a baseline.
2. Path A primary: vendor `marlin-codes/HTGN` as git submodule, write thin adapter (~150-200 LOC).
3. Path B fallback: if path A fails within a 1-day timebox, reimplement HTGN core (~300 LOC across `poincare.py`, `hgcn_layer.py`, `temporal_gru.py`, `model.py`).
4. Hybrid hyperparameter policy: shared compute and capacity with GCN_MA/EvolveGCN-O; baseline-specific architecture choices from HTGN paper.
5. Run 18 experiments (6 datasets × 3 seeds {42, 123, 2024}).
6. Produce 3-model cross-comparison table (GCN_MA vs EvolveGCN-O vs HTGN).

### Non-goals (Plan 3b)

- Other 2 baselines (DyGNN, DGCN) — Plans 3c/3d.
- HTGN-H, HVGNN, or other hyperbolic variants.
- Riemannian optimizers (RAdam, RSGD) — fixed Adam per Hybrid policy.
- Learnable curvature — fixed at 1.0 (paper deviation documented).
- Per-dataset hyperparameter tuning.
- Direct comparison with HTGN paper's reported numbers (paper uses different datasets — Reddit, BlogCatalog).

## 2. Reference

- **Baseline paper:** Menglin Yang et al., *Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning in Hyperbolic Space*, CIKM 2021. ([arxiv](https://arxiv.org/abs/2107.03767))
- **Reference implementation:** [`marlin-codes/HTGN`](https://github.com/marlin-codes/HTGN) — pin a specific commit during Task 1.
- **Paper Mei & Zhao 2024:** Does not include HTGN in their Table 2 — Plan 3b compares against GCN_MA and EvolveGCN-O directly, plus carries forward Paper EvolveGCN reference columns from Plan 3a.

## 3. Constraints

- **Stack:** Python 3.11, PyTorch 2.4, PyG 2.6.
- **Hardware:** NVIDIA RTX 3060 12GB local.
- **Compute budget per run:** epochs ≤ 200, early stop patience 20 on val AUC.
- **No modifications** to `third_party/HTGN/` — adaptations live in `src/models/htgn.py` (path A) or `src/models/htgn/*.py` (path B).
- **Test pairs identical** to Plans 2/3a — deterministic via `seed=999` neg sampling on the same `DynamicGraph`.

## 4. Approach: A → B fallback

### Path A (preferred)

1. Add `marlin-codes/HTGN` as git submodule under `third_party/HTGN/`, pinned to a specific commit.
2. Smoke-test imports under PyTorch 2.4. Inspect the upstream class API. Time budget: 1 day max.
3. Write `HTGN` adapter class extending `DynamicLinkPredictor`. Adapter wraps upstream encoder, applies log map projection at output, feeds shared MLP decoder.
4. If shims exceed ~10 lines OR upstream-API discovery takes >1 day → BLOCKED → switch to path B.

### Path B (fallback)

If A fails:

1. Keep `third_party/HTGN/` as submodule (for citation), but DO NOT use upstream code.
2. Reimplement HTGN core in `src/models/htgn/` as 4 focused modules:
   - `poincare.py` — log_map_0, exp_map_0, Möbius addition, hyperbolic distance (~80 LOC).
   - `hgcn_layer.py` — hyperbolic linear + Möbius aggregation (~50 LOC).
   - `temporal_gru.py` — recurrent weight evolution (~30 LOC, can reuse pattern from GCN_MA's `LSTMWeightUpdater`).
   - `model.py` — composition (~70 LOC).
3. Total: ~300 LOC, no geoopt dep.
4. Document deviation honestly in `reproduction-log.md`: list the specific upstream-incompatibility findings discovered in Task 2 (e.g. "ImportError on `torch._six`", "geoopt.manifolds.PoincareBall constructor signature changed", "training loop reads a 'tasker' object we cannot construct without their data pipeline") and quote upstream's actual error traceback.

Both paths feed downstream pipeline identically.

## 5. Hyperparameter policy: Hybrid

`configs/models/htgn.yaml`:

```yaml
name: htgn
feat_dim: 64                  # node embedding dim, matches hidden_dim
hidden_dim: 64                # shared with GCN_MA, EvolveGCN-O
num_layers: 2                 # HTGN paper default
curvature: 1.0                # Poincaré ball curvature (paper default)
trainable_curvature: false    # fix for stability; paper allows learnable
dropout: 0.1                  # shared
lr: 1.0e-3                    # shared (Adam, not RAdam)
weight_decay: 1.0e-5          # shared
optimizer: adam               # shared (Hybrid policy)
epochs: 200                   # shared
early_stop_patience: 20       # shared
grad_clip_max_norm: 5.0       # shared
```

### Deviations from HTGN paper

1. **Adam optimizer** instead of paper's RAdam. Rationale: Hybrid policy keeps optimizer consistent across all 3 models. The hyperbolic geometry survives in the encoder's intermediate computations and the output log-map projection; only the final-step parameter update is Euclidean instead of Riemannian.
2. **Fixed curvature** at 1.0 (paper recommends learnable). Rationale: with Adam (not RAdam), learnable curvature can wander outside numerically-valid range and produce NaN. Fixed curvature is a documented simplification.
3. **Shared MLP decoder** (`LinkDecoderMLP`) instead of paper's Fermi-Dirac hyperbolic decoder. Rationale: same as Plan 3a EvolveGCN-O — uniform decoder isolates encoder differences in cross-model comparison.
4. **Symmetric adjacency** (`A + A^T`) instead of paper's potentially-directed convention. Rationale: same fix as Plan 3a; bipartite datasets require symmetric A for valid signal propagation.

## 6. Architecture & file map

### Path A (preferred)

```
gcn-ma-link-prediction/
├── third_party/HTGN/                              # NEW git submodule, pinned commit
├── configs/
│   ├── models/htgn.yaml                           # NEW
│   └── experiments/
│       └── htgn_{collegemsg,bitcoinotc,eut,mooc_actions,lastfm,wikipedia}.yaml  # NEW × 6
├── src/models/htgn.py                             # NEW — adapter (~150-200 LOC)
├── scripts/train.py                               # MODIFY — add htgn branch to _build_model
├── tests/test_htgn_smoke.py                       # NEW (~5 tests)
└── docs/reproduction-log.md                       # MODIFY — append Plan 3b section
```

### Path B (fallback)

```
gcn-ma-link-prediction/
├── third_party/HTGN/                              # KEEP for citation, NOT used
├── src/models/htgn/                               # NEW directory
│   ├── __init__.py
│   ├── poincare.py                                # NEW — hyperbolic ops (~80 LOC)
│   ├── hgcn_layer.py                              # NEW — HGCN convolution (~50 LOC)
│   ├── temporal_gru.py                            # NEW — weight evolution (~30 LOC)
│   └── model.py                                   # NEW — HTGN composition (~70 LOC)
... rest same as path A
```

## 7. Adapter design

### 7.1 Hyperbolic log map at origin

Used by both paths to project Poincaré ball → Euclidean tangent space:

```python
def log_map_origin(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Poincaré ball → tangent space at origin.

    log_0(x) = (1/sqrt(c)) * arctanh(sqrt(c) * ||x||) * (x / ||x||)
    """
    sqrt_c = c ** 0.5
    max_norm = 1.0 / sqrt_c - eps
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps).clamp(max=max_norm)
    factor = torch.atanh(sqrt_c * norm) / (sqrt_c * norm)
    return factor * x
```

### 7.2 Symmetric adjacency (carry-forward from Plan 3a)

```python
ei_sym = torch.cat([ei, ei.flip(0)], dim=1)  # add reverse edges
A = torch.sparse_coo_tensor(ei_sym, vals, (N, N)).coalesce()
```

### 7.3 Path A adapter outline

```python
class HTGN(DynamicLinkPredictor):
    def __init__(self, num_nodes, feat_dim=64, hidden_dim=64, num_layers=2,
                 curvature=1.0, trainable_curvature=False, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.curvature = nn.Parameter(torch.tensor(curvature), requires_grad=trainable_curvature)

        # Learnable node embedding (shared deviation from spec §6.6)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Upstream HTGN encoder — specific construction details from Task 2 discovery
        self.core = UpstreamHTGN(...)  # filled in during adapter task

        # Shared decoder
        self.decoder = LinkDecoderMLP(embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, snapshots, time_step):
        # ... build A_list (symmetric) + X_list (constant node_emb)
        z_hyp = self.core(A_list, X_list)        # hyperbolic embeddings
        z_euc = log_map_origin(z_hyp, c=self.curvature.item())
        return z_euc

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

### 7.4 Path B reimplement outline

If A fails, the 4 modules:

- **poincare.py**: log_map_0, exp_map_0, mobius_add, hyperbolic_distance (~80 LOC).
- **hgcn_layer.py**: HGCNLayer — wraps Euclidean linear + Möbius aggregation (`x' = exp_0(sum_neighbors(log_0(x)))`) + activation (~50 LOC).
- **temporal_gru.py**: TemporalGRU — evolves weight matrix across snapshots (~30 LOC, mirror GCN_MA's `LSTMWeightUpdater` but GRU).
- **model.py**: HTGN — composes node_emb → 2 × HGCNLayer with temporal GRU between → log_map_0 → output (~70 LOC).

## 8. Training, evaluation, orchestration

### 8.1 Reused from prior plans (zero changes)

- `src/data/loaders/*` — 6 datasets, cached preprocessing.
- `src/training/trainer.py` — agnostic loop, Adam optimizer.
- `src/training/negative_sampling.py` — seed=999 for test pairs.
- `src/eval/evaluator.py`, `metrics.py`.
- `src/utils/seed.py`, `logging.py`.
- `scripts/run_seeds.sh` — already accepts `(dataset, model)` args from Plan 3a.
- `scripts/aggregate_results.py` — already supports `--models gcn_ma evolvegcn_o htgn` for 3-model table.

### 8.2 Modified in Plan 3b

- `scripts/train.py`: add `elif name == "htgn":` branch to `_build_model`.

### 8.3 Full execution

18 HTGN runs. Estimated wall-clock (based on Plan 3a EvolveGCN-O times, scaled by ~50% for hyperbolic ops):

| Dataset | EvolveGCN-O per-run | HTGN estimate | × 3 seeds |
|---|---|---|---|
| CollegeMsg | ~50-200s | ~80-300s | ~10 min |
| Bitcoinotc | ~85-145s | ~120-200s | ~10 min |
| EUT | ~1280-1772s | ~1800-2500s | ~110 min |
| Mooc | ~250-580s | ~400-900s | ~35 min |
| LastFM | ~650-905s | ~900-1300s | ~55 min |
| Wikipedia | ~110-180s | ~150-250s | ~10 min |
| **Total** | | | **~3.5-4 hours** |

If HTGN forward is significantly slower than estimated, EUT alone could exceed 4 hours. Plan handles partial-fail tolerance: each (dataset, seed) re-runnable individually.

## 9. Testing

**Tier 1 — Unit tests for hyperbolic ops (CPU, <30s):**
- `test_log_map_origin_inverse_of_exp_map` — log_map(exp_map(x)) ≈ x for small x.
- `test_log_map_origin_at_zero_returns_zero` — log_map_0(0) = 0.
- `test_log_map_origin_handles_near_boundary` — no NaN for ||x|| close to 1/√c.

**Tier 2 — Adapter smoke (CPU, <2 min):**
- `test_can_import_upstream_htgn` (path A only) — upstream class imports.
- `test_can_construct_htgn` — adapter __init__ works with realistic shapes.
- `test_htgn_forward_shape` — N=50, T=5, returns [N, hidden_dim] Euclidean.
- `test_htgn_gradient_flows` — backward produces finite gradients.

**Tier 3 — Integration (CPU/GPU, ~2 min):**
- `test_htgn_runs_on_collegemsg_smoke` — epochs=3, real CollegeMsg data, val_auc ∈ (0.5, 1.0). Escalate if stuck at exactly 0.5.

**Tier 4 — Full reproduction (~3.5-4 hours, manual):**
- 18 runs via `scripts/run_seeds.sh <dataset> htgn`.

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Upstream HTGN incompatible with PyTorch 2.4 | **High** | 1-day timebox; >10 lines shims → BLOCKED → path B. |
| geoopt manifold lib stale deps | Medium-High | Adapter implements `log_map_0` directly. Path B avoids geoopt entirely. |
| Hyperbolic ops NaN during training | Medium | Clamping in log_map (norm in [eps, 1/√c − eps]). Grad clipping at 5.0. |
| Curvature drift breaks numerics | Low | `trainable_curvature=False` fixed at 1.0. |
| Model can't learn (Z all zeros, like bipartite EvolveGCN bug) | Medium | Apply symmetric adjacency from start. Smoke-test on CollegeMsg before launching full 18-run. |
| HTGN forward 2-3× slower than expected | Medium | Plan timeline already pads. Partial-fail re-runnable. |
| Both paths A and B fail | Low | Document attempts. Skip Plan 3b. Move to Plan 3c (DyGNN). |

## 11. Spec compliance

- §7.4 Fork upstream as `third_party/`: ✅ path A. Path B keeps submodule for citation only.
- §6.6 Baselines use one-hot identity: ⚠️ Same deviation as Plan 3a (learnable embedding, RAM justified).
- §9.4 Baselines use authors' defaults: ⚠️ Hybrid policy with documented deviations (Adam not RAdam, fixed curvature, shared MLP decoder, symmetric adjacency).
- §6.4 Train/val/test split: ✅ inherited from trainer.
- §6.7 Reproducibility seeding: ✅ same 3 seeds, same test pairs.

## 12. Deliverables

**Path A success:**
- `src/models/htgn.py` (~150-200 LOC)
- `third_party/HTGN/` submodule pinned
- `configs/models/htgn.yaml` + 6 experiment configs
- 5+ smoke tests passing
- 18 HTGN rows in `results/metrics.jsonl`
- `results/report/baselines_summary.md` with 3-model table
- Plan 3b section in `docs/reproduction-log.md`
- Tag `v0.3b-htgn`

**Path B success:**
- `src/models/htgn/` directory (4 modules, ~300 LOC)
- `third_party/HTGN/` submodule for citation
- Same configs, tests, metrics, report, tag

**Both paths abandoned (very unlikely):**
- Document attempts in reproduction-log
- Skip Plan 3b, move to Plan 3c

## 13. Timeline

| Phase | Path A | Path B |
|---|---|---|
| Submodule setup + smoke import | 30 min - 4 hours | (already done) |
| Adapter implementation | 4-8 hours | — |
| Hyperbolic ops + HGCN + temporal GRU + composition | — | 8-12 hours |
| Config + dispatch refactor | 30 min | 30 min |
| Smoke test on CollegeMsg | 1 hour | 1 hour |
| 18-run full experiment | ~4 hours | ~4 hours |
| Docs + tag | 30 min | 30 min |
| **Total** | **~1-2 days** | **~2-3 days** |

## 14. Out of scope (deferred)

- DyGNN, DGCN — Plans 3c/3d.
- HTGN-H, HVGNN.
- RAdam / RSGD optimizers.
- Learnable curvature.
- Per-dataset hyperparameter tuning.
- Implementing HTGN's Fermi-Dirac decoder.

## 15. Open questions / pending

- Exact upstream class name and `forward` signature in `marlin-codes/HTGN`. Inspect during Task 2.
- Pin commit SHA of `marlin-codes/HTGN`. Use latest commit on default branch as of Task 1.
- Whether path A's shims fit under 10 lines. Decided at end of Task 2.
