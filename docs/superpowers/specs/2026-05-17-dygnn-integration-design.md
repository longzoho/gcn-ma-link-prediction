# Design Spec — DyGNN Baseline Integration (Plan 3c)

**Date:** 2026-05-17
**Author:** long.huynh@siliconprime.com
**Status:** DRAFT (awaiting user review)
**Project:** `gcn-ma-link-prediction`
**Purpose:** Thesis / coursework reproduction — Plan 3c of 4 (baselines integration).

**Predecessor work:**
- Plan 1 (`v0.1-foundation`): GCN_MA on CollegeMsg.
- Plan 2 (`v0.2-gcn-ma-full`): GCN_MA × 6 datasets × 3 seeds.
- Plan 3a (`v0.3a-evolvegcn-o`): EvolveGCN-O integrated, symmetrize fix.
- Plan 3b (`v0.3b-htgn`): HTGN integrated, dominates all 6 datasets.

**Successor plans:**
- Plan 3d: DGCN reimplementation.
- Plan 4: Full aggregation, plots, thesis assets.

---

## 1. Goals & non-goals

### Goals (Plan 3c only)

1. Integrate DyGNN (Streaming Graph Neural Network, Ma et al., SIGIR 2020) as the 4th baseline.
2. Path A primary: vendor `alge24/DyGNN` as git submodule, write adapter (~200 LOC) wrapping upstream edge-update class. Path B fallback: reimplement DyGNN core (~250 LOC across 4 modules).
3. Apply Hybrid hyperparameter policy.
4. Run **15 experiments** (5 datasets × 3 seeds — **NOT LastFM**, see §3).
5. Produce 4-model cross-comparison table.

### Non-goals (Plan 3c)

- DGCN (Plan 3d).
- LastFM with DyGNN (skipped due to compute budget — 1.29M edges incompatible with edge-sequence model under our trainer).
- DyGNN's hierarchical attention variant (paper §5.3) — only base DyGNN.
- Per-dataset hyperparameter tuning.

## 2. Reference

- **Baseline paper:** Yao Ma, Ziyi Guo, Zhaocun Ren, Eric Zhao, Jiliang Tang, Dawei Yin. *Streaming Graph Neural Networks*. SIGIR 2020. ([paper](https://arxiv.org/abs/1810.10627))
- **Reference implementation:** [`alge24/DyGNN`](https://github.com/alge24/DyGNN) — pin a specific commit at integration time.
- **Paper Mei & Zhao 2024 Table 2 (DyGNN column):** to be verified against published paper PDF in Plan 4.

## 3. Constraints

- **Stack:** Python 3.11, PyTorch 2.4, PyG 2.6.
- **Hardware:** NVIDIA RTX 3060 12GB local.
- **Compute budget:** epochs ≤ 200, early stop patience 20 on val AUC.
- **LastFM skipped:** 1.29M edges × edge-sequence processing × 200 epochs estimated at 50+ hours per seed. Skip and document.
- **No modifications** to `third_party/DyGNN/` — adaptations in `src/models/dygnn.py` (path A) or `src/models/dygnn/*.py` (path B).
- **Test pairs identical** to prior plans — deterministic via seed=999.

## 4. Approach: A → B fallback

### Path A (preferred)

1. Vendor `alge24/DyGNN` as git submodule under `third_party/DyGNN/`, pinned commit.
2. Smoke-test imports under PyTorch 2.4. Inspect API. Time budget: 1 day max.
3. Write `DyGNN` adapter wrapping upstream edge-update class. Adapter:
   - Reconstructs edge sequence from snapshots using cached original timestamps.
   - Processes edges chronologically once per epoch, caching per-snapshot memory states.
   - Returns cached `memory_at_time[t]` for each `forward(snapshots, t)` call.
   - Feeds shared MLP decoder.
4. If shims exceed ~10 lines OR upstream is task-coupled (DyGNN repos are notoriously fragile) → BLOCKED → switch to path B.

### Path B (fallback)

If A fails:

1. Keep submodule for citation; don't import upstream code.
2. Reimplement DyGNN core in 4 focused modules (~250 LOC total):
   - `node_memory.py` — per-node memory state matrix (~40 LOC).
   - `edge_update.py` — coupled GRU for (u, v) memory update from paper Eq. 4-7 (~80 LOC).
   - `interaction.py` — interaction unit between u and v (~50 LOC).
   - `model.py` — composition + per-epoch cache logic (~80 LOC).

Both paths feed downstream pipeline identically.

## 5. Critical design decision: per-epoch memory cache

DyGNN is an edge-sequence model with per-node memory. Under our shared snapshot trainer:

- Naive approach: each `forward(t)` re-processes all edges [0..t]. Total O(T²·E) per epoch. For EUT (T=127, ~330k edges) this is **23 hours per epoch** — catastrophic.

- **Per-epoch cache (adopted):** at first `forward()` call per epoch, do single chronological pass through all edges, cache `memory_at_time[t]` for t=0..T-1. Subsequent `forward(t)` returns cached state. Total O(T·E) per epoch.

### Gradient approximation

Cache stores memory snapshots. Gradient through `cache[t]` flows back to:
- DyGNN encoder parameters (GRU weights, interaction unit) — via the chain of memory updates that built cache[t]
- Initial `self.memory` Parameter — copied at cache build time (detached fresh copy)

Within ONE epoch, all `forward(t)` calls share the SAME cache (built at start of epoch with epoch-start parameters). After backprop, optimizer.step() updates parameters but cache is not rebuilt until next epoch.

**Implication:** gradient signal across the inner training loop is "stale" — biased gradient. The model still learns, just sub-optimally. This is consistent with how DyGNN paper trains (sequential edge processing, accumulated loss, less-frequent optimizer steps).

### Epoch detection

Adapter uses heuristic: track `time_step` argument; reset cache when `current_t < previous_max_t` (regression indicates new epoch). Trainer's `for t in range(val_step)` is monotonic per epoch, so detection is robust.

## 6. Architecture & file map

### Path A (preferred)

```
gcn-ma-link-prediction/
├── third_party/DyGNN/                             # NEW submodule
├── configs/
│   ├── models/dygnn.yaml                          # NEW
│   ├── models/dygnn_smoke.yaml                    # NEW
│   └── experiments/
│       ├── dygnn_collegemsg.yaml                  # NEW
│       ├── dygnn_collegemsg_smoke.yaml            # NEW
│       ├── dygnn_bitcoinotc.yaml                  # NEW
│       ├── dygnn_eut.yaml                         # NEW
│       ├── dygnn_mooc_actions.yaml                # NEW
│       ├── dygnn_wikipedia.yaml                   # NEW
│       # NO dygnn_lastfm.yaml — skipped per §3
├── src/
│   ├── data/loaders/_base.py                      # MODIFY — fmt3 cache (adds edge_ts)
│   └── models/dygnn.py                            # NEW — adapter (~200 LOC)
├── scripts/train.py                               # MODIFY — add dygnn branch
├── tests/test_dygnn_smoke.py                      # NEW (~6 tests)
└── docs/reproduction-log.md                       # MODIFY — Plan 3c section
```

### Path B (fallback)

```
src/models/dygnn/                                  # replaces dygnn.py
├── __init__.py
├── node_memory.py                                 # NEW (~40 LOC)
├── edge_update.py                                 # NEW (~80 LOC)
├── interaction.py                                 # NEW (~50 LOC)
└── model.py                                       # NEW (~80 LOC)
```

## 7. Loader fmt3 cache migration

Bumping `_CACHE_FORMAT_VERSION` from `"fmt2"` to `"fmt3"` invalidates all existing caches. Preprocessing cost per dataset (Plan 2 timings): ~2.5 min total for 5 datasets. One-time acceptable cost.

### Schema change

Before (fmt2):
```python
{"features": [...], "edge_index": [...], "num_nodes": N, "num_time_steps": T}
```

After (fmt3):
```python
{"features": [...], "edge_index": [...], "edge_ts": [...], "num_nodes": N, "num_time_steps": T}
```

`edge_ts[t]` is a `[E_t]` tensor (float64) of original timestamps for edges in snapshot t. Used by DyGNN adapter to sort edges chronologically within a snapshot.

### Loader modification

In `_preprocess()`, add timestamp extraction:

```python
for t in range(num_time_steps):
    mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
    sub = df.loc[mask, ["src", "dst", "ts"]].values  # add "ts"
    edges_list = [(int(u), int(v), float(ts)) for u, v, ts in sub if u != v]
    # ... existing feature/A computation ...
    if edges_list:
        ts_array = torch.tensor([ts for u, v, ts in edges_list], dtype=torch.float64)
    else:
        ts_array = torch.empty(0, dtype=torch.float64)
    edge_index_list.append(edge_index)
    edge_ts_list.append(ts_array)
```

`build()` method exposes `edge_ts` via `snap.edge_ts = cached["edge_ts"][t]` (custom attribute on PyG `Data`).

### Backward compatibility

Other models (GCN_MA, EvolveGCN, HTGN) don't use `snap.edge_ts` and ignore the new attribute. No regression risk.

## 8. Hyperparameter policy: Hybrid

`configs/models/dygnn.yaml`:

```yaml
name: dygnn
hidden_dim: 64                  # shared with all other models
node_memory_dim: 64             # = hidden_dim
edge_dim: 16                    # DyGNN paper default
dropout: 0.1                    # shared
lr: 1.0e-3                      # shared
weight_decay: 1.0e-5            # shared
optimizer: adam                 # shared (paper used Adam too)
epochs: 200                     # shared
early_stop_patience: 20         # shared
grad_clip_max_norm: 5.0         # shared
# DyGNN-specific
decay_method: log               # time decay function: w(Δt) = (Δt + 1)^-c
decay_rate: 1.0
```

### Deviations from DyGNN paper

1. **Per-epoch memory cache** (gradient approximation) — paper does per-edge gradient updates; we cache per-epoch for compute. Documented in §5.
2. **Symmetric edge processing** — each (u, v) edge processed twice (u→v then v→u with ε offset) to give item-side signal flow, mirroring Plan 3a/3b symmetrize approach.
3. **Shared MLP decoder** — paper has its own scoring head; we use shared `LinkDecoderMLP` for cross-model fairness.
4. **LastFM skipped** — compute budget.

## 9. Adapter design

### 9.1 Per-epoch cache (path A pseudocode)

```python
class DyGNN(DynamicLinkPredictor):
    def __init__(self, num_nodes, hidden_dim, node_memory_dim, edge_dim,
                 dropout, decay_method="log", decay_rate=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Initial node memory (learnable, init to zero)
        self.memory_init = nn.Parameter(torch.zeros(num_nodes, node_memory_dim))

        # Upstream DyGNN update module
        self.update_module = UpstreamDyGNNUpdate(
            node_dim=node_memory_dim,
            edge_dim=edge_dim,
            decay_method=decay_method,
            decay_rate=decay_rate,
        )

        # Shared decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Per-epoch cache state
        self._cached_memory_per_t: list[torch.Tensor] | None = None
        self._prev_max_t: int = -1

    def forward(self, snapshots, time_step):
        # New-epoch detection
        if time_step < self._prev_max_t or self._cached_memory_per_t is None:
            self._build_cache(snapshots)
        self._prev_max_t = max(self._prev_max_t, time_step)
        return self._cached_memory_per_t[time_step]

    def _build_cache(self, snapshots):
        device = self.memory_init.device
        N = self.num_nodes
        memory = self.memory_init.clone()  # starts gradient chain at memory_init
        cache = []
        for t in range(len(snapshots)):
            ei = snapshots[t].edge_index.to(device)
            ts = snapshots[t].edge_ts.to(device) if hasattr(snapshots[t], "edge_ts") else None
            if ei.numel() > 0 and ts is not None:
                # Sort edges chronologically within snapshot
                sorted_idx = torch.argsort(ts)
                ei_sorted = ei[:, sorted_idx]
                ts_sorted = ts[sorted_idx]
                # Process each edge (with symmetric u→v then v→u)
                for k in range(ei_sorted.shape[1]):
                    u, v = int(ei_sorted[0, k]), int(ei_sorted[1, k])
                    dt = ts_sorted[k]
                    memory = self.update_module(memory, u, v, dt)
                    memory = self.update_module(memory, v, u, dt + 1e-6)
            cache.append(memory.clone())
        self._cached_memory_per_t = cache
        self._prev_max_t = -1

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

### 9.2 Memory update function (path A)

Upstream's `UpstreamDyGNNUpdate` likely has signature like:
```python
update_module(memory: Tensor[N, D], src: int, dst: int, dt: float) -> Tensor[N, D]
```

Where the call modifies the `src` and `dst` rows of memory using coupled GRU + time decay. Task 2 inspects actual API.

### 9.3 Path B reimplementation (if path A fails)

Reimplement DyGNN paper §3 minimal:

- `node_memory.py`: `class NodeMemory(nn.Module)` — wraps a `nn.Parameter` matrix, exposes `get(idx)`, `set(idx, val)`.
- `edge_update.py`: `class CoupledGRUUpdate(nn.Module)` — paper Eq. 4-7:
  - For source: `s_u = GRU_S(s_u, h_v, Δt)` (source updated)
  - For target: `s_v = GRU_T(s_v, h_u, Δt)` (target updated)
  - Time decay: `decay = (Δt + 1)^(-c)` applied multiplicatively
- `interaction.py`: `class InteractionUnit(nn.Module)` — propagates effect from updated u/v to their neighborhoods (paper Eq. 8-9). Simplified for our use case.
- `model.py`: composition + cache logic identical to Section 9.1.

## 10. Testing

**Tier 1 — Unit tests (CPU, <30s):**
- `test_dygnn_memory_init_zeros` — memory_init starts at 0
- `test_dygnn_edge_update_shape` — single edge update preserves [N, D]
- `test_dygnn_chronological_sort_within_snapshot` — adapter sorts edges by recovered timestamps
- `test_loader_base_cache_fmt3_includes_edge_ts` — cached payload has `edge_ts` key

**Tier 2 — Smoke (CPU, <2 min):**
- `test_can_import_upstream_dygnn` (path A only)
- `test_can_construct_dygnn`
- `test_dygnn_forward_shape` — N=50, T=5, returns [N, hidden_dim]
- `test_dygnn_gradient_flows`
- `test_dygnn_cache_reuses_within_epoch`
- `test_dygnn_cache_rebuilds_on_t_regression`

**Tier 3 — Integration (~2 min):**
- `test_dygnn_runs_on_collegemsg_smoke` — epochs=3, real data, val_auc ∈ (0.5, 1.0).

**Tier 4 — Full (manual, ~3.5 hours):**
- 15 runs via `scripts/run_seeds.sh <dataset> dygnn` for 5 datasets.

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Upstream `alge24/DyGNN` incompatible với PyTorch 2.4 | High | 1-day timebox Task 2; >10 shims → path B. |
| Upstream code is task-coupled | High | DyGNN repos notoriously fragile. Path B likely. |
| Per-epoch cache produces NaN | Medium | Clip memory norms; ReLU on update output. |
| Gradient approximation degrades convergence | Medium | If val_auc never escapes 0.6, escalate. |
| EUT or Mooc exceeds 30 min/seed | Medium | Reduce node_memory_dim to 32; document. |
| Cache fmt3 invalidation accidentally breaks other models | Low | Other models ignore `edge_ts` attribute. |

## 12. Spec compliance

- §7.4 Fork upstream as submodule: ✅ path A.
- §6.6 Baselines use one-hot identity: ⚠️ DyGNN uses learnable per-node memory (natural fit, not a deviation per se).
- §9.4 Authors' defaults: ⚠️ Hybrid policy + documented gradient approximation.
- §6.4 Train/val/test split: ✅ inherited.
- §6.7 Reproducibility: ✅ same 3 seeds; same test pairs (fixed seed=999).

## 13. Deliverables

**Path A success:**
- `src/models/dygnn.py` (~200 LOC)
- `third_party/DyGNN/` submodule pinned
- `src/data/loaders/_base.py` modified (fmt3 schema)
- `configs/models/dygnn.yaml` + 5 experiment configs
- ~10 unit/smoke tests
- 15 DyGNN rows in `results/metrics.jsonl`
- `results/report/baselines_summary.md` with 4-model table
- Plan 3c section in reproduction-log
- Tag `v0.3c-dygnn`

**Path B success:**
- `src/models/dygnn/` directory (~250 LOC across 4 modules)
- Otherwise same deliverables

## 14. Timeline

| Phase | Path A | Path B |
|---|---|---|
| Submodule + smoke import | 30 min - 4 hours | (skipped if go straight to B) |
| Loader fmt3 schema | 1 hour | 1 hour |
| Adapter implementation | 4-8 hours | — |
| 4-module reimplement | — | 8-12 hours |
| Configs + dispatch | 30 min | 30 min |
| Smoke train on CollegeMsg | 1 hour | 1 hour |
| 15-run experiment | ~3.5 hours | ~3.5 hours |
| Docs + tag | 30 min | 30 min |
| **Total** | **~1-2 days** | **~2 days** |

## 15. Open questions / pending

- Exact upstream class name and `forward` signature of `alge24/DyGNN`. Inspect during Task 2.
- Pin commit SHA — use latest on default branch.
- Whether path A's shims fit under 10 lines.
- How DyGNN handles isolated nodes (no edges in a snapshot). Defensive coding in adapter: skip empty snapshots gracefully.
