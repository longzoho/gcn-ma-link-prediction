# Design Spec — EvolveGCN-O Baseline Integration (Plan 3a)

**Date:** 2026-05-16
**Author:** long.huynh@siliconprime.com
**Status:** DRAFT (awaiting user review)
**Project:** `gcn-ma-link-prediction`
**Purpose:** Thesis / coursework reproduction — Plan 3a of 4 (baselines integration).

**Predecessor work:**
- Plan 1 (`v0.1-foundation`): GCN_MA end-to-end on CollegeMsg.
- Plan 2 (`v0.2-gcn-ma-full`): GCN_MA scaled to 6 datasets × 3 seeds. β grid → β=0.8, hidden_dim=64.

**Successor plans:**
- Plan 3b: HTGN integration.
- Plan 3c: DyGNN integration.
- Plan 3d: DGCN reimplementation.
- Plan 4: Full aggregation, plots, thesis assets.

---

## 1. Goals & non-goals

### Goals (Plan 3a only)

1. Integrate EvolveGCN-O (the Output-only variant from Pareja et al., AAAI 2020) as a baseline via git submodule + thin adapter.
2. Apply the **Hybrid hyperparameter policy** (shared compute and capacity with GCN_MA; baseline-specific architecture choices follow the paper).
3. Run 18 experiments (6 datasets × 3 seeds {42, 123, 2024}).
4. Produce a side-by-side mean ± std comparison table vs GCN_MA and vs the GCN_MA paper's reported EvolveGCN numbers.

### Non-goals (Plan 3a)

- Other 3 baselines (HTGN, DyGNN, DGCN).
- EvolveGCN-H variant.
- Hyperparameter tuning beyond paper defaults for EvolveGCN-O-specific knobs.
- Re-running GCN_MA at different `hidden_dim` for Bitcoinotc/LastFM (separate concern).
- Modifying upstream IBM/EvolveGCN code — only adapter shims allowed.

## 2. Reference

- **Baseline paper:** Aldo Pareja et al., *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs*, AAAI 2020. EvolveGCN-O variant: `W^(t+1) = GRU(W^(t))` — GRU treats the weight matrix as its hidden state.
- **Reference implementation:** [`IBM/EvolveGCN`](https://github.com/IBM/EvolveGCN) (pin a specific commit at integration time; record SHA in reproduction log).
- **Paper Mei & Zhao 2024 Table 2 (EvolveGCN column):** numbers to be verified against the published paper PDF during Plan 4 aggregation. The implementer must extract the actual reported EvolveGCN AUC/AP per dataset (6 numbers each) directly from Table 2 — do not infer from memory.

## 3. Constraints

- **Stack:** Python 3.11, PyTorch 2.4, PyG 2.6 — same as Plan 2.
- **Hardware:** NVIDIA RTX 3060 12GB local.
- **Compute budget per run:** epochs ≤ 200, early stop patience 20 on val AUC.
- **No modifications** to `third_party/EvolveGCN/` — adaptations live in `src/models/evolvegcn.py`.
- **Test pairs identical to Plan 2 GCN_MA** — deterministic via `seed=999` neg sampling on the same `DynamicGraph`.

## 4. Approach (selected: A — git submodule + thin adapter)

1. Add IBM/EvolveGCN as a git submodule under `third_party/EvolveGCN/`, pinned to a specific commit.
2. Smoke-test imports under PyTorch 2.4. Apply only minimal compatibility shims in `src/models/evolvegcn.py` (max ~5 lines). If incompatibility is deeper than that, abandon Approach A and switch to a reimplementation (separate plan revision).
3. Write `EvolveGCN_O` adapter class extending `DynamicLinkPredictor`. Adapter translates between our `DynamicGraph` and upstream's expected input format.
4. Reuse Plan 1/2 trainer, evaluator, negative sampling, metrics, loaders, runner, aggregator (with generalization for multi-model).
5. Run 18 experiments. Append to existing `results/metrics.jsonl`.

Rejected: Approach B (full reimplement) — risk of subtle bug making baseline look weak. Rejected C (timeboxed hybrid) — adds conditional planning complexity for low expected savings.

## 5. Hyperparameter policy: Hybrid (option C from brainstorm)

`configs/models/evolvegcn_o.yaml`:

```yaml
name: evolvegcn_o
feat_dim: 64                  # node embedding dim, matches hidden_dim
hidden_dim: 64                # shared with GCN_MA
num_layers: 2                 # EvolveGCN-O paper default
activation: rrelu             # paper default
dropout: 0.1                  # shared with GCN_MA
lr: 1.0e-3                    # shared with GCN_MA
weight_decay: 1.0e-5          # shared
optimizer: adam               # shared
epochs: 200                   # shared compute budget
early_stop_patience: 20       # shared
grad_clip_max_norm: 5.0       # shared
```

Justification:
- **Shared compute + capacity (h=64, lr, epochs, early stop):** fair comparison vs GCN_MA at matched compute.
- **Baseline-specific (num_layers=2, RReLU):** follow EvolveGCN paper.
- **β does not apply** — EvolveGCN-O has no NRNAE mixing factor.

## 6. Architecture & file map

```
gcn-ma-link-prediction/
├── .gitmodules                                    # NEW
├── third_party/EvolveGCN/                         # NEW git submodule, pinned
├── configs/
│   ├── models/evolvegcn_o.yaml                    # NEW
│   └── experiments/
│       └── evolvegcn_o_{collegemsg,bitcoinotc,eut,mooc_actions,lastfm,wikipedia}.yaml  # NEW × 6
├── src/models/evolvegcn.py                        # NEW — adapter (~120 LOC)
├── scripts/
│   ├── train.py                                   # MODIFY — model dispatch helper
│   ├── run_seeds.sh                               # MODIFY — accept (dataset, model) args
│   └── aggregate_results.py                       # NEW — generalize from aggregate_gcn_ma_results.py
├── tests/test_evolvegcn_smoke.py                  # NEW — 4 tests
└── docs/reproduction-log.md                       # MODIFY — append Plan 3a section
```

## 7. Adapter design

### 7.1 Node features (deviation from spec §6.6)

Spec §6.6 originally specified one-hot identity `I_N` for baselines. RAM cost is prohibitive for Bitcoinotc, Mooc, and Wikipedia:

| Dataset | N | One-hot per snapshot | Total across T |
|---|---|---|---|
| Mooc | 7144 | 816 MB | ~59 GB |
| Wikipedia | 7474 | 893 MB | ~37 GB |
| Bitcoinotc | 5881 | 553 MB | ~34 GB |

**Replacement:** learnable node embedding `nn.Embedding(N, feat_dim)`, Xavier-initialized. Standard practice in IBM/EvolveGCN's own code when N is large. RAM cost is N × feat_dim × 4 bytes ≈ 1.9 MB for Wikipedia.

This deviation is documented in `docs/reproduction-log.md` Plan 3a section.

### 7.2 Adapter class outline

```python
class EvolveGCN_O(DynamicLinkPredictor):
    def __init__(self, num_nodes, feat_dim=64, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        # Node embedding replaces one-hot identity
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Upstream EGCN_O with constructed args namespace
        args = SimpleNamespace(
            num_layers=num_layers,
            feats_per_node=feat_dim,
            layer_1_feats=hidden_dim,
            layer_2_feats=hidden_dim,
        )
        self.core = UpstreamEGCN_O(args, activation=nn.RReLU(), device="cpu")

        # MLP decoder (shared pattern with GCN_MA)
        self.decoder = LinkDecoderMLP(embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, snapshots, time_step):
        device = self.node_emb.weight.device
        N = self.node_emb.num_embeddings
        node_ids = torch.arange(N, device=device)

        A_list, X_list, mask_list = [], [], []
        for tau in range(time_step + 1):
            ei = snapshots[tau].edge_index.to(device)
            vals = torch.ones(ei.shape[1], device=device)
            A = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce()
            A_list.append(A)
            X_list.append(self.node_emb(node_ids))
            mask_list.append(torch.ones(N, device=device))

        Z = self.core(A_list, X_list, mask_list)
        return Z

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

Exact upstream API to be inspected during the adapter task. If `forward(...)` signature differs from `(A_list, X_list, mask_list)`, the adapter is adjusted accordingly.

### 7.3 PyTorch 2.4 compatibility shims

Anticipated minor shims to add at top of `src/models/evolvegcn.py`:

```python
import torch
if not hasattr(torch, '_six'):
    torch._six = SimpleNamespace(inf=float('inf'), nan=float('nan'))
```

If patches grow beyond ~5 lines, the implementer reports BLOCKED and the plan switches to a reimplementation path (separate plan revision).

## 8. Test pair sharing & fair comparison

Plan 1/2 already established deterministic test pair generation via `seed_test=999` in `scripts/train.py:_build_test_pairs`. Same `DynamicGraph` (same dataset config, same loader) yields the same test pairs.

→ GCN_MA's Plan 2 results in `results/metrics.jsonl` and EvolveGCN-O's Plan 3a results will be **comparable on the exact same pairs** without explicit caching.

## 9. Training, evaluation, orchestration

### 9.1 Reused from Plan 1/2

- `src/data/loaders/*` (loader for each of 6 datasets)
- `src/training/trainer.py` (DynamicLinkPredictor agnostic loop)
- `src/training/negative_sampling.py`
- `src/eval/evaluator.py` and `src/eval/metrics.py`
- `src/utils/seed.py`, `src/utils/logging.py`

### 9.2 Modified in Plan 3a

`scripts/train.py`: replace single-model construction with helper `_build_model(model_cfg, graph) -> DynamicLinkPredictor` switching on `model_cfg["name"]`. Set `record["model"]` from config name so aggregation can distinguish.

`scripts/run_seeds.sh`: accept positional args `(dataset, model)` with `model` defaulting to `"gcn_ma"` for backward compatibility. Reads `configs/experiments/<model>_<dataset>.yaml`.

`scripts/aggregate_results.py`: new file generalizing `aggregate_gcn_ma_results.py`. Supports:
- `--model <name>`: single-model summary (current behavior).
- `--models gcn_ma evolvegcn_o`: cross-model comparison table.

### 9.3 Full execution

18 EvolveGCN-O runs. Estimated wall-clock (based on Plan 2 GCN_MA times scaled by EvolveGCN's lighter architecture):

| Dataset | Estimated per-run | × 3 seeds |
|---|---|---|
| CollegeMsg | ~50s | ~2.5 min |
| Bitcoinotc | ~7 min | ~21 min |
| EUT | ~3 min | ~9 min |
| Mooc | ~20 min | ~60 min |
| LastFM | ~6 min | ~18 min |
| Wikipedia | ~9 min | ~27 min |
| **Total** | | **~2.5 hours** |

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| IBM/EvolveGCN incompatible with PyTorch 2.4 | Medium-High | 1-day timebox on adapter task; >5 lines of shims → escalate to reimplement |
| Upstream API differs from documented expectation | Medium | Smoke test inspects actual signature before implementing forward |
| Sparse adjacency on CUDA crashes | Low | torch 2.4 supports sparse coo on CUDA; fallback to dense if N < 10k |
| OOM on largest datasets | Low | hidden_dim=64, num_layers=2; RTX 3060 12GB has plenty of headroom |
| AUC dramatically below paper | Medium | Document in reproduction log; possible causes (hyperparameter mismatch, different node-feature convention) listed |
| Empty validation snapshot (EUT-like issue) | Low | Trainer already handles via walk-forward fix; EvolveGCN inherits |

## 11. Testing

**Tier 1 — Unit / smoke (CPU, <30s):**
- `test_can_import_upstream_egcn_o`
- `test_can_construct_egcn_o`
- `test_evolvegcn_forward_shape` (50-node fake graph, T=5)
- `test_evolvegcn_gradient_flows`

**Tier 2 — Integration (CPU/GPU, ~2 min):**
- `test_evolvegcn_runs_on_collegemsg_smoke` — epochs=3, real CollegeMsg data, val_auc in [0.5, 1.0] (not NaN or stuck at 0.5)

**Tier 3 — Full reproduction (manual, ~2.5 hours):**
- 18 runs × 3 seeds via `scripts/run_seeds.sh evolvegcn_o <dataset>`.

## 12. Deliverables

- `src/models/evolvegcn.py` (adapter, ~120 LOC)
- `third_party/EvolveGCN/` submodule pinned to a documented commit
- `configs/models/evolvegcn_o.yaml` + 6 experiment configs
- 4 smoke tests passing
- 18 EvolveGCN-O rows in `results/metrics.jsonl`
- Updated `scripts/aggregate_results.py` with cross-model comparison
- `results/report/baselines_summary.md` (GCN_MA vs EvolveGCN-O mean±std table)
- Plan 3a section in `docs/reproduction-log.md`
- Tag `v0.3a-evolvegcn-o`

## 13. Spec compliance

- §7.4 Fork upstream as `third_party/` submodule: ✅ (Approach A).
- §6.6 Baselines use one-hot identity: ⚠️ Deviation — learnable embedding instead (RAM justified). Documented.
- §9.4 Baselines use authors' defaults, not tuned: ✅ (Hybrid policy: paper defaults for EvolveGCN-specific knobs; shared compute/capacity with GCN_MA for fair comparison).
- §6.4 Train/val/test split: ✅ (inherited from Plan 2's trainer).
- §6.7 Reproducibility seeding: ✅ (same 3 seeds, same test pairs).

## 14. Out of scope (deferred)

- HTGN, DyGNN, DGCN — Plans 3b/3c/3d.
- Per-dataset hyperparameter tuning.
- EvolveGCN-H variant.
- TBPTT for very long T datasets.

## 15. Open questions / pending

- Exact upstream API of `EGCN` class (`forward` signature, args namespace fields). Inspect during smoke test.
- Specific commit SHA of IBM/EvolveGCN to pin. Use latest commit on `master` as of integration day.
