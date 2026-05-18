# Plan 3d: DGCN Integration — Design Spec

**Date:** 2026-05-18
**Status:** Approved (brainstorming) — ready for plan
**Baseline:** Manessi et al. 2020, "Dynamic Graph Convolutional Networks", *Pattern Recognition*

---

## 1. Goal

Add **DGCN** as the 4th and final baseline alongside GCN_MA, EvolveGCN-O, HTGN, DyGNN. Specifically, reproduce the **WD-GCN** (Waterfall Dynamic GCN) variant from Manessi 2020. Output: 18 metric records (6 datasets × 3 seeds) added to `results/metrics.jsonl`; reproduction-log section; tag `v0.3d-dgcn`.

## 2. Scope

**In scope:**
- WD-GCN variant only.
- 6 datasets: `collegemsg`, `bitcoinotc`, `eut`, `mooc_actions`, `lastfm`, `wikipedia`.
- 3 seeds: 42, 123, 2024.
- Single-file model at `src/models/dgcn.py` (~150 LOC).
- Smoke tests + 1 dataset experiment first, then full 6-dataset run.

**Out of scope:**
- CD-GCN variant (interleaved GCN/LSTM, no plan to implement).
- Hyperparameter grid search beyond Hybrid policy.
- Plan 4 aggregation, plots, thesis writeup (handled separately).

## 3. Why path A/B fallback does NOT apply

Plans 3a/3b/3c had A→B fallback because they wrap upstream repos. **DGCN has no canonical repo** — design spec §7.4 of the project root already specified "Likely reimplement from Manessi et al. 2020 in `src/models/dgcn.py`." There is no fork attempt to time-box; this is reimplement-only from the start. The "approach" we picked is which reimplementation strategy.

## 4. Architecture (WD-GCN)

**Paradigm:** Snapshot-based (same as GCN_MA, EvolveGCN-O, HTGN — NOT edge-sequence).

```
For each snapshot t ∈ [0, T-1]:
    X^t = nn.Embedding[node_ids]                # [N, F=64], learnable, shared across t
    A^t = symmetric_normalize(edge_index^t + I) # standard D^(-1/2)·Â·D^(-1/2)
    H1^t = ReLU(A^t · X^t · W1)                 # [N, hidden]
    H1^t = dropout(H1^t)
    H2^t = ReLU(A^t · H1^t · W2)                # [N, hidden]
    H2^t = dropout(H2^t)
    # H2^t is the per-snapshot GCN output

Per-node temporal LSTM (Waterfall = LSTM AFTER full GCN stack):
    stacked = stack([H2^0, ..., H2^t], dim=0)   # [t+1, N, hidden]
    seq = stacked.permute(1, 0, 2)              # [N, t+1, hidden] — batch=N, seq_len=t+1
    Z^t, _ = LSTM(seq, batch_first=True)        # [N, t+1, hidden]
    Z^t = Z^t[:, -1, :]                         # last time step → [N, hidden]

Decoder:
    P(u, v) = sigmoid(LinkDecoderMLP(Z^t[u] ⊕ Z^t[v]))
```

**No NRNAE.** Fair-baseline policy: DGCN sees only the raw temporal graph. NRNAE features are GCN_MA-specific.

## 5. Components & file structure

**Single file**: `src/models/dgcn.py` (~150 LOC total).

### 5.1 `SpectralGCNLayer(in_dim, out_dim, dropout=0.1)`

One `Ĥ = ReLU(D̂^(-1/2) · Â · D̂^(-1/2) · X · W)` step.

- `forward(X: [N, in_dim], edge_index: [2, E], N: int) → [N, out_dim]`
- Builds the symmetric-normalized adjacency on-the-fly from `edge_index` using `torch.sparse_coo_tensor` (no dense N×N — fits Wikipedia/LastFM in memory).
- Adds self-loops via `cat([edge_index, arange(N).unsqueeze(0).repeat(2, 1)], dim=1)` before normalization.
- ~30 LOC.

### 5.2 `DGCN(DynamicLinkPredictor)`

Composition module.

- `__init__(num_nodes, feat_dim=64, hidden_dim=64, num_gcn_layers=2, num_lstm_layers=1, dropout=0.1)`
- Layers:
  - `self.node_emb = nn.Embedding(num_nodes, feat_dim)` (Xavier init)
  - `self.gcn_layers = nn.ModuleList([SpectralGCNLayer(feat_dim if i==0 else hidden_dim, hidden_dim, dropout) for i in range(num_gcn_layers)])`
  - `self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)`
  - `self.decoder = LinkDecoderMLP(embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)`

### 5.3 `forward(snapshots, time_step) → [N, hidden_dim]`

```python
def forward(self, snapshots, time_step):
    N = self.num_nodes
    device = self.node_emb.weight.device
    gcn_outputs = []
    for t in range(time_step + 1):
        snap = snapshots[t]
        ei = snap.edge_index.to(device)
        # Symmetrize for bipartite (mooc, wikipedia, lastfm)
        ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
        x = self.node_emb.weight  # [N, feat_dim]
        for layer in self.gcn_layers:
            x = layer(x, ei_sym, N)
        gcn_outputs.append(x)
    stacked = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)  # [N, T, D]
    out, _ = self.lstm(stacked)
    return out[:, -1, :]  # [N, D]
```

### 5.4 `predict_link(Z, edges)`

3 LOC, delegates to `self.decoder(Z, edges)`.

## 6. Data flow

- Loader: existing `SNAPTemporalLoader.build()` — uses cache fmt3 (DyGNN's edge_ts ignored).
- `data.x` and `data.S_hat` from the cache: **ignored**. DGCN uses raw `edge_index` only.
- Per-snapshot `Data(edge_index=[2, E], num_nodes=N)` is sufficient input.
- Trainer: existing `train_dynamic()` in `src/training/trainer.py` — no changes.

## 7. Hyperparameters (Hybrid policy)

Shared with all 4 prior baselines, matching `configs/models/{gcn_ma,evolvegcn_o,htgn,dygnn}.yaml`:

```yaml
name: dgcn
feat_dim: 64
hidden_dim: 64
num_gcn_layers: 2
num_lstm_layers: 1
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 200
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

Smoke variant: `hidden_dim=32`, `epochs=3` (for fast iteration).

## 8. Testing

`tests/test_dgcn_smoke.py` (~80 LOC, ~6 tests):

1. `test_spectral_gcn_layer_shape` — single layer `[N, in_dim] → [N, out_dim]` with finite output.
2. `test_spectral_gcn_layer_handles_self_loops` — verify self-loop addition (a node always has at least one neighbor: itself).
3. `test_dgcn_construct` — instantiate with default args; check `node_emb` and `gcn_layers` modules attached.
4. `test_dgcn_forward_shape` — `model(snapshots, T-1).shape == (N, D)` on 5 dummy snapshots.
5. `test_dgcn_gradient_flows` — backward populates ≥1 trainable param with finite, non-zero grad.
6. `test_dgcn_handles_empty_snapshot` — when `edge_index.numel() == 0`, layer falls back to identity-only adjacency (self-loops keep it well-defined).

## 9. Compute estimate

WD-GCN per-epoch ≈ GCN_MA per-epoch (replace MultiHeadAttn with single LSTM cell). Reference: GCN_MA full 18-run took ~6h. DGCN dự kiến ~6-8h trên 12GB RTX 3060.

Per-dataset rough estimates (from comparable models):
- collegemsg: ~3 min/seed
- bitcoinotc: ~6 min/seed
- eut: ~55 min/seed (T=127 → long LSTM unroll)
- mooc_actions: ~4 min/seed
- lastfm: ~30-60 min/seed (largest)
- wikipedia: ~8 min/seed

Total across 6 datasets × 3 seeds: ~5-7h wall-clock (eut and lastfm dominate; others are minutes). Acceptable on the project's 12GB 3060 budget.

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Sparse adjacency normalization slow | Use `torch.sparse_coo_tensor.coalesce()`; reuse cached `degree^(-1/2)` per snapshot if needed |
| LSTM accumulates large compute over T=127 (eut) | Native PyTorch LSTM is C-optimized; if too slow, consider sequence chunking |
| OOM on Wikipedia (~9k nodes) | `nn.Embedding` is `[9k, 64]` = 2.3MB, fine. GCN intermediate `[9k, 64]` = 2.3MB per layer. LSTM seq `[9k, 42, 64]` = 96MB peak. OK on 12GB |
| Bipartite datasets (mooc, wikipedia, lastfm) | Symmetrize adjacency via `edge_index = cat([ei, ei.flip(0)])`. Same fix as Plan 3a/3b |
| LastFM compute | Same as GCN_MA — manageable. Not skipped (unlike DyGNN where edge-sequence was prohibitive) |

## 11. Deviations from Manessi 2020

1. **Learnable `nn.Embedding` instead of one-hot `I_N`** — Plan 2 deviation, documented in reproduction-log. RAM constraint on Wikipedia/LastFM.
2. **Symmetric adjacency for bipartite datasets** — Plan 3a/3b/3c fix, paper assumed undirected.
3. **Shared `LinkDecoderMLP` decoder** — same as all other baselines, not paper's scoring head.
4. **Adam optimizer** — paper used SGD; project-wide Hybrid policy.
5. **Hyperparameters** — paper didn't report hidden_dim/lr; project Hybrid defaults applied.

## 12. Carry-forwards from prior plans

- Loader cache fmt3 (added in Plan 3c) — DGCN ignores `edge_ts` cleanly.
- Symmetrize adjacency pattern (Plan 3a fix).
- `LinkDecoderMLP` shared decoder (Plan 1).
- Hybrid hyperparameter policy.
- Multi-seed runner `scripts/run_seeds.sh <dataset> dgcn`.

## 13. Acceptance criteria

- 6 model+smoke configs + 6 experiment configs + 1 smoke config parse.
- 6 smoke tests pass.
- Smoke training (`scripts/train.py --config configs/experiments/dgcn_collegemsg_smoke.yaml`) emits one JSON record with `model: "dgcn"` and reasonable AUC (>0.85 on collegemsg).
- Full 18-run completes without OOM or crash.
- Cross-model summary updated to 5 baselines.
- Reproduction-log Plan 3d section appended.
- Tag `v0.3d-dgcn` applied.

## 14. Out-of-band concerns

- **No upstream submodule for DGCN** — unlike Plans 3a/3b/3c, `third_party/` won't grow.
- **Plan 4 starts after this** — final aggregation, plots, thesis writeup.

## 15. Files affected

| File | Action |
|---|---|
| `src/models/dgcn.py` | create |
| `tests/test_dgcn_smoke.py` | create |
| `configs/models/dgcn.yaml` | create |
| `configs/models/dgcn_smoke.yaml` | create |
| `configs/experiments/dgcn_collegemsg.yaml` | create |
| `configs/experiments/dgcn_bitcoinotc.yaml` | create |
| `configs/experiments/dgcn_eut.yaml` | create |
| `configs/experiments/dgcn_mooc_actions.yaml` | create |
| `configs/experiments/dgcn_lastfm.yaml` | create |
| `configs/experiments/dgcn_wikipedia.yaml` | create |
| `configs/experiments/dgcn_collegemsg_smoke.yaml` | create |
| `scripts/train.py` | modify (`_build_model` adds `dgcn` branch — 5 models supported) |
| `scripts/aggregate_results.py` | no change (already supports `--models` arbitrary list) |
| `docs/reproduction-log.md` | modify (append Plan 3d section) |
| git tag | add `v0.3d-dgcn` |
