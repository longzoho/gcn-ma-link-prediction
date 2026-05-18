# Plan 3d: DGCN Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate DGCN (Manessi 2020, WD-GCN variant) as the 4th and final baseline alongside GCN_MA / EvolveGCN-O / HTGN / DyGNN, producing 18 metric records (6 datasets × 3 seeds) and a 5-baseline cross-comparison table.

**Architecture:** Single-file from-scratch reimplementation at `src/models/dgcn.py`. Stack of 2 `SpectralGCNLayer` per snapshot → `nn.LSTM` over the time dimension → shared `LinkDecoderMLP`. Adjacency from `edge_index` with self-loops + symmetric normalization, built sparse on-the-fly (no dense N×N). Snapshot paradigm same as EvolveGCN-O / HTGN; no upstream submodule.

**Tech Stack:** PyTorch 2.4 (sparse_coo_tensor), PyG 2.6 Data objects (consumed but no PyG-conv used), uv-managed venv at `.venv/`. Reuses `src.models.base.DynamicLinkPredictor` and `src.models.gcn_ma.link_decoder.LinkDecoderMLP`.

**File map:**

| File | Action | Purpose |
|---|---|---|
| `src/models/dgcn.py` | create | `SpectralGCNLayer` + `DGCN` composition (~150 LOC) |
| `tests/test_dgcn_smoke.py` | create | 6 smoke tests (layer + composition + grad) |
| `configs/models/dgcn.yaml` | create | Hybrid policy hyperparams |
| `configs/models/dgcn_smoke.yaml` | create | smaller hidden/epochs for fast iteration |
| `configs/experiments/dgcn_collegemsg.yaml` | create | first dataset experiment |
| `configs/experiments/dgcn_collegemsg_smoke.yaml` | create | smoke experiment |
| `configs/experiments/dgcn_bitcoinotc.yaml` | create | dataset experiment |
| `configs/experiments/dgcn_eut.yaml` | create | dataset experiment |
| `configs/experiments/dgcn_mooc_actions.yaml` | create | dataset experiment |
| `configs/experiments/dgcn_lastfm.yaml` | create | dataset experiment |
| `configs/experiments/dgcn_wikipedia.yaml` | create | dataset experiment |
| `scripts/train.py` | modify | add `dgcn` branch to `_build_model` |
| `docs/reproduction-log.md` | modify | append Plan 3d section |
| git tag | add | `v0.3d-dgcn` |

---

## Task 1: SpectralGCNLayer with sparse symmetric normalization

**Files:**
- Create: `src/models/dgcn.py` (initial)
- Create: `tests/test_dgcn_smoke.py` (initial)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dgcn_smoke.py`:

```python
"""Smoke tests for DGCN (path B reimpl from Manessi 2020, WD-GCN variant)."""
import torch
from torch_geometric.data import Data

from src.models.dgcn import DGCN, SpectralGCNLayer


# --------------------------------------------------------------------------
# SpectralGCNLayer
# --------------------------------------------------------------------------

def test_spectral_gcn_layer_shape():
    """Single forward maps [N, in_dim] -> [N, out_dim] with finite output."""
    N, D_in, D_out = 20, 16, 32
    torch.manual_seed(0)
    x = torch.randn(N, D_in)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    layer = SpectralGCNLayer(D_in, D_out, dropout=0.0)
    out = layer(x, ei, N)
    assert out.shape == (N, D_out)
    assert torch.isfinite(out).all()


def test_spectral_gcn_layer_self_loops_keep_isolated_node_active():
    """An isolated node still receives a self-loop signal (degree >= 1)."""
    N, D = 5, 8
    x = torch.randn(N, D, requires_grad=True)
    # Node 4 isolated (no edges)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    layer = SpectralGCNLayer(D, D, dropout=0.0)
    out = layer(x, ei, N)
    assert torch.isfinite(out[4]).all()
    # Output for isolated node should be a function of its own input (self-loop)
    assert not torch.allclose(out[4], torch.zeros(D))


def test_spectral_gcn_layer_empty_edge_index():
    """When edge_index has zero edges, self-loops alone keep the layer well-defined."""
    N, D = 5, 8
    x = torch.randn(N, D)
    ei = torch.empty(2, 0, dtype=torch.long)
    layer = SpectralGCNLayer(D, D, dropout=0.0)
    out = layer(x, ei, N)
    assert out.shape == (N, D)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py -v
```

Expected: `ImportError: cannot import name 'DGCN' from 'src.models.dgcn'` (file doesn't exist yet).

- [ ] **Step 3: Implement SpectralGCNLayer in `src/models/dgcn.py`**

Create `src/models/dgcn.py`:

```python
"""DGCN baseline (Manessi 2020, WD-GCN variant) — from-scratch reimplementation.

Architecture: snapshot-based GCN stack + LSTM over time per node.
Per spec §4-5. No upstream repo exists; this is the project's own implementation.
"""
import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class SpectralGCNLayer(nn.Module):
    """One D^(-1/2) Â D^(-1/2) X W step with self-loops.

    Builds a sparse normalized adjacency on-the-fly from edge_index — no dense
    N×N matrix is ever materialized (fits Wikipedia/LastFM in 12GB easily).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        device = x.device
        # Add self-loops: each node i gets edge (i, i)
        self_loops = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)  # [2, N]
        if edge_index.numel() > 0:
            ei = torch.cat([edge_index.to(device), self_loops], dim=1)  # [2, E+N]
        else:
            ei = self_loops

        # Degree (symmetric): each edge contributes 1 to both endpoints' degrees
        ones = torch.ones(ei.shape[1], device=device)
        deg = torch.zeros(N, device=device).index_add(0, ei[0], ones)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

        # Edge weights = D^(-1/2)[src] * D^(-1/2)[dst]
        edge_weight = deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]  # [E+N]

        # Sparse adjacency Â = sum_e edge_weight[e] · e_{src,dst}
        A_hat = torch.sparse_coo_tensor(ei, edge_weight, (N, N)).coalesce()

        # Â · X · W
        agg = torch.sparse.mm(A_hat, x)  # [N, in_dim]
        out = self.linear(agg)            # [N, out_dim]
        out = torch.relu(out)
        out = self.dropout(out)
        return out


# DGCN composition (Task 2)
```

- [ ] **Step 4: Run tests to verify the 3 layer tests pass**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py::test_spectral_gcn_layer_shape tests/test_dgcn_smoke.py::test_spectral_gcn_layer_self_loops_keep_isolated_node_active tests/test_dgcn_smoke.py::test_spectral_gcn_layer_empty_edge_index -v
```

Expected: 3 passed.

If shape mismatch: check that `self.linear` projects `in_dim → out_dim`, not the other way.

If isolated-node test fails: verify self-loops added BEFORE degree computation (otherwise isolated node has degree 0, deg_inv_sqrt = inf, NaN propagates).

- [ ] **Step 5: Commit**

```bash
git add src/models/dgcn.py tests/test_dgcn_smoke.py
git commit -m "[models] DGCN: SpectralGCNLayer with sparse normalized adjacency

Hand-rolled D^(-1/2) Â D^(-1/2) X W step using torch.sparse_coo_tensor.
Self-loops added unconditionally so isolated nodes remain well-defined.
3 smoke tests cover shape, self-loop preservation, empty edge_index."
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- pytest output before AND after
- git log --oneline -2

---

## Task 2: DGCN composition scaffold + construction test

**Files:**
- Modify: `src/models/dgcn.py`
- Modify: `tests/test_dgcn_smoke.py`

- [ ] **Step 1: Append construction test**

Append to `tests/test_dgcn_smoke.py`:

```python
# --------------------------------------------------------------------------
# DGCN composition
# --------------------------------------------------------------------------

def test_dgcn_construct():
    """Default construction wires up embedding, 2 GCN layers, 1-layer LSTM, decoder."""
    m = DGCN(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_gcn_layers=2,
        num_lstm_layers=1,
        dropout=0.1,
    )
    assert m is not None
    assert m.num_nodes == 50
    assert m.hidden_dim == 64
    assert m.node_emb.weight.shape == (50, 64)
    assert len(m.gcn_layers) == 2
    assert m.gcn_layers[0].in_dim == 64   # feat_dim
    assert m.gcn_layers[0].out_dim == 64  # hidden_dim
    assert m.gcn_layers[1].in_dim == 64   # hidden_dim
    assert m.gcn_layers[1].out_dim == 64  # hidden_dim
    assert isinstance(m.lstm, torch.nn.LSTM)
    assert m.lstm.input_size == 64
    assert m.lstm.hidden_size == 64
    assert m.lstm.num_layers == 1
    assert hasattr(m, "decoder")
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py::test_dgcn_construct -v
```

Expected: `ImportError: cannot import name 'DGCN' from 'src.models.dgcn'`.

- [ ] **Step 3: Add DGCN class scaffold**

Replace the final comment line in `src/models/dgcn.py` (`# DGCN composition (Task 2)`) with the full class:

```python
class DGCN(DynamicLinkPredictor):
    """DGCN baseline (Manessi 2020, WD-GCN variant).

    WD-GCN = "Waterfall" Dynamic GCN: stack of GCN layers per snapshot, then
    a single LSTM over the time dimension per node, then a shared MLP decoder.
    No NRNAE — fair-baseline policy. See spec §4.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError(f"num_gcn_layers must be >= 1, got {num_gcn_layers}")
        if num_lstm_layers < 1:
            raise ValueError(f"num_lstm_layers must be >= 1, got {num_lstm_layers}")
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Learnable node features (replaces paper's one-hot I_N — RAM constraint,
        # documented in reproduction-log.md as a Plan 2 deviation).
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # GCN stack: layer 0 takes feat_dim, subsequent layers take hidden_dim
        layers = []
        for i in range(num_gcn_layers):
            in_d = feat_dim if i == 0 else hidden_dim
            layers.append(SpectralGCNLayer(in_d, hidden_dim, dropout=dropout))
        self.gcn_layers = nn.ModuleList(layers)

        # Per-node temporal LSTM over the GCN stack output sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Shared decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step: int) -> torch.Tensor:
        """Forward — implemented in Task 3."""
        raise NotImplementedError("DGCN.forward — implemented in Task 3")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py -v
```

Expected: 4 passed (3 layer tests + construction test).

If `ValueError` on `num_gcn_layers`: the validation block should pass since test uses default `num_gcn_layers=2`.

If `len(m.gcn_layers) != 2`: verify the `for i in range(num_gcn_layers)` loop runs the right number of times.

- [ ] **Step 5: Commit**

```bash
git add src/models/dgcn.py tests/test_dgcn_smoke.py
git commit -m "[models] DGCN: composition scaffold (embedding + GCN stack + LSTM + decoder)

forward() raises NotImplementedError — wired in Task 3.
Construction validates num_gcn_layers >= 1, num_lstm_layers >= 1."
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- pytest output (expect 4 passed)
- git log --oneline -2

---

## Task 3: DGCN forward (build snapshot stack, LSTM over time, return Z^t)

**Files:**
- Modify: `src/models/dgcn.py`
- Modify: `tests/test_dgcn_smoke.py`

- [ ] **Step 1: Append failing forward tests**

Append to `tests/test_dgcn_smoke.py`:

```python
def _make_dummy_snapshots(N: int, T: int, max_edges: int = 20) -> list[Data]:
    """Build T dummy snapshots with random edges and timestamps."""
    snaps = []
    torch.manual_seed(0)
    for t in range(T):
        e = max(1, max_edges // 2)
        ei = torch.randint(0, N, (2, e))
        ts = torch.rand(e, dtype=torch.float64) + float(t)
        d = Data(edge_index=ei, num_nodes=N)
        d.edge_ts = ts  # DGCN ignores this; included for cache fmt3 consistency
        snaps.append(d)
    return snaps


def test_dgcn_forward_shape():
    """forward(snapshots, T-1) returns Z [N, hidden_dim]."""
    N, T, D = 50, 5, 64
    m = DGCN(num_nodes=N, feat_dim=D, hidden_dim=D, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()


def test_dgcn_gradient_flows():
    """Backward populates at least one trainable param with finite, non-zero grad."""
    N, T, D = 50, 4, 64
    m = DGCN(num_nodes=N, feat_dim=D, hidden_dim=D, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    Z.sum().backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no trainable parameter received gradient"


def test_dgcn_handles_empty_snapshot_in_middle():
    """A snapshot with edge_index.numel() == 0 still produces finite output."""
    N, T, D = 30, 3, 32
    m = DGCN(num_nodes=N, feat_dim=D, hidden_dim=D, dropout=0.0)
    # Snapshot 1 has zero edges; 0 and 2 have edges
    snaps = []
    for t in range(T):
        if t == 1:
            ei = torch.empty(2, 0, dtype=torch.long)
        else:
            ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        d = Data(edge_index=ei, num_nodes=N)
        d.edge_ts = torch.tensor([], dtype=torch.float64) if t == 1 else torch.tensor([0.1, 0.2, 0.3])
        snaps.append(d)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py::test_dgcn_forward_shape -v
```

Expected: `NotImplementedError: DGCN.forward — implemented in Task 3`.

- [ ] **Step 3: Replace the forward body**

In `src/models/dgcn.py`, replace the `raise NotImplementedError(...)` body with:

```python
    def forward(self, snapshots, time_step: int) -> torch.Tensor:
        """Return per-node embedding Z^t [N, hidden_dim] at end of snapshot `time_step`.

        Pipeline:
            1. For each snapshot t' ∈ [0, time_step]:
                 X = self.node_emb.weight              # [N, feat_dim], shared learnable
                 ei_sym = symmetrize(snapshots[t'].edge_index)
                 H = X
                 for layer in self.gcn_layers:
                     H = layer(H, ei_sym, N)
                 gcn_outputs.append(H)                  # [N, hidden_dim]
            2. Stack: [time_step+1, N, hidden_dim] -> permute to [N, time_step+1, hidden_dim]
            3. LSTM (batch=N, seq_len=time_step+1) -> last hidden state per node
            4. Return [N, hidden_dim]
        """
        N = self.num_nodes
        device = self.node_emb.weight.device
        gcn_outputs = []
        for t in range(time_step + 1):
            snap = snapshots[t]
            ei = snap.edge_index.to(device) if snap.edge_index.numel() > 0 else snap.edge_index
            # Symmetrize for bipartite datasets (mooc, wikipedia, lastfm).
            # Harmless on already-undirected datasets — adds duplicate edges
            # whose weights renormalize together.
            if ei.numel() > 0:
                ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                ei_sym = ei
            h = self.node_emb.weight  # [N, feat_dim]
            for layer in self.gcn_layers:
                h = layer(h, ei_sym, N)
            gcn_outputs.append(h)

        # [T, N, D] -> [N, T, D] for batch_first LSTM
        stacked = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)
        out, _ = self.lstm(stacked)  # [N, T, D]
        return out[:, -1, :]  # last time step per node
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dgcn_smoke.py -v
```

Expected: 7 passed (3 layer + construct + 3 composition forward tests).

If shape mismatch on `test_dgcn_forward_shape`: check `permute(1, 0, 2)` is correct (stacking on dim=0 gives [T, N, D]; permute to [N, T, D] for batch_first).

If `test_dgcn_gradient_flows` fails: verify both the embedding AND the LSTM trainable params get grad. If only embedding has grad, the LSTM might be in `eval` mode or its output is being detached somewhere.

If `test_dgcn_handles_empty_snapshot_in_middle` fails with NaN: check `SpectralGCNLayer.forward` self-loop addition path handles `edge_index.numel() == 0` correctly (Task 1 test already covers this; this test verifies the composition handles it through the full forward).

- [ ] **Step 5: Run full test suite for regression**

```bash
.venv/bin/pytest tests/ -v 2>&1 | tail -15
```

Expected: prior baseline tests still pass, total grows by 7 (DGCN smoke).

- [ ] **Step 6: Commit**

```bash
git add src/models/dgcn.py tests/test_dgcn_smoke.py
git commit -m "[models] DGCN: forward — per-snapshot GCN stack + temporal LSTM

For each snapshot t' in [0, time_step]: feed shared learnable node
embedding through the GCN stack with symmetrized adjacency; collect
per-snapshot outputs; stack as [N, T, D] and run a single batch-first
LSTM over time; return the last-step hidden state as Z^t.

3 new composition tests cover shape, gradient flow, and an empty
middle-snapshot edge case."
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- pytest output (expect 7 DGCN tests passing + no regression on suite)
- Full suite count
- git log --oneline -3

---

## Task 4: Model + smoke + first experiment configs

**Files:**
- Create: `configs/models/dgcn.yaml`
- Create: `configs/models/dgcn_smoke.yaml`
- Create: `configs/experiments/dgcn_collegemsg.yaml`
- Create: `configs/experiments/dgcn_collegemsg_smoke.yaml`

- [ ] **Step 1: Create `configs/models/dgcn.yaml`**

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

- [ ] **Step 2: Create `configs/models/dgcn_smoke.yaml`**

```yaml
name: dgcn
feat_dim: 32
hidden_dim: 32
num_gcn_layers: 2
num_lstm_layers: 1
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 3
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

- [ ] **Step 3: Create `configs/experiments/dgcn_collegemsg.yaml`**

```yaml
experiment_name: dgcn_collegemsg
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Create `configs/experiments/dgcn_collegemsg_smoke.yaml`**

```yaml
experiment_name: dgcn_collegemsg_smoke
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/dgcn_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_smoke.jsonl
```

- [ ] **Step 5: Verify all 4 YAML files parse**

```bash
for cfg in configs/models/dgcn.yaml configs/models/dgcn_smoke.yaml configs/experiments/dgcn_collegemsg.yaml configs/experiments/dgcn_collegemsg_smoke.yaml; do
    .venv/bin/python -c "import yaml; d=yaml.safe_load(open('$cfg')); print('OK $cfg ->', list(d.keys())[:3])"
done
```

Expected: 4 OK lines.

- [ ] **Step 6: Commit**

```bash
git add configs/models/dgcn.yaml configs/models/dgcn_smoke.yaml configs/experiments/dgcn_collegemsg.yaml configs/experiments/dgcn_collegemsg_smoke.yaml
git commit -m "[configs] DGCN model + smoke + CollegeMsg experiment configs"
```

## Report
- YAML parse output for all 4
- git log --oneline -2

---

## Task 5: CLI dispatch — add `dgcn` branch to `_build_model`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Read current `_build_model`**

```bash
grep -n "elif name ==" scripts/train.py
```

It should have `gcn_ma`, `evolvegcn_o`, `htgn`, `dygnn` branches.

- [ ] **Step 2: Add dgcn branch**

In `scripts/train.py`, locate the existing `dygnn` elif block:

```python
    elif name == "dygnn":
        from src.models.dygnn.model import DyGNN
        return DyGNN(
            num_nodes=graph.num_nodes,
            hidden_dim=model_cfg["hidden_dim"],
            node_memory_dim=model_cfg["node_memory_dim"],
            edge_dim=model_cfg["edge_dim"],
            dropout=model_cfg["dropout"],
            decay_method=model_cfg["decay_method"],
            decay_rate=model_cfg["decay_rate"],
        )
    raise ValueError(f"Unknown model name: {name!r}")
```

Insert the dgcn branch immediately BEFORE the `raise ValueError(...)` line:

```python
    elif name == "dygnn":
        from src.models.dygnn.model import DyGNN
        return DyGNN(
            num_nodes=graph.num_nodes,
            hidden_dim=model_cfg["hidden_dim"],
            node_memory_dim=model_cfg["node_memory_dim"],
            edge_dim=model_cfg["edge_dim"],
            dropout=model_cfg["dropout"],
            decay_method=model_cfg["decay_method"],
            decay_rate=model_cfg["decay_rate"],
        )
    elif name == "dgcn":
        from src.models.dgcn import DGCN
        return DGCN(
            num_nodes=graph.num_nodes,
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_gcn_layers=model_cfg["num_gcn_layers"],
            num_lstm_layers=model_cfg["num_lstm_layers"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unknown model name: {name!r}")
```

- [ ] **Step 3: Sanity check imports**

```bash
.venv/bin/python scripts/train.py --help
```

Expected: argparse usage without ImportError.

- [ ] **Step 4: Regression check (4 prior models)**

```bash
for model in gcn_ma evolvegcn_o htgn dygnn; do
    echo "=== $model smoke ==="
    .venv/bin/python scripts/train.py --config configs/experiments/${model}_collegemsg_smoke.yaml 2>&1 | tail -3
done
```

Expected: 4 final lines each (closing brace of JSON record). No ImportError.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "[scripts] _build_model: add dgcn branch (5 models supported)"
```

## Report
- argparse help output (first 3 lines)
- 4 smoke last-3-lines blocks
- git log --oneline -2

---

## Task 6: Smoke train DGCN on CollegeMsg

**Files:**
- (uses existing `configs/experiments/dgcn_collegemsg_smoke.yaml` from Task 4)

- [ ] **Step 1: Run smoke training**

```bash
.venv/bin/python scripts/train.py --config configs/experiments/dgcn_collegemsg_smoke.yaml 2>&1 | tail -25
```

Expected: ~10-30s wall clock for 3 epochs (CollegeMsg has 47 snapshots, GCN+LSTM is light). Final JSON record with `"model": "dgcn"` and AUC roughly in the 0.85-0.95 range.

If runtime exceeds 5 min: something is wrong — likely dense adjacency materialization. Inspect `SpectralGCNLayer.forward`; the sparse path should keep it fast.

If AUC < 0.7: smoke epochs (3) may simply be too few for DGCN on CollegeMsg. Verify training loss is decreasing, then accept — full 200-epoch run will land properly.

- [ ] **Step 2: Verify smoke record landed**

```bash
.venv/bin/python -c "
import json
recs = [json.loads(l) for l in open('results/metrics_smoke.jsonl')]
dgcn = [r for r in recs if r['model'] == 'dgcn']
assert len(dgcn) >= 1, 'no DGCN smoke record found'
r = dgcn[-1]
print(f\"experiment={r['experiment_name']}  auc={r['auc']:.4f}  val_auc={r['val_auc']:.4f}  runtime={r['runtime_s']:.1f}s\")
"
```

Expected: one line showing `experiment=dgcn_collegemsg_smoke ...`.

- [ ] **Step 3: Commit the smoke config (if not yet committed from Task 4)**

The smoke config was already committed in Task 4. Nothing new to commit here; this is a validation task only.

## Report
- Smoke command tail output (last 15 lines)
- Single-line smoke record summary
- Wall-clock runtime

---

## Task 7: Remaining 5 experiment configs (full 6-dataset coverage)

**Files:**
- Create: `configs/experiments/dgcn_bitcoinotc.yaml`
- Create: `configs/experiments/dgcn_eut.yaml`
- Create: `configs/experiments/dgcn_mooc_actions.yaml`
- Create: `configs/experiments/dgcn_lastfm.yaml`
- Create: `configs/experiments/dgcn_wikipedia.yaml`

- [ ] **Step 1: Create `configs/experiments/dgcn_bitcoinotc.yaml`**

```yaml
experiment_name: dgcn_bitcoinotc
seed: 42
dataset_config: configs/datasets/bitcoinotc.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 2: Create `configs/experiments/dgcn_eut.yaml`**

```yaml
experiment_name: dgcn_eut
seed: 42
dataset_config: configs/datasets/eut.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 3: Create `configs/experiments/dgcn_mooc_actions.yaml`**

```yaml
experiment_name: dgcn_mooc_actions
seed: 42
dataset_config: configs/datasets/mooc_actions.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Create `configs/experiments/dgcn_lastfm.yaml`**

```yaml
experiment_name: dgcn_lastfm
seed: 42
dataset_config: configs/datasets/lastfm.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 5: Create `configs/experiments/dgcn_wikipedia.yaml`**

```yaml
experiment_name: dgcn_wikipedia
seed: 42
dataset_config: configs/datasets/wikipedia.yaml
model_config: configs/models/dgcn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 6: Verify all 6 experiment configs parse**

```bash
for cfg in configs/experiments/dgcn_*.yaml; do
    .venv/bin/python -c "import yaml; d=yaml.safe_load(open('$cfg')); print('OK $cfg ->', d['experiment_name'])"
done
```

Expected: 6 OK lines (collegemsg + smoke + 5 new).

- [ ] **Step 7: Commit**

```bash
git add configs/experiments/dgcn_bitcoinotc.yaml configs/experiments/dgcn_eut.yaml configs/experiments/dgcn_mooc_actions.yaml configs/experiments/dgcn_lastfm.yaml configs/experiments/dgcn_wikipedia.yaml
git commit -m "[configs] DGCN experiment configs (5 datasets: bitcoinotc, eut, mooc_actions, lastfm, wikipedia)"
```

## Report
- YAML parse output (6 OK lines)
- git log --oneline -2

---

## Task 8: Full 18-run experiment (6 datasets × 3 seeds)

**Files:**
- (uses existing `scripts/run_seeds.sh`)

- [ ] **Step 1: Verify all DGCN caches exist (one-time loader run per dataset)**

Run the smoke for each dataset first to trigger cache build (if not already cached from prior plans):

```bash
for ds in bitcoinotc eut mooc_actions lastfm wikipedia; do
    echo "=== Cache warm-up: $ds ==="
    # Create a one-off smoke experiment that loads the dataset (epochs=1)
    cat > /tmp/dgcn_${ds}_warmup.yaml <<EOF
experiment_name: dgcn_${ds}_warmup
seed: 42
dataset_config: configs/datasets/${ds}.yaml
model_config: configs/models/dgcn_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_warmup.jsonl
EOF
    timeout 600 .venv/bin/python scripts/train.py --config /tmp/dgcn_${ds}_warmup.yaml 2>&1 | tail -3
done
```

Expected: each dataset finishes within timeout (caches were built in prior plans; loader is a no-op on existing cache). LastFM may take longest (~5-10 min to warm up if cache miss).

If a dataset's cache doesn't exist yet (fresh project), this may run 1 full smoke epoch — acceptable, just slower.

- [ ] **Step 2: Launch full 18-run sequentially in background**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    echo "=========================================="
    echo "=== DATASET: $ds (3 seeds) ==="
    echo "=========================================="
    bash scripts/run_seeds.sh $ds dgcn 2>&1 | tail -10
    echo
done
echo "=== ALL 18 DGCN RUNS COMPLETE ==="
date
```

Run this with `run_in_background: true`. Estimated 5-7h wall clock.

If a seed crashes (OOM on LastFM, etc.):
- Capture the log: `results/logs/dgcn_<dataset>_seed<N>_*.log`
- Resume just that one: `.venv/bin/python scripts/train.py --config results/configs_runtime/dgcn_<dataset>_seed<N>.yaml`

If LastFM OOMs: fall back to `hidden_dim=32` for that dataset by creating `configs/models/dgcn_lastfm.yaml` with reduced dim, and editing `configs/experiments/dgcn_lastfm.yaml` to reference it. Document the deviation in reproduction-log.

- [ ] **Step 3: Verify 18 DGCN records exist**

```bash
.venv/bin/python -c "
import json, statistics
recs = [json.loads(l) for l in open('results/metrics.jsonl')]
dgcn = [r for r in recs if r['model'] == 'dgcn']
print(f'Total DGCN records: {len(dgcn)}/18')
by_ds = {}
for r in dgcn:
    by_ds.setdefault(r['dataset'], []).append(r['auc'])
print()
print('Mean ± std per dataset:')
for ds in ['collegemsg', 'bitcoinotc', 'eut', 'mooc_actions', 'lastfm', 'wikipedia']:
    aucs = by_ds.get(ds, [])
    if aucs:
        m = statistics.mean(aucs); s = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        print(f'  {ds:15s} N={len(aucs)} mean={m:.4f} std={s:.4f}')
    else:
        print(f'  {ds:15s} N=0 (MISSING)')
"
```

Expected: `Total DGCN records: 18/18`, all 6 datasets with N=3.

- [ ] **Step 4: Generate 5-model cross-comparison**

```bash
.venv/bin/python scripts/aggregate_results.py --models gcn_ma evolvegcn_o htgn dygnn dgcn > results/report/baselines_summary.md
cat results/report/baselines_summary.md
```

Expected: 6-row table with AUC ± std for each of the 5 models on each of 6 datasets (DyGNN row shows "—" on lastfm since it was skipped).

- [ ] **Step 5: Commit the 5-baseline summary**

```bash
git add results/report/baselines_summary.md
git commit -m "[results] DGCN full 18-run + 5-baseline cross-comparison"
```

## Report
- DGCN per-dataset mean±std table
- Final cross-comparison table (or a confirmation it generated)
- Wall-clock time for the 18-run

---

## Task 9: Update reproduction-log + tag v0.3d-dgcn

**Files:**
- Modify: `docs/reproduction-log.md`

- [ ] **Step 1: Append Plan 3d section**

Open `docs/reproduction-log.md` and append:

```markdown

---

## Plan 3d: DGCN baseline integration

### Approach taken

**From-scratch reimplementation** — DGCN has no canonical repo (per project root design spec §7.4). Single-file at `src/models/dgcn.py` (~150 LOC). WD-GCN variant from Manessi 2020 ("Dynamic Graph Convolutional Networks", *Pattern Recognition*). Architecture:

```
For each snapshot t:
    X = nn.Embedding[node_ids]                  # [N, 64], shared learnable
    A_hat = sym_normalize(edge_index + self_loops)
    H = SpectralGCNLayer(X, A_hat)              # × 2 stacked
    gcn_outputs.append(H)
LSTM([H^0, ..., H^t]) -> last hidden -> Z^t [N, 64]
predict_link via shared LinkDecoderMLP.
```

Sparse adjacency: `torch.sparse_coo_tensor` on-the-fly per snapshot, no dense N×N (fits Wikipedia/LastFM in 12GB easily).

### Hyperparameter policy (Hybrid)

Shared with GCN_MA / EvolveGCN-O / HTGN / DyGNN: `feat_dim=64`, `hidden_dim=64`, `dropout=0.1`, `lr=1e-3`, `weight_decay=1e-5`, Adam, `epochs=200`, patience 20, `grad_clip_max_norm=5.0`.

DGCN-specific: `num_gcn_layers=2`, `num_lstm_layers=1`.

### Deviations from Manessi 2020

1. **Learnable `nn.Embedding` instead of one-hot `I_N`** — Plan 2 deviation carried forward, RAM constraint on Wikipedia/LastFM.
2. **Symmetric adjacency for bipartite datasets** — Plans 3a/3b/3c fix, paper assumed undirected.
3. **Shared `LinkDecoderMLP` decoder** — same as all other baselines, not paper's scoring head.
4. **Adam optimizer** — paper used SGD; project Hybrid policy.
5. **Default `hidden_dim=64`** — paper didn't report; project Hybrid default.

### Final results — 6 datasets × 3 seeds (18 records)

_TODO: paste DGCN section of `results/report/baselines_summary.md` once Task 8 lands._

### Cross-model observations (5-baseline)

_TODO: 2-3 bullets after the 5-baseline table is finalized — where DGCN wins/loses vs GCN_MA / EvolveGCN-O / HTGN / DyGNN; whether the simple GCN-then-LSTM stack is competitive or if more complex temporal models dominate._

### Engineering wins

- **Pure-PyTorch sparse implementation** — no PyG-conv dependency, no upstream submodule. Reads like the math.
- **6 datasets in one run** — DGCN handles LastFM without the compute issues DyGNN had (snapshot-based, not edge-sequence).
- **5th baseline** — completes the cross-comparison from the paper's Table 2.

### Carry-forwards to Plan 4

- All 5 baselines integrated; metrics.jsonl has full coverage minus DyGNN's LastFM.
- Reproduction-log has per-plan sections — Plan 4 aggregates into thesis-ready writeup.
- `aggregate_results.py --models gcn_ma evolvegcn_o htgn dygnn dgcn` produces the canonical comparison.
- No more baselines to add. Plan 4 is final aggregation, plots, thesis assets.
```

Replace the two `_TODO: ..._` placeholders with the actual results from Task 8.

Concretely:
1. Read the per-dataset block from Task 8 step 3 output and paste into the "Final results" section formatted as a markdown table with columns `Dataset | DGCN AUC | DGCN AP | best vs other 4 baselines`.
2. Write 2-3 bullets in "Cross-model observations" based on the rankings.

- [ ] **Step 2: Verify reproduction-log renders**

```bash
tail -60 docs/reproduction-log.md
```

Expected: clean markdown ending at the "Carry-forwards to Plan 4" section, no `_TODO:_` placeholders left.

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reproduction-log.md
git commit -m "[docs] Plan 3d final: DGCN 18-run results + 5-baseline cross-model table"
git tag v0.3d-dgcn
git log --oneline -5
git tag
```

Expected: latest commit is the docs commit; `v0.3d-dgcn` in tag list alongside `v0.3c-dygnn`, `v0.3b-htgn`, etc.

## Report
- final `docs/reproduction-log.md` tail (last 50 lines)
- `git tag` output
- `git log --oneline -5`

---

## Done

After Task 9, the project has 5 baselines integrated (GCN_MA, EvolveGCN-O, HTGN, DyGNN, DGCN), 87 metric records total (18+18+18+15+18), and a 5-baseline cross-comparison table. Plan 4 starts next: final aggregation, plots, thesis writeup.
