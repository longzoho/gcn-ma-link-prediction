# HTGN Baseline Integration (Plan 3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTGN (Hyperbolic Temporal Graph Network) as a baseline running end-to-end on all 6 datasets × 3 seeds. Produce a 3-model cross-comparison (GCN_MA, EvolveGCN-O, HTGN).

**Architecture:** A→B fallback. Path A primary: vendor `marlin-codes/HTGN` as git submodule, write thin adapter (~150-200 LOC) wrapping upstream encoder, project Poincaré ball output → Euclidean via log map, feed shared MLP decoder. Path B fallback (triggered by Task 2 BLOCKED): reimplement HTGN core ~300 LOC across 4 modules.

**Tech Stack:** Python 3.11, PyTorch 2.4, PyTorch Geometric 2.6. Reuses Plan 1/2/3a infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-17-htgn-integration-design.md`

**Predecessor:** Plan 3a (tag `v0.3a-evolvegcn-o`, commit `1786880`).

**Out of scope:** DyGNN, DGCN (Plans 3c/3d). HTGN-H, HVGNN variants. RAdam/RSGD optimizers. Learnable curvature.

---

## File map (path A — preferred)

```
gcn-ma-link-prediction/
├── third_party/HTGN/                              # NEW submodule
├── configs/
│   ├── models/htgn.yaml                           # NEW
│   ├── models/htgn_smoke.yaml                     # NEW (Task 8)
│   └── experiments/
│       ├── htgn_collegemsg.yaml                   # NEW
│       ├── htgn_collegemsg_smoke.yaml             # NEW (Task 8)
│       ├── htgn_bitcoinotc.yaml                   # NEW (Task 9)
│       ├── htgn_eut.yaml                          # NEW
│       ├── htgn_mooc_actions.yaml                 # NEW
│       ├── htgn_lastfm.yaml                       # NEW
│       └── htgn_wikipedia.yaml                    # NEW
├── src/models/
│   ├── hyperbolic_ops.py                          # NEW — shared log_map_0 (~30 LOC, Task 3)
│   └── htgn.py                                    # NEW — adapter (~150 LOC, Tasks 4-5)
├── scripts/train.py                               # MODIFY — add htgn branch
├── tests/
│   ├── test_hyperbolic_ops.py                     # NEW (Task 3)
│   └── test_htgn_smoke.py                         # NEW (Tasks 2, 4, 5)
└── docs/reproduction-log.md                       # MODIFY (Task 11)
```

## Path B fallback file map (if Task 2 BLOCKED)

Used INSTEAD OF `src/models/htgn.py`:

```
src/models/htgn/                                   # NEW directory
├── __init__.py
├── poincare.py                                    # NEW (~80 LOC, Task B1)
├── hgcn_layer.py                                  # NEW (~50 LOC, Task B2)
├── temporal_gru.py                                # NEW (~30 LOC, Task B3)
└── model.py                                       # NEW (~70 LOC, Task B4)
```

`third_party/HTGN/` kept for citation but not imported.

---

## Task 1: Add `marlin-codes/HTGN` as git submodule

**Files:**
- Modify: `.gitmodules`
- Create: `third_party/HTGN/`

- [ ] **Step 1: Add submodule**

```bash
git submodule add https://github.com/marlin-codes/HTGN third_party/HTGN
```

- [ ] **Step 2: Record pin SHA**

```bash
cd third_party/HTGN
PIN_SHA=$(git rev-parse HEAD)
echo "HTGN pinned commit: $PIN_SHA"
cd ../..
```

Record SHA in your report. Will appear in commit message and reproduction-log.

- [ ] **Step 3: Inspect repo top-level structure**

```bash
ls third_party/HTGN/
find third_party/HTGN -maxdepth 2 -type f -name "*.py" | head -30
```

Identify likely files: `HTGN.py`, `model.py`, or under `models/`. Note the path for Task 2.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/HTGN
git commit -m "[deps] vendor marlin-codes/HTGN as submodule, pinned to <SHA>"
```

Replace `<SHA>` with the actual pin SHA.

## Report
- **Status:** DONE | BLOCKED (clone failed)
- Pin SHA
- Repo file listing (Python files at depth ≤ 2)
- git log --oneline -2

---

## Task 2: Smoke-test upstream import + document API (1-day timebox gate)

**Files:**
- Create: `tests/test_htgn_smoke.py` (initial version with import test)

**This task gates path A vs path B.** If shims exceed ~10 lines OR upstream-API discovery takes >1 day → report BLOCKED → controller switches to path B (Task B1-B4 from the appendix at the bottom of this plan).

- [ ] **Step 1: Inspect HTGN class location and signature**

```bash
.venv/bin/python << 'EOF'
import sys
sys.path.insert(0, 'third_party/HTGN')
# Try common module names — find which works
for name in ['HTGN', 'model.HTGN', 'models.HTGN', 'main', 'models.model']:
    try:
        mod = __import__(name, fromlist=['*'])
        classes = [c for c in dir(mod) if 'HTGN' in c.upper() or 'HYPER' in c.upper() or c[0].isupper()]
        print(f'OK: {name}: classes={classes[:10]}')
    except Exception as e:
        print(f'FAIL: {name}: {type(e).__name__}: {str(e)[:80]}')
EOF
```

Find the module that contains the HTGN encoder class. Could be `HTGN`, `models.HTGN`, `script.htgn`, etc. — depends on upstream layout.

- [ ] **Step 2: Print init + forward signatures**

Once you find the module:

```bash
.venv/bin/python << 'EOF'
import sys
sys.path.insert(0, 'third_party/HTGN')
from <MODULE> import <CLASS>  # use names from Step 1
import inspect
print('=== __init__ ===')
print(inspect.signature(<CLASS>.__init__))
print(inspect.getsource(<CLASS>.__init__)[:2000])
print('=== forward ===')
print(inspect.signature(<CLASS>.forward))
print(inspect.getsource(<CLASS>.forward)[:2000])
EOF
```

Record signatures and forward body in your report. Critical for Task 4.

- [ ] **Step 3: Write `tests/test_htgn_smoke.py` import test**

```python
"""Smoke tests for HTGN upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HTGN"))


def test_can_import_upstream_htgn():
    """Upstream HTGN class must import under PyTorch 2.4."""
    # Replace MODULE_NAME and CLASS_NAME with Step 1 findings
    from <MODULE_NAME> import <CLASS_NAME>  # noqa: F401
    assert <CLASS_NAME> is not None
```

- [ ] **Step 4: Run smoke test + count required shims**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_can_import_upstream_htgn -v
```

Common failure modes and fixes:
- `ImportError: cannot import name 'inf' from 'torch._six'` → add `torch._six = SimpleNamespace(inf=float('inf'))` shim.
- `ModuleNotFoundError: No module named 'geoopt'` → install: `.venv/bin/uv pip install geoopt`. If geoopt itself fails on PyTorch 2.4, document the error and proceed to BLOCKED path.
- `ModuleNotFoundError: No module named 'tasker'` → upstream uses task-coupled imports; may not be salvageable for our adapter pattern.
- `AttributeError: module 'torch.distributions' has no attribute 'XXX'` → API rename in PyTorch 2.x; add shim if 1-2 lines.

**Count each shim line.** If total exceeds 10 lines OR you spend >1 day on shims, STOP and report BLOCKED with the specific failure mode.

- [ ] **Step 5: Run test to confirm passing**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_can_import_upstream_htgn -v
```

Expected: PASS.

If PASS within timebox → continue to Task 3.
If BLOCKED → report and controller will switch to path B (Task B1 next).

- [ ] **Step 6: Document findings in your report**

Include verbatim:
1. Module path (e.g., `HTGN`, `models.HTGN`)
2. Class name (e.g., `HTGN`, `HGNN`)
3. `__init__` signature
4. `forward` signature
5. Shim lines applied (each line on its own bullet)
6. geoopt installed? what version?

- [ ] **Step 7: Commit**

```bash
git add tests/test_htgn_smoke.py
git commit -m "[tests] smoke import test for upstream HTGN

Module: <MODULE>
Class: <CLASS>
PyTorch 2.4 compat: <N lines of shims>
geoopt: <version or N/A>"
```

## Report
- **Status:** DONE | BLOCKED | NEEDS_CONTEXT
- All 7 documentation items from Step 6
- pytest output
- git log --oneline -2

---

## Task 3: Hyperbolic log_map utility (shared between A and B paths)

**Files:**
- Create: `src/models/hyperbolic_ops.py`
- Test: `tests/test_hyperbolic_ops.py`

This module is reused regardless of which path we take. Both adapters/reimplementations call `log_map_origin` to project Poincaré ball embeddings → Euclidean tangent space before feeding the shared MLP decoder.

- [ ] **Step 1: Write the failing tests**

`tests/test_hyperbolic_ops.py`:

```python
import torch

from src.models.hyperbolic_ops import log_map_origin


def test_log_map_origin_at_zero_returns_zero():
    """log_map at the ball's origin should be zero."""
    x = torch.zeros(5, 4)
    out = log_map_origin(x, c=1.0)
    torch.testing.assert_close(out, torch.zeros(5, 4))


def test_log_map_origin_finite_for_typical_norm():
    """log_map output should be finite (no NaN/Inf) for typical embeddings."""
    torch.manual_seed(0)
    x = torch.randn(10, 8) * 0.1  # small enough to be inside ball at c=1
    out = log_map_origin(x, c=1.0)
    assert torch.isfinite(out).all(), "log_map output has NaN/Inf"


def test_log_map_origin_clamps_at_boundary():
    """log_map should not produce Inf when ||x|| is at or beyond 1/sqrt(c)."""
    # c=1 means boundary at ||x||=1. Try x close to and at the boundary.
    x_at_boundary = torch.eye(4)  # unit vectors, each with ||x||=1
    out = log_map_origin(x_at_boundary, c=1.0)
    assert torch.isfinite(out).all(), "log_map crashed at boundary"

    # Beyond boundary (atanh undefined for >1) — clamp should keep it finite
    x_beyond = torch.eye(4) * 1.5
    out = log_map_origin(x_beyond, c=1.0)
    assert torch.isfinite(out).all(), "log_map crashed beyond boundary"


def test_log_map_origin_preserves_direction():
    """log_map_0 should preserve angular direction (just scale magnitude)."""
    x = torch.tensor([[3.0, 4.0]]) * 0.01  # small enough to be in ball
    out = log_map_origin(x, c=1.0)
    # Should be in the same direction as x
    x_normed = x / x.norm()
    out_normed = out / out.norm()
    torch.testing.assert_close(x_normed, out_normed, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_hyperbolic_ops.py -v
```

Expected: `ImportError: No module named 'src.models.hyperbolic_ops'`.

- [ ] **Step 3: Implement `src/models/hyperbolic_ops.py`**

```python
"""Hyperbolic operations on the Poincaré ball.

Used by HTGN baseline (path A adapter or path B reimplement) to project
hyperbolic node embeddings to the Euclidean tangent space at the origin,
where the shared MLP decoder consumes them.
"""
import torch


def log_map_origin(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Poincaré ball → tangent space at origin (Euclidean approximation).

    log_0(x) = (1 / sqrt(c)) * arctanh(sqrt(c) * ||x||) * (x / ||x||)

    Args:
        x: hyperbolic embeddings, shape [..., D]. Each row's norm must be
           strictly less than 1/sqrt(c); rows on or beyond the boundary
           are clamped before atanh to avoid NaN.
        c: curvature parameter (>0). Default 1.0 (unit ball).
        eps: small constant for numerical stability.

    Returns:
        Euclidean embeddings of the same shape as `x`.
    """
    sqrt_c = c ** 0.5
    max_norm = 1.0 / sqrt_c - eps
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps).clamp(max=max_norm)
    factor = torch.atanh(sqrt_c * norm) / (sqrt_c * norm)
    return factor * x
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_hyperbolic_ops.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/hyperbolic_ops.py tests/test_hyperbolic_ops.py
git commit -m "[models] add hyperbolic log_map_origin (Poincaré ball → tangent space)"
```

## Report
- pytest before AND after
- git log --oneline -2

---

## Task 4: HTGN adapter scaffold + construction test (path A)

**Files:**
- Create: `src/models/htgn.py`
- Modify: `tests/test_htgn_smoke.py`

**Skip this task if Task 2 reported BLOCKED — proceed to Task B1 instead.**

- [ ] **Step 1: Add failing construction test**

Append to `tests/test_htgn_smoke.py`:

```python
import torch

from src.models.htgn import HTGN


def test_can_construct_htgn():
    """Adapter constructor succeeds with realistic shapes."""
    m = HTGN(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_layers=2,
        curvature=1.0,
        trainable_curvature=False,
        dropout=0.1,
    )
    assert m is not None
    assert m.node_emb.weight.shape == (50, 64)
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_can_construct_htgn -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/htgn.py` scaffold**

Adjust import path and class name to match Task 2 findings (substitute `<UPSTREAM_MODULE>` and `<UPSTREAM_CLASS>`).

```python
"""Adapter for marlin-codes/HTGN baseline.

Wraps the upstream HTGN encoder as a DynamicLinkPredictor. Projects
Poincaré ball output to Euclidean tangent space via log_map_origin
before the shared MLP decoder. Uses learnable nn.Embedding for input
features (spec §6.6 deviation, same as EvolveGCN-O).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# Upstream submodule path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM = _REPO_ROOT / "third_party" / "HTGN"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

# PyTorch 2.4 compat shims discovered in Task 2 (REPLACE BELOW with actual shims)
# Example placeholder:
if not hasattr(torch, "_six"):
    torch._six = SimpleNamespace(inf=float("inf"), nan=float("nan"))

# Substitute with Task 2 findings
from <UPSTREAM_MODULE> import <UPSTREAM_CLASS> as UpstreamHTGN  # noqa: E402

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP
from src.models.hyperbolic_ops import log_map_origin


class HTGN(DynamicLinkPredictor):
    """HTGN encoder wrapped as a DynamicLinkPredictor.

    Differences from upstream HTGN paper:
        - Adam optimizer (not RAdam) — Hybrid policy.
        - Fixed curvature at 1.0 (paper allows learnable; we keep fixed
          for numerical stability under Adam).
        - Shared MLP decoder (not paper's Fermi-Dirac) for fair
          comparison with GCN_MA and EvolveGCN-O.
        - Symmetric adjacency (same fix as Plan 3a EvolveGCN-O).
        - Learnable nn.Embedding input features (not one-hot identity).
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.curvature = nn.Parameter(
            torch.tensor(float(curvature)), requires_grad=trainable_curvature
        )

        # Learnable node embedding (spec §6.6 deviation, same as EvolveGCN-O)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Upstream HTGN encoder — CONSTRUCTION ARGUMENTS DEPEND ON TASK 2 FINDINGS.
        # Common pattern: UpstreamHTGN(input_dim, hidden_dim, num_layers, manifold=...)
        # Construct here. If upstream requires args namespace, build it like Plan 3a:
        #   upstream_args = SimpleNamespace(...)
        #   self.core = UpstreamHTGN(args=upstream_args, ...)
        self.core = UpstreamHTGN(
            # Fill in based on Task 2 inspection of __init__ signature.
            input_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Shared MLP decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step):
        """Implemented in Task 5. Returns Euclidean embeddings [N, hidden_dim]."""
        raise NotImplementedError("HTGN.forward — implemented in Task 5")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

If Task 2 found that upstream's `__init__` takes additional args (manifold name, dropout, etc.), pass them through. Refer to Task 2's signature dump.

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py -v
```

Expected: 2 passed (import + construction).

If constructor crashes:
- Print the full traceback
- Most likely cause: upstream `__init__` requires args not provided in scaffold
- Fix: add the missing kwargs based on Task 2's signature inspection
- If can't construct after 30 min, escalate: may indicate path A unsuitable → switch to path B

- [ ] **Step 5: Commit**

```bash
git add src/models/htgn.py tests/test_htgn_smoke.py
git commit -m "[models] HTGN adapter scaffold + construction test (path A)"
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- pytest output
- git log --oneline -2
- Any unexpected upstream args added

---

## Task 5: HTGN adapter forward — shape + gradient flow (path A)

**Files:**
- Modify: `src/models/htgn.py`
- Modify: `tests/test_htgn_smoke.py`

**Skip if Task 2 BLOCKED.**

- [ ] **Step 1: Add failing shape + gradient tests**

Append to `tests/test_htgn_smoke.py`:

```python
from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int) -> list[Data]:
    snaps = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, N * 2))
        d = Data(edge_index=ei, num_nodes=N)
        snaps.append(d)
    return snaps


def test_htgn_forward_shape():
    N, T, D = 50, 5, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D), f"expected ({N}, {D}), got {Z.shape}"
    assert torch.isfinite(Z).all(), "HTGN forward produced NaN/Inf"


def test_htgn_gradient_flows():
    N, T, D = 50, 4, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no finite non-zero gradient on any parameter"
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_htgn_forward_shape -v
```

Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `forward` in `src/models/htgn.py`**

Replace the `raise NotImplementedError(...)` line with:

```python
    def forward(self, snapshots, time_step):
        """Run upstream HTGN through snapshots [0..time_step].

        Returns Z^{time_step} ∈ R^{num_nodes × hidden_dim} (Euclidean).
        Hyperbolic embeddings from the upstream encoder are projected to
        the Euclidean tangent space at the origin via log_map_origin.
        """
        device = self.node_emb.weight.device
        N = self.num_nodes
        node_ids = torch.arange(N, device=device)
        node_emb = self.node_emb(node_ids)  # [N, feat_dim]

        A_list, X_list = [], []
        for tau in range(time_step + 1):
            snap = snapshots[tau]
            ei = snap.edge_index.to(device)
            if ei.numel() == 0:
                ei = torch.stack([node_ids, node_ids], dim=0)
            # Symmetrize (Plan 3a EvolveGCN-O fix — bipartite signal preservation)
            ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
            vals = torch.ones(ei_sym.shape[1], device=device)
            A = torch.sparse_coo_tensor(ei_sym, vals, (N, N)).coalesce()
            A_list.append(A)
            X_list.append(node_emb)

        # Upstream forward signature varies — Task 2 should have documented it.
        # COMMON PATTERNS (pick the one matching Task 2 inspection):
        # (a) self.core(A_list, X_list) → hyperbolic embeddings [N, hidden_dim]
        # (b) self.core(A_list[-1], X_list[-1], h_prev) returning (h, ...)
        # (c) self.core(X_list[-1], A_list[-1]) — single-snapshot call (loop here)
        #
        # Default to pattern (a). If forward returns a tuple, unpack:
        #   z_hyp, *_ = self.core(A_list, X_list)
        z_hyp = self.core(A_list, X_list)

        # Project Poincaré ball → Euclidean tangent space at origin
        c = self.curvature.item() if not self.curvature.requires_grad else self.curvature
        z_euc = log_map_origin(z_hyp, c=c)
        return z_euc
```

Adjust the `self.core(...)` call to match the upstream `forward` signature found in Task 2. If the upstream is designed for single-snapshot calls, wrap in a Python loop and maintain hidden state.

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py -v
```

Expected: 4 passed.

If shape mismatch, gradient zero, or NaN:
- Print upstream's actual output shape: `print(z_hyp.shape, z_hyp.norm(dim=-1).max())`
- Norm should be < 1/√c = 1.0 (it lives on Poincaré ball). If norms > 1, upstream returned Euclidean — no log_map needed; skip the projection.
- If gradient zero on `node_emb`: confirm upstream forward goes through `node_emb(node_ids)` — should have, since we built X_list from it.
- If NaN: lower the embedding init scale to 0.001 (smaller than Xavier) and retry.

- [ ] **Step 5: Commit**

```bash
git add src/models/htgn.py tests/test_htgn_smoke.py
git commit -m "[models] HTGN adapter forward + smoke tests (path A)"
```

## Report
- pytest before AND after
- git log --oneline -2

---

## Task 6: Model config + first experiment config

**Files:**
- Create: `configs/models/htgn.yaml`
- Create: `configs/experiments/htgn_collegemsg.yaml`

- [ ] **Step 1: Create `configs/models/htgn.yaml`**

```yaml
name: htgn
feat_dim: 64
hidden_dim: 64
num_layers: 2
curvature: 1.0
trainable_curvature: false
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 200
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

- [ ] **Step 2: Create `configs/experiments/htgn_collegemsg.yaml`**

```yaml
experiment_name: htgn_collegemsg
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 3: Verify YAML parses**

```bash
.venv/bin/python -c "
import yaml
cfg = yaml.safe_load(open('configs/experiments/htgn_collegemsg.yaml'))
mcfg = yaml.safe_load(open('configs/models/htgn.yaml'))
print('Experiment:', cfg)
print('Model:', mcfg)
"
```

Expected: both configs print as dicts with the listed fields.

- [ ] **Step 4: Commit**

```bash
git add configs/models/htgn.yaml configs/experiments/htgn_collegemsg.yaml
git commit -m "[configs] HTGN model config + CollegeMsg experiment"
```

---

## Task 7: CLI dispatch — add `htgn` to `_build_model`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add `htgn` branch to `_build_model`**

Find the existing `_build_model` helper in `scripts/train.py`. It already has `gcn_ma` and `evolvegcn_o` branches. Add a third:

```python
    elif name == "htgn":
        from src.models.htgn import HTGN
        return HTGN(
            num_nodes=graph.num_nodes,
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            curvature=model_cfg["curvature"],
            trainable_curvature=model_cfg["trainable_curvature"],
            dropout=model_cfg["dropout"],
        )
```

Insert just before the existing `raise ValueError(f"Unknown model name: {name!r}")` line.

**Path B note:** if Task 2 escalated and `src/models/htgn/` is a directory instead of a file, change the import to `from src.models.htgn.model import HTGN` (or the appropriate class name from path B Task B4).

- [ ] **Step 2: Verify train.py imports cleanly**

```bash
.venv/bin/python scripts/train.py --help
```

Expected: argparse usage prints without ImportError.

- [ ] **Step 3: Smoke-test GCN_MA still works (regression check)**

```bash
.venv/bin/python scripts/train.py --config configs/experiments/gcn_ma_collegemsg_smoke.yaml 2>&1 | tail -10
```

Expected: JSON record with `"model": "gcn_ma"`, AUC ~0.90.

- [ ] **Step 4: Smoke-test EvolveGCN-O still works**

```bash
.venv/bin/python scripts/train.py --config configs/experiments/evolvegcn_o_collegemsg_smoke.yaml 2>&1 | tail -10
```

Expected: JSON record with `"model": "evolvegcn_o"`, AUC > 0.5.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "[scripts] _build_model: add htgn branch (gcn_ma + evolvegcn_o + htgn)"
```

---

## Task 8: Smoke-train HTGN on CollegeMsg

**Files:**
- Create: `configs/models/htgn_smoke.yaml`
- Create: `configs/experiments/htgn_collegemsg_smoke.yaml`

- [ ] **Step 1: Create smoke configs**

`configs/models/htgn_smoke.yaml`:

```yaml
name: htgn
feat_dim: 32
hidden_dim: 32
num_layers: 2
curvature: 1.0
trainable_curvature: false
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 3
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

`configs/experiments/htgn_collegemsg_smoke.yaml`:

```yaml
experiment_name: htgn_collegemsg_smoke
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/htgn_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_smoke.jsonl
```

- [ ] **Step 2: Run smoke training**

```bash
mkdir -p results/logs
.venv/bin/python scripts/train.py --config configs/experiments/htgn_collegemsg_smoke.yaml 2>&1 | tee "results/logs/htgn_smoke_$(date +%Y%m%d-%H%M%S).log" | tail -25
```

Expected:
- 3 epochs complete without Python errors.
- val_auc per epoch is finite and in (0.0, 1.0), not stuck exactly at 0.5.
- Final JSON record with `"model": "htgn"`, `auc` and `ap` ∈ [0.0, 1.0].
- `results/metrics_smoke.jsonl` has 1 new line.

If val_auc stuck at exactly 0.5: same pathology as Plan 3a's bipartite collapse. Investigate:
1. Verify `Z = m(snaps, time_step=...)` has non-zero std and not all-NaN.
2. Check that upstream forward isn't silently returning a constant.
3. The symmetric adjacency fix should already be in place — confirm in Task 5's forward implementation.

If CUDA OOM: reduce feat_dim and hidden_dim to 16 in the smoke config.

If NaN loss appears mid-epoch: likely log_map clamping is too aggressive OR hyperbolic norm escapes the ball. Investigate by printing `z_hyp.norm(dim=-1).max()` after each snapshot in the forward.

- [ ] **Step 3: Commit smoke configs**

```bash
git add configs/models/htgn_smoke.yaml configs/experiments/htgn_collegemsg_smoke.yaml
git commit -m "[exp] HTGN smoke configs + verified end-to-end on CollegeMsg"
```

## Report
- Per-epoch summary lines (loss, val_auc, val_ap for each of 3 epochs)
- Final JSON record
- git log --oneline -2
- Any concerns

---

## Task 9: Remaining 5 experiment configs

**Files:**
- Create: `configs/experiments/htgn_{bitcoinotc,eut,mooc_actions,lastfm,wikipedia}.yaml`

All 5 follow the same template — only `dataset_config` differs.

- [ ] **Step 1: Create `configs/experiments/htgn_bitcoinotc.yaml`**

```yaml
experiment_name: htgn_bitcoinotc
seed: 42
dataset_config: configs/datasets/bitcoinotc.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 2: Create `configs/experiments/htgn_eut.yaml`**

```yaml
experiment_name: htgn_eut
seed: 42
dataset_config: configs/datasets/eut.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 3: Create `configs/experiments/htgn_mooc_actions.yaml`**

```yaml
experiment_name: htgn_mooc_actions
seed: 42
dataset_config: configs/datasets/mooc_actions.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Create `configs/experiments/htgn_lastfm.yaml`**

```yaml
experiment_name: htgn_lastfm
seed: 42
dataset_config: configs/datasets/lastfm.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 5: Create `configs/experiments/htgn_wikipedia.yaml`**

```yaml
experiment_name: htgn_wikipedia
seed: 42
dataset_config: configs/datasets/wikipedia.yaml
model_config: configs/models/htgn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 6: Verify all 6 configs parse**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    .venv/bin/python -c "import yaml; print('$ds:', yaml.safe_load(open('configs/experiments/htgn_${ds}.yaml'))['experiment_name'])"
done
```

Expected: 6 lines printing each `experiment_name`.

- [ ] **Step 7: Commit**

```bash
git add configs/experiments/htgn_bitcoinotc.yaml \
        configs/experiments/htgn_eut.yaml \
        configs/experiments/htgn_mooc_actions.yaml \
        configs/experiments/htgn_lastfm.yaml \
        configs/experiments/htgn_wikipedia.yaml
git commit -m "[exp] HTGN experiment configs for remaining 5 datasets"
```

---

## Task 10: Full 18-run experiment

The big run. Estimated ~3.5-4 hours wall-clock (HTGN hyperbolic ops ~20-50% slower than EvolveGCN-O).

- [ ] **Step 1: Verify all 6 datasets are cached**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    n_cache=$(ls data/processed/${ds}/*.pt 2>/dev/null | wc -l)
    echo "$ds: ${n_cache} cache file(s)"
done
```

Expected: each dataset shows ≥ 1 cache file (already prepared by Plan 2). If any missing, run the loader smoke from Plan 2 to rebuild.

- [ ] **Step 2: Launch the 18-run sequence**

Order: small → large datasets so any failure surfaces fast.

```bash
mkdir -p results/logs results/configs_runtime
for ds in collegemsg bitcoinotc eut wikipedia lastfm mooc_actions; do
    echo "===== $(date '+%H:%M:%S') ===== HTGN × ${ds} ====="
    scripts/run_seeds.sh "$ds" htgn
done
echo "===== $(date '+%H:%M:%S') ===== ALL DONE ====="
```

Expected total ~3.5-4 hours. If EUT exceeds 2 hours per seed, consider truncated history (limit `time_step` in forward to last 50 snapshots).

If a run crashes:
- NaN loss → reduce curvature initialization to 0.5, retry that dataset.
- CUDA OOM → reduce hidden_dim to 32 in `configs/models/htgn.yaml`, retry only the affected dataset.
- Stuck at val_auc=0.5 → escalate; should have been caught in Task 8 smoke.

- [ ] **Step 3: Verify 18 HTGN records exist**

```bash
.venv/bin/python -c "
import json
n = 0
with open('results/metrics.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['model'] == 'htgn':
            n += 1
print(f'HTGN records: {n}')
"
```

Expected: 18. If less, re-run missing (dataset, seed) pairs individually:

```bash
RUN_CFG="results/configs_runtime/htgn_<DATASET>_seed<SEED>.yaml"
.venv/bin/python scripts/train.py --config "$RUN_CFG"
```

- [ ] **Step 4: Generate 3-model cross-comparison**

```bash
.venv/bin/python scripts/aggregate_results.py --models gcn_ma evolvegcn_o htgn --output results/report/baselines_summary.md
```

Expected: Markdown table with all 3 models × 6 datasets, mean ± std auc and ap.

- [ ] **Step 5: Commit results**

```bash
git add results/metrics.jsonl results/report/baselines_summary.md
git commit -m "[milestone] HTGN reproduced on 6 datasets × 3 seeds"
```

---

## Task 11: Update reproduction-log + tag v0.3b-htgn

**Files:**
- Modify: `docs/reproduction-log.md`

- [ ] **Step 1: Append Plan 3b section**

Append to `docs/reproduction-log.md`:

```markdown

---

## Plan 3b: HTGN baseline integration

### Approach taken

Path: <A or B> — fill based on Task 2 outcome.

If Path A (preferred): vendored [`marlin-codes/HTGN`](https://github.com/marlin-codes/HTGN) at pin `<SHA>`. Adapter in `src/models/htgn.py` (~<N>LOC) wraps the upstream `<CLASS>` encoder, projects Poincaré ball output → Euclidean via `log_map_origin`, feeds shared MLP decoder. `<N_SHIMS>` lines of PyTorch 2.4 compat shims.

If Path B (fallback): kept submodule for citation but reimplemented core ~<N>LOC across `src/models/htgn/{poincare,hgcn_layer,temporal_gru,model}.py`. Reason path A failed: <SPECIFIC FINDINGS FROM TASK 2>.

### Hyperparameter policy (Hybrid)

Shared with GCN_MA / EvolveGCN-O: `hidden_dim=64`, `dropout=0.1`, `lr=1e-3`, `weight_decay=1e-5`, Adam, `epochs=200`, patience 20, `grad_clip_max_norm=5.0`.

HTGN-specific: `num_layers=2`, `curvature=1.0` fixed, learnable embedding `nn.Embedding(N, 64)`.

### Deviations from HTGN paper

1. **Adam (not RAdam Riemannian)** — Hybrid policy. Hyperbolic geometry survives in encoder; only param update step is Euclidean.
2. **Fixed curvature** at 1.0 (paper recommends learnable) — numerical stability under Adam.
3. **Shared MLP decoder** instead of Fermi-Dirac hyperbolic decoder — uniform decoder isolates encoder differences across models.
4. **Symmetric adjacency** — carry-forward from Plan 3a, required for bipartite signal preservation.

### Final results — 6 datasets × 3 seeds

(paste cross-model table from `results/report/baselines_summary.md`)

### Observations

- GCN_MA wins / ties / loses to HTGN on N/6 datasets — fill in based on data
- HTGN's hyperbolic encoding helps most on datasets with hierarchical structure
- Notable gaps vs paper (if any) discussed

### Bugs / surprises caught

(populate as encountered: e.g., NaN at boundary, curvature instability)

### Carry-forwards to Plan 3c

- `_build_model` now dispatches 3 models — same pattern for DyGNN.
- Hyperbolic ops module (`src/models/hyperbolic_ops.py`) is general-purpose, reusable.
- Pattern of "fork → adapter → optional reimplement fallback" is robust for 2019-2021 baseline integrations.
```

Fill in `<A or B>`, `<SHA>`, `<N>LOC`, `<CLASS>`, `<N_SHIMS>`, and the result table paste.

- [ ] **Step 2: Verify reproduction-log renders**

```bash
tail -50 docs/reproduction-log.md
```

Expected: latest section is "Plan 3b: HTGN baseline integration" with all placeholders replaced.

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reproduction-log.md
git commit -m "[docs] record Plan 3b HTGN results"
git tag v0.3b-htgn
git log --oneline -5
git tag
```

Expected: latest commit is the docs commit; `v0.3b-htgn` in tag list.

---

## Done (path A)

After Task 11, HTGN is integrated as a 3rd baseline with 3-model cross-comparison table. Pattern for Plan 3c (DyGNN): submodule → smoke import → adapter scaffold → forward → configs → no further changes to trainer/evaluator/CLI.

---

# Appendix: Path B (fallback) tasks

**Trigger:** Controller invokes these if Task 2 reports BLOCKED.

Path B replaces Tasks 4-5 with B1-B4 (write minimal HTGN core from paper). All other tasks (Task 3 hyperbolic ops + Tasks 6-11 configs/run/docs) remain the same — they only depend on having a working `HTGN` class with the correct interface.

---

## Task B1: Poincaré ball operations module

**Files:**
- Create: `src/models/htgn/__init__.py`
- Create: `src/models/htgn/poincare.py`
- Test: extend `tests/test_hyperbolic_ops.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_hyperbolic_ops.py`:

```python
from src.models.htgn.poincare import exp_map_origin, mobius_add, hyperbolic_distance


def test_exp_map_origin_inverse_of_log_map():
    """exp_map_0(log_map_0(x)) ≈ x for x with small norm."""
    torch.manual_seed(0)
    x = torch.randn(10, 8) * 0.05  # inside ball
    out = exp_map_origin(log_map_origin(x, c=1.0), c=1.0)
    torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-4)


def test_mobius_add_identity():
    """mobius_add(x, 0) == x and mobius_add(0, x) == x on Poincaré ball."""
    torch.manual_seed(0)
    x = torch.randn(5, 4) * 0.05
    zero = torch.zeros(5, 4)
    torch.testing.assert_close(mobius_add(x, zero, c=1.0), x, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mobius_add(zero, x, c=1.0), x, atol=1e-6, rtol=1e-6)


def test_hyperbolic_distance_zero_at_origin():
    """Distance from origin to itself is zero."""
    zero = torch.zeros(1, 4)
    d = hyperbolic_distance(zero, zero, c=1.0)
    torch.testing.assert_close(d, torch.zeros(1), atol=1e-6, rtol=1e-6)


def test_hyperbolic_distance_positive():
    """Distance between distinct points is positive."""
    torch.manual_seed(0)
    x = torch.randn(5, 4) * 0.05
    y = torch.randn(5, 4) * 0.05
    d = hyperbolic_distance(x, y, c=1.0)
    assert (d >= 0).all()
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_hyperbolic_ops.py -v
```

Expected: 4 new tests fail with ImportError.

- [ ] **Step 3: Implement `src/models/htgn/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `src/models/htgn/poincare.py`**

```python
"""Poincaré ball operations for HTGN reimplement (path B fallback).

References:
- Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018
- Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019

Formulas use the Poincaré ball model with curvature c > 0:
    B_c = {x ∈ R^D : sqrt(c) * ||x|| < 1}

Operations:
    log_0(x) = (1/sqrt(c)) * arctanh(sqrt(c) * ||x||) * (x / ||x||)
    exp_0(v) = (1/sqrt(c)) * tanh(sqrt(c) * ||v||) * (v / ||v||)
    x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / (1 + 2c<x,y> + c²||x||²||y||²)
    d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕_c y||)
"""
import torch

# Re-export log_map_origin so callers can import everything from poincare
from src.models.hyperbolic_ops import log_map_origin  # noqa: F401


def exp_map_origin(v: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Tangent space at origin → Poincaré ball.

    exp_0(v) = (1/sqrt(c)) * tanh(sqrt(c) * ||v||) * (v / ||v||)
    """
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    factor = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return factor * v


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Möbius addition on the Poincaré ball."""
    xy_dot = (x * y).sum(dim=-1, keepdim=True)
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy_dot + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
    den = 1 + 2 * c * xy_dot + (c ** 2) * x_norm_sq * y_norm_sq
    return num / den.clamp(min=eps)


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Distance between two points on the Poincaré ball.

    Returns a [...] tensor (last dim of x/y reduced).
    """
    sqrt_c = c ** 0.5
    neg_x = -x
    diff = mobius_add(neg_x, y, c=c, eps=eps)
    diff_norm = diff.norm(dim=-1).clamp(max=1 / sqrt_c - eps)
    return (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)
```

- [ ] **Step 5: Run, verify pass**

```bash
.venv/bin/pytest tests/test_hyperbolic_ops.py -v
```

Expected: 8 passed (4 original + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/models/htgn/__init__.py src/models/htgn/poincare.py tests/test_hyperbolic_ops.py
git commit -m "[models] HTGN path B: Poincaré ball operations"
```

---

## Task B2: HGCN convolution layer

**Files:**
- Create: `src/models/htgn/hgcn_layer.py`
- Test: append to `tests/test_htgn_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_htgn_smoke.py`:

```python
from src.models.htgn.hgcn_layer import HGCNLayer


def test_hgcn_layer_output_shape():
    N, D_in, D_out = 10, 16, 24
    layer = HGCNLayer(in_dim=D_in, out_dim=D_out, c=1.0)
    X = torch.randn(N, D_in) * 0.05  # small enough to be inside ball
    A = torch.eye(N) + 0.1 * torch.rand(N, N)  # dense adjacency-like
    out = layer(X, A)
    assert out.shape == (N, D_out)


def test_hgcn_layer_gradient_flows():
    N, D_in, D_out = 10, 16, 24
    layer = HGCNLayer(in_dim=D_in, out_dim=D_out, c=1.0)
    X = torch.randn(N, D_in, requires_grad=True) * 0.05
    A = torch.eye(N) + 0.1 * torch.rand(N, N)
    out = layer(X, A)
    out.sum().backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )
    assert has_grad
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_hgcn_layer_output_shape -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/models/htgn/hgcn_layer.py`**

```python
"""HGCN convolution layer for HTGN path B reimplementation.

x' = exp_0(σ(W · (Â · log_0(x))))

where W is a Euclidean linear transformation. We log to tangent space,
aggregate + transform in Euclidean, then exp back to hyperbolic.
"""
import torch
from torch import nn

from src.models.htgn.poincare import exp_map_origin, log_map_origin


class HGCNLayer(nn.Module):
    """Single HGCN layer operating on Poincaré ball.

    Steps:
        1. log_map_0(x) — project to tangent space
        2. A @ x_tangent — Euclidean aggregation
        3. W @ aggregated — Euclidean linear
        4. ReLU activation
        5. exp_map_0 — project back to ball
    """

    def __init__(self, in_dim: int, out_dim: int, c: float = 1.0):
        super().__init__()
        self.c = c
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x_tan = log_map_origin(x, c=self.c)
        if A.is_sparse:
            agg = torch.sparse.mm(A, x_tan)
        else:
            agg = A @ x_tan
        out_tan = torch.relu(self.linear(agg))
        return exp_map_origin(out_tan, c=self.c)
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py -v
```

Expected: prior + 2 new tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/htgn/hgcn_layer.py tests/test_htgn_smoke.py
git commit -m "[models] HTGN path B: HGCN convolution layer"
```

---

## Task B3: Temporal GRU weight evolution

**Files:**
- Create: `src/models/htgn/temporal_gru.py`
- Test: append to `tests/test_htgn_smoke.py`

- [ ] **Step 1: Add failing test**

Append:

```python
from src.models.htgn.temporal_gru import TemporalGRUWeight


def test_temporal_gru_preserves_shape():
    in_dim, out_dim = 16, 24
    updater = TemporalGRUWeight(in_dim=in_dim, out_dim=out_dim)
    W = torch.randn(in_dim, out_dim)
    h = updater.init_state(W.device)
    W_next, h_next = updater(W, h)
    assert W_next.shape == (in_dim, out_dim)
    assert h_next.shape == h.shape


def test_temporal_gru_gradients_flow():
    in_dim, out_dim = 8, 12
    updater = TemporalGRUWeight(in_dim=in_dim, out_dim=out_dim)
    W = torch.randn(in_dim, out_dim, requires_grad=True)
    h = updater.init_state(W.device)
    W_next, _ = updater(W, h)
    W_next.sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_temporal_gru_preserves_shape -v
```

- [ ] **Step 3: Implement `src/models/htgn/temporal_gru.py`**

```python
"""Temporal weight evolution via GRU for HTGN path B.

Treats the [in_dim, out_dim] weight matrix as a flat vector of size
in_dim * out_dim, runs it through a nn.GRUCell, reshapes back.

Same pattern as GCN_MA's LSTMWeightUpdater but with GRU instead.
"""
import torch
from torch import nn


class TemporalGRUWeight(nn.Module):
    """Recurrent weight evolver: W^t = GRU(W^{t-1})."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = in_dim * out_dim
        self.cell = nn.GRUCell(input_size=self.dim, hidden_size=self.dim)

    def init_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, self.dim, device=device)

    def forward(self, W: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = W.reshape(1, self.dim)
        h_next = self.cell(flat, h)
        W_next = h_next.reshape(self.in_dim, self.out_dim)
        return W_next, h_next
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/models/htgn/temporal_gru.py tests/test_htgn_smoke.py
git commit -m "[models] HTGN path B: temporal GRU weight evolver"
```

---

## Task B4: HTGN composition (path B model.py)

**Files:**
- Create: `src/models/htgn/model.py`
- Test: append to `tests/test_htgn_smoke.py`

- [ ] **Step 1: Add failing test**

Append:

```python
from torch_geometric.data import Data

from src.models.htgn.model import HTGN


def _make_dummy_snapshots(N: int, T: int) -> list[Data]:
    snaps = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, N * 2))
        d = Data(edge_index=ei, num_nodes=N)
        snaps.append(d)
    return snaps


def test_htgn_pathb_construct():
    m = HTGN(
        num_nodes=50, feat_dim=64, hidden_dim=64, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    assert m is not None


def test_htgn_pathb_forward_shape():
    N, T, D = 50, 5, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()


def test_htgn_pathb_gradient_flows():
    N, T, D = 50, 4, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    Z.sum().backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py::test_htgn_pathb_construct -v
```

- [ ] **Step 3: Implement `src/models/htgn/model.py`**

```python
"""HTGN composition (path B fallback reimplement).

Architecture:
    1. Node embedding (Xavier, projected to ball via exp_map_0)
    2. For each timestep t:
       a. Evolve weights via TemporalGRUWeight
       b. HGCN layer 1: x → x'
       c. HGCN layer 2: x' → x''  (using the same evolved weights or per-layer)
    3. Output: x''[T] in hyperbolic space → log_map_0 → Euclidean
    4. Decoder: shared LinkDecoderMLP
"""
import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP
from src.models.hyperbolic_ops import log_map_origin
from src.models.htgn.hgcn_layer import HGCNLayer
from src.models.htgn.poincare import exp_map_origin
from src.models.htgn.temporal_gru import TemporalGRUWeight


class HTGN(DynamicLinkPredictor):
    """HTGN reimplemented from paper (path B fallback)."""

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        if num_layers != 2:
            raise ValueError(f"HTGN path B uses 2 HGCN layers; got num_layers={num_layers}")
        self.curvature = nn.Parameter(
            torch.tensor(float(curvature)), requires_grad=trainable_curvature
        )

        # Learnable node embedding in Euclidean (projected to ball at start of forward)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # HGCN layers (in→hidden, hidden→hidden)
        self.hgcn1 = HGCNLayer(in_dim=feat_dim, out_dim=hidden_dim, c=curvature)
        self.hgcn2 = HGCNLayer(in_dim=hidden_dim, out_dim=hidden_dim, c=curvature)

        # Temporal weight evolution for each layer
        self.tgru1 = TemporalGRUWeight(in_dim=feat_dim, out_dim=hidden_dim)
        self.tgru2 = TemporalGRUWeight(in_dim=hidden_dim, out_dim=hidden_dim)

        # Shared decoder (Euclidean)
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step):
        device = self.node_emb.weight.device
        N = self.num_nodes
        c = self.curvature.item() if not self.curvature.requires_grad else self.curvature
        node_ids = torch.arange(N, device=device)

        # Start in Euclidean space — projected to ball at first HGCN call
        node_emb = self.node_emb(node_ids)  # [N, feat_dim], Euclidean

        # Project to Poincaré ball
        x = exp_map_origin(node_emb, c=c)  # [N, feat_dim], hyperbolic

        # Temporal evolution states
        h1 = self.tgru1.init_state(device)
        h2 = self.tgru2.init_state(device)

        for tau in range(time_step + 1):
            snap = snapshots[tau]
            ei = snap.edge_index.to(device)
            if ei.numel() == 0:
                ei = torch.stack([node_ids, node_ids], dim=0)
            # Symmetrize (Plan 3a fix)
            ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
            vals = torch.ones(ei_sym.shape[1], device=device)
            A = torch.sparse_coo_tensor(ei_sym, vals, (N, N)).coalesce()

            # Evolve layer weights
            W1, h1 = self.tgru1(self.hgcn1.linear.weight.t().detach(), h1)
            W2, h2 = self.tgru2(self.hgcn2.linear.weight.t().detach(), h2)
            # Inject evolved weights into the layers (transposed back)
            self.hgcn1.linear.weight.data.copy_(W1.t())
            self.hgcn2.linear.weight.data.copy_(W2.t())

            # HGCN propagation: hyperbolic → hyperbolic
            x = self.hgcn1(x, A)
            x = self.hgcn2(x, A)

        # Final: project to Euclidean tangent space
        z_euc = log_map_origin(x, c=c)
        return z_euc

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

Note: the weight injection via `.data.copy_()` is a pragmatic shortcut. A cleaner design would pass `W` explicitly to HGCNLayer's forward, but this keeps each module focused. Document this in reproduction-log if Path B taken.

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_htgn_smoke.py -v
```

Expected: all path B smoke tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/htgn/model.py tests/test_htgn_smoke.py
git commit -m "[models] HTGN path B: full composition (poincare + hgcn + temporal-gru)"
```

After Task B4, resume mainline at Task 6 (configs). Note: Task 7 needs adjustment — change `from src.models.htgn import HTGN` to `from src.models.htgn.model import HTGN`.
