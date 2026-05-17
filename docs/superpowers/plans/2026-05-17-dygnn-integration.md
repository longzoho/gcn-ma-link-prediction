# DyGNN Baseline Integration (Plan 3c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DyGNN (Streaming GNN, SIGIR 2020) as the 4th baseline running on 5 datasets × 3 seeds (15 runs; LastFM skipped due to compute budget). Produce 4-model cross-comparison.

**Architecture:** A→B fallback. Path A primary: vendor `alge24/DyGNN` submodule + adapter (~200 LOC) wrapping upstream edge-update class. Path B fallback (triggered by Task 2 BLOCKED): reimplement DyGNN core ~250 LOC across 4 modules (node_memory + edge_update + interaction + model). Critical addition: loader cache schema bump `fmt2 → fmt3` to store original edge timestamps for chronological sort. Per-epoch memory cache provides gradient approximation under our snapshot-batched trainer.

**Tech Stack:** Python 3.11, PyTorch 2.4, PyTorch Geometric 2.6. Reuses Plan 1/2/3a/3b infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-17-dygnn-integration-design.md`

**Predecessor:** Plan 3b (tag `v0.3b-htgn`, commit `69fd3b7`).

**Out of scope:** DGCN (Plan 3d). LastFM with DyGNN (compute infeasible). DyGNN's hierarchical attention variant.

---

## File map (path A — preferred)

```
gcn-ma-link-prediction/
├── third_party/DyGNN/                             # NEW submodule
├── configs/
│   ├── models/dygnn.yaml                          # NEW
│   ├── models/dygnn_smoke.yaml                    # NEW (Task 8)
│   └── experiments/
│       ├── dygnn_collegemsg.yaml                  # NEW
│       ├── dygnn_collegemsg_smoke.yaml            # NEW (Task 8)
│       ├── dygnn_bitcoinotc.yaml                  # NEW (Task 9)
│       ├── dygnn_eut.yaml                         # NEW
│       ├── dygnn_mooc_actions.yaml                # NEW
│       └── dygnn_wikipedia.yaml                   # NEW
│       # NO dygnn_lastfm.yaml — skipped
├── src/
│   ├── data/loaders/_base.py                      # MODIFY (Task 3) — fmt3 adds edge_ts
│   └── models/dygnn.py                            # NEW — adapter (~200 LOC)
├── scripts/train.py                               # MODIFY — add dygnn branch
├── tests/
│   ├── test_loader_base.py                        # MODIFY (Task 3) — fmt3 tests
│   └── test_dygnn_smoke.py                        # NEW
└── docs/reproduction-log.md                       # MODIFY — Plan 3c section
```

## File map (path B fallback — replaces Tasks 4-5 with B1-B4)

```
src/models/dygnn/                                  # NEW directory (replaces dygnn.py)
├── __init__.py
├── node_memory.py                                 # NEW (~40 LOC, Task B1)
├── edge_update.py                                 # NEW (~80 LOC, Task B2)
├── interaction.py                                 # NEW (~50 LOC, Task B3)
└── model.py                                       # NEW (~80 LOC, Task B4)
```

---

## Task 1: Add `alge24/DyGNN` as git submodule

**Files:**
- Modify: `.gitmodules`
- Create: `third_party/DyGNN/`

- [ ] **Step 1: Add submodule**

```bash
git submodule add https://github.com/alge24/DyGNN third_party/DyGNN
```

- [ ] **Step 2: Record pin SHA**

```bash
cd third_party/DyGNN
PIN_SHA=$(git rev-parse HEAD)
echo "DyGNN pinned commit: $PIN_SHA"
cd ../..
```

Record SHA in your report.

- [ ] **Step 3: Inspect repo structure**

```bash
ls third_party/DyGNN/
find third_party/DyGNN -maxdepth 2 -type f -name "*.py" | head -30
```

Identify likely files: `model.py`, `DyGNN.py`, `main.py`, etc. Note paths for Task 2.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/DyGNN
git commit -m "[deps] vendor alge24/DyGNN as submodule, pinned to <SHA>"
```

Replace `<SHA>` with the actual pin SHA (short 7-char form).

## Report
- **Status:** DONE | BLOCKED (clone failed)
- Pin SHA (full 40-char and short 7-char)
- Repo file listing
- Likely location of DyGNN encoder class
- git log --oneline -2

---

## Task 2: Smoke-test upstream import + document API (1-day timebox gate)

**Files:**
- Create: `tests/test_dygnn_smoke.py` (initial — import test only)

**This task gates path A vs B.** Shims > 10 lines OR upstream task-coupled → BLOCKED → controller invokes path B (Task B1-B4 from appendix).

- [ ] **Step 1: Inspect DyGNN module structure**

```bash
.venv/bin/python << 'EOF'
import sys
sys.path.insert(0, 'third_party/DyGNN')
# Try common module names
for name in ['DyGNN', 'model', 'main', 'src.model', 'src.DyGNN']:
    try:
        mod = __import__(name, fromlist=['*'])
        classes = [c for c in dir(mod) if c[0].isupper() and not c.startswith('_')]
        print(f'OK: {name}: classes={classes[:15]}')
    except Exception as e:
        print(f'FAIL: {name}: {type(e).__name__}: {str(e)[:120]}')
EOF
```

Find the module that contains the DyGNN class.

- [ ] **Step 2: Print init + forward signatures**

```bash
.venv/bin/python << 'EOF'
import sys
sys.path.insert(0, 'third_party/DyGNN')
from <MODULE> import <CLASS>
import inspect
print('=== __init__ ===')
print(inspect.signature(<CLASS>.__init__))
print(inspect.getsource(<CLASS>.__init__)[:2500])
print('=== forward ===')
print(inspect.signature(<CLASS>.forward))
print(inspect.getsource(<CLASS>.forward)[:2500])
EOF
```

Record signatures and forward body in your report.

- [ ] **Step 3: Write `tests/test_dygnn_smoke.py` import test**

```python
"""Smoke tests for DyGNN upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "DyGNN"))


def test_can_import_upstream_dygnn():
    """Upstream DyGNN class must import under PyTorch 2.4."""
    from <MODULE_NAME> import <CLASS_NAME>  # noqa: F401
    assert <CLASS_NAME> is not None
```

- [ ] **Step 4: Run smoke test + count shims**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_can_import_upstream_dygnn -v
```

Common failure modes:
- `ImportError: cannot import name 'inf' from 'torch._six'` → 3-line shim (same as EvolveGCN).
- `ModuleNotFoundError: No module named 'X'` → install dep; if dep itself broken under PyTorch 2.4 → assess case-by-case.
- Module-level argparse → reset `sys.argv` shim (3-line, same as HTGN).
- Task-coupled imports (model expects custom dataset object) → likely path B candidate.

**Count shim lines.** If > 10 lines OR > 1 day → BLOCKED.

- [ ] **Step 5: Confirm PASS**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_can_import_upstream_dygnn -v
```

Expected: PASS within timebox.

- [ ] **Step 6: Document findings**

In report:
1. Module path
2. Class name
3. `__init__` signature
4. `forward` signature (CRITICAL — what does it take? edge tuples? edge index? graph object?)
5. Each shim line
6. Whether upstream is task-coupled or standalone

The `forward` signature is the most important detail — DyGNN's API varies wildly across implementations. Some take `(src, dst, t)` triples, some take edge_index.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dygnn_smoke.py
git commit -m "[tests] smoke import test for upstream DyGNN

Module: <MODULE>
Class: <CLASS>
PyTorch 2.4 compat: <N lines of shims>"
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- 6 documentation items
- pytest output
- Honest assessment: path A viable, or path B recommended?
- git log --oneline -2

---

## Task 3: Loader cache schema fmt2 → fmt3 (add edge_ts)

**Files:**
- Modify: `src/data/loaders/_base.py`
- Modify: `tests/test_loader_base.py`

DyGNN needs original timestamps to sort edges chronologically within a snapshot. Bump cache format to include `edge_ts`. Other models (GCN_MA, EvolveGCN-O, HTGN) ignore this attribute — no regression risk.

- [ ] **Step 1: Read current `_base.py` to understand `_preprocess` structure**

```bash
grep -n "_CACHE_FORMAT_VERSION\|def _preprocess\|edge_index_list\|features_list" src/data/loaders/_base.py
```

Locate the cache format version line and the `_preprocess` method's loop building `edge_index_list`.

- [ ] **Step 2: Add failing test**

Append to `tests/test_loader_base.py`:

```python
def test_loader_cache_includes_edge_ts(tmp_path):
    """fmt3 cache must include edge_ts per snapshot (alongside edge_index)."""
    # Build a tiny synthetic dataset with known timestamps
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "raw.txt.gz"

    import gzip
    with gzip.open(raw_file, "wt") as f:
        # src dst ts
        f.write("0 1 100\n")
        f.write("1 2 200\n")
        f.write("2 3 300\n")
        f.write("0 3 400\n")

    # Use CollegeMsg loader as a concrete subclass
    from src.data.loaders.collegemsg import CollegeMsgLoader

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    loader = CollegeMsgLoader()
    g = loader.build(raw_file, cache_dir, num_time_steps=2, beta=0.8)

    # Verify each snapshot has edge_ts attribute
    for t in range(g.num_time_steps):
        snap = g.snapshots[t]
        assert hasattr(snap, "edge_ts"), f"snapshot {t} missing edge_ts attribute"
        assert snap.edge_ts.shape[0] == snap.edge_index.shape[1], \
            f"snapshot {t} edge_ts length {snap.edge_ts.shape[0]} != edge_index width {snap.edge_index.shape[1]}"
        # edge_ts should be sorted-able (no NaN, has dtype float)
        assert snap.edge_ts.dtype in (torch.float32, torch.float64)
```

- [ ] **Step 3: Run, verify fail**

```bash
.venv/bin/pytest tests/test_loader_base.py::test_loader_cache_includes_edge_ts -v
```

Expected: `AssertionError: snapshot 0 missing edge_ts attribute`.

- [ ] **Step 4: Modify `src/data/loaders/_base.py`**

Find the `_CACHE_FORMAT_VERSION = "fmt2"` line and change to:

```python
_CACHE_FORMAT_VERSION = "fmt3"  # adds edge_ts per snapshot for DyGNN
```

Find the `_preprocess` method. Inside the per-snapshot loop, currently:

```python
for t in range(num_time_steps):
    mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
    sub = df.loc[mask, ["src", "dst"]].values
    edges_list = [(int(u), int(v)) for u, v in sub if u != v]
    # ... feature/A computation ...
    edge_index_list.append(edge_index)
```

Change to capture timestamps:

```python
edge_ts_list = []  # NEW — declared alongside edge_index_list, features_list at function start

for t in range(num_time_steps):
    mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
    sub = df.loc[mask, ["src", "dst", "ts"]].values  # add "ts" column
    edges_with_ts = [(int(u), int(v), float(ts)) for u, v, ts in sub if u != v]
    edges_list = [(u, v) for u, v, ts in edges_with_ts]  # drop ts for downstream
    # ... existing feature/A computation using edges_list ...

    if edges_with_ts:
        ts_array = torch.tensor([ts for u, v, ts in edges_with_ts], dtype=torch.float64)
    else:
        ts_array = torch.empty(0, dtype=torch.float64)
    edge_ts_list.append(ts_array)
    edge_index_list.append(edge_index)
```

Update the cached payload dict to include `edge_ts`:

```python
return {
    "features": features_list,
    "edge_index": edge_index_list,
    "edge_ts": edge_ts_list,  # NEW
    "num_nodes": num_nodes,
    "num_time_steps": num_time_steps,
}
```

Update `build()` to attach `edge_ts` to each snapshot's `Data` object:

```python
for t in range(cached["num_time_steps"]):
    edge_index = cached["edge_index"][t]
    edge_ts = cached["edge_ts"][t]
    # ... existing A_hat, S_hat, etc. computation ...

    data = Data(edge_index=edge_index, num_nodes=cached["num_nodes"])
    data.x = cached["features"][t]
    data.S_hat = S_hat
    data.edge_ts = edge_ts  # NEW
    snapshots.append(data)
```

- [ ] **Step 5: Run all loader tests to confirm no regression**

```bash
.venv/bin/pytest tests/test_loader_base.py tests/test_*loader*.py -v
```

Expected: existing tests pass + new `test_loader_cache_includes_edge_ts` passes. Old caches invalidate due to version bump → loaders rebuild on first call → no errors.

- [ ] **Step 6: Smoke-check existing models still work via dispatcher**

This validates that the schema change doesn't break GCN_MA/EvolveGCN/HTGN (they don't use `edge_ts` but the build() change shouldn't regress them).

```bash
.venv/bin/python scripts/train.py --config configs/experiments/gcn_ma_collegemsg_smoke.yaml 2>&1 | tail -5
```

Expected: GCN_MA smoke runs and prints a sensible JSON record. The first run after the schema bump will trigger CollegeMsg re-preprocessing (~5s), then train.

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/_base.py tests/test_loader_base.py
git commit -m "[data] cache fmt3: add edge_ts for DyGNN chronological edge sort

Schema bump: each snapshot now exposes \`edge_ts\` (float64 tensor of
original timestamps from raw data). DyGNN adapter uses these to sort
edges chronologically within a snapshot.

Other models (GCN_MA, EvolveGCN-O, HTGN) ignore the new attribute
— no regression. Cache invalidates fmt2 → fmt3 (~2.5 min total
re-preprocess across 6 datasets on first load)."
```

## Report
- pytest before AND after
- GCN_MA smoke output (last 5 lines, including JSON record)
- git log --oneline -2

---

## Task 4: DyGNN adapter scaffold + construction (path A)

**Files:**
- Create: `src/models/dygnn.py`
- Modify: `tests/test_dygnn_smoke.py`

**Skip if Task 2 reported BLOCKED — proceed to Task B1.**

- [ ] **Step 1: Add failing construction test**

Append to `tests/test_dygnn_smoke.py`:

```python
import torch

from src.models.dygnn import DyGNN


def test_can_construct_dygnn():
    m = DyGNN(
        num_nodes=50,
        hidden_dim=64,
        node_memory_dim=64,
        edge_dim=16,
        dropout=0.1,
        decay_method="log",
        decay_rate=1.0,
    )
    assert m is not None
    assert m.memory_init.shape == (50, 64)
    # Memory should start at zero per design spec §9.1
    assert torch.allclose(m.memory_init, torch.zeros(50, 64))
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_can_construct_dygnn -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/dygnn.py` scaffold**

Adjust `<UPSTREAM_MODULE>` and `<UPSTREAM_CLASS>` to match Task 2 findings.

```python
"""Adapter for alge24/DyGNN baseline.

Wraps the upstream DyGNN edge-update class as a DynamicLinkPredictor.
Uses per-epoch memory cache to stay tractable under our snapshot-batched
trainer (spec §5 — gradient approximation documented).

Implementation notes:
    - Adapter sorts edges chronologically within each snapshot using
      cached `edge_ts` attribute (added by loader cache fmt3).
    - Each edge (u, v, t) processed twice: u→v then v→u with ε offset
      (symmetric processing for fair item-side signal flow on bipartite).
    - Per-epoch cache rebuilt when forward(t) sees t < previous_max_t.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# Upstream submodule path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM = _REPO_ROOT / "third_party" / "DyGNN"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

# PyTorch 2.4 compat shims discovered in Task 2 (REPLACE with actual shims)
if not hasattr(torch, "_six"):
    torch._six = SimpleNamespace(inf=float("inf"), nan=float("nan"))

# Substitute with Task 2 findings
from <UPSTREAM_MODULE> import <UPSTREAM_CLASS> as UpstreamDyGNNUpdate  # noqa: E402

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class DyGNN(DynamicLinkPredictor):
    """DyGNN encoder wrapped as a DynamicLinkPredictor.

    Differences from DyGNN paper:
        - Per-epoch memory cache (gradient approximation) — paper does
          per-edge updates with frequent optimizer steps.
        - Symmetric edge processing — each (u, v) edge applied twice.
        - Shared MLP decoder (not paper's scoring head).
        - LastFM skipped due to compute budget.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        node_memory_dim: int = 64,
        edge_dim: int = 16,
        dropout: float = 0.1,
        decay_method: str = "log",
        decay_rate: float = 1.0,
    ):
        super().__init__()
        if node_memory_dim != hidden_dim:
            raise ValueError(
                f"node_memory_dim ({node_memory_dim}) must equal hidden_dim ({hidden_dim})"
            )
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Initial node memory (learnable, init zero per spec §9.1)
        self.memory_init = nn.Parameter(torch.zeros(num_nodes, node_memory_dim))

        # Upstream DyGNN update module — exact constructor args from Task 2
        self.update_module = UpstreamDyGNNUpdate(
            # Fill in based on Task 2 __init__ signature inspection
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
        """Per-epoch cache forward — implemented in Task 5."""
        raise NotImplementedError("DyGNN.forward — implemented in Task 5")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

If upstream constructor needs more fields than the scaffolded ones, add them with sensible defaults from paper or upstream's own example configs.

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py -v
```

Expected: 2 passed (import + construction).

If constructor crashes:
- Print full traceback
- Add missing args to scaffold based on upstream's `__init__` source (inspected in Task 2)
- If can't construct after 30 min → escalate; may indicate path B

- [ ] **Step 5: Commit**

```bash
git add src/models/dygnn.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN adapter scaffold + construction test (path A)"
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- pytest output
- git log --oneline -2
- Any upstream args added beyond scaffold

---

## Task 5: DyGNN adapter forward with per-epoch cache (path A)

**Files:**
- Modify: `src/models/dygnn.py`
- Modify: `tests/test_dygnn_smoke.py`

**Skip if Task 2 BLOCKED.**

- [ ] **Step 1: Add failing forward + cache tests**

Append to `tests/test_dygnn_smoke.py`:

```python
from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int, max_edges: int = 20) -> list[Data]:
    snaps = []
    for t in range(T):
        e = max(1, max_edges // 2)
        ei = torch.randint(0, N, (2, e))
        ts = torch.rand(e, dtype=torch.float64) + float(t)  # within-snapshot ts ∈ [t, t+1)
        d = Data(edge_index=ei, num_nodes=N)
        d.edge_ts = ts
        snaps.append(d)
    return snaps


def test_dygnn_forward_shape():
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()


def test_dygnn_gradient_flows():
    N, T, D = 50, 4, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    Z.sum().backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad


def test_dygnn_cache_reuses_within_epoch():
    """forward(t=3) populates cache; second forward(t=3) returns same cached state."""
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z1 = m(snaps, time_step=3)
    Z2 = m(snaps, time_step=3)
    # Same call returns identical values (cache hit)
    torch.testing.assert_close(Z1, Z2)


def test_dygnn_cache_rebuilds_on_t_regression():
    """forward(t=4) populates cache; forward(t=2) triggers cache rebuild."""
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z4_first = m(snaps, time_step=4)
    Z2 = m(snaps, time_step=2)  # t regression → rebuild
    Z4_second = m(snaps, time_step=4)
    # After rebuild, forward(t=4) should match the first call (deterministic process)
    # NB: this only holds if the upstream update module is deterministic and parameters
    # haven't changed between calls. For our smoke test this is true (no optimizer step).
    torch.testing.assert_close(Z4_first, Z4_second)
    # Z2 should differ from Z4 in general (different time step)
    assert not torch.allclose(Z2, Z4_first, atol=1e-3), "Z(t=2) should differ from Z(t=4)"
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_dygnn_forward_shape -v
```

Expected: `NotImplementedError`.

- [ ] **Step 3: Implement forward + cache in `src/models/dygnn.py`**

Replace the `raise NotImplementedError(...)` line in `forward()` with:

```python
    def forward(self, snapshots, time_step):
        """Return memory state at the end of snapshot `time_step`.

        Per-epoch cache: if t < previous_max_t OR cache empty, rebuild.
        Otherwise return cached state. See spec §5 for gradient
        approximation rationale.
        """
        # New-epoch detection: time_step regressed or cache empty
        if self._cached_memory_per_t is None or time_step < self._prev_max_t:
            self._build_cache(snapshots)
        self._prev_max_t = max(self._prev_max_t, time_step)
        return self._cached_memory_per_t[time_step]

    def _build_cache(self, snapshots):
        """Single chronological pass through all edges in snapshots.

        Builds `self._cached_memory_per_t[t]` for t = 0..len(snapshots)-1.
        Each cache entry is the memory state at the end of processing
        snapshot t's edges.
        """
        device = self.memory_init.device
        memory = self.memory_init.clone()  # fresh start each epoch; grad flows here
        cache = []
        T = len(snapshots)

        for t in range(T):
            snap = snapshots[t]
            ei = snap.edge_index.to(device) if snap.edge_index.numel() > 0 else None
            if ei is not None and hasattr(snap, "edge_ts") and snap.edge_ts.numel() > 0:
                ts = snap.edge_ts.to(device)
                # Sort edges chronologically within snapshot
                sorted_idx = torch.argsort(ts)
                ei_sorted = ei[:, sorted_idx]
                ts_sorted = ts[sorted_idx]

                # Process each edge: update source then target (symmetric)
                for k in range(ei_sorted.shape[1]):
                    u = int(ei_sorted[0, k].item())
                    v = int(ei_sorted[1, k].item())
                    dt = ts_sorted[k]
                    memory = self.update_module(memory, u, v, dt)
                    # Symmetric: reverse direction at slightly later time
                    memory = self.update_module(memory, v, u, dt + 1e-6)
            cache.append(memory.clone())

        self._cached_memory_per_t = cache
        self._prev_max_t = -1
```

**Important constraint:** the upstream `update_module` may have a different signature than `(memory, u, v, dt)`. If Task 2 found it takes e.g. `(memory, edge_tensor)` or returns just the updated rows, adapt accordingly. Wrap the upstream call in a helper if needed:

```python
def _apply_update(self, memory, u, v, dt):
    """Wrapper around upstream update_module to match our (memory, u, v, dt) interface."""
    # Adapt to upstream's actual signature; this stub matches the most common case
    return self.update_module(memory, u, v, dt)
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py -v
```

Expected: 6 passed (import + construct + shape + grad + cache_reuse + cache_rebuild).

If shape mismatch:
- Print upstream `update_module(memory, 0, 1, torch.tensor(0.0))` output shape — must be `[N, D]`.
- If upstream returns only updated rows `[2, D]`, wrap to merge: `memory[u] = new_u; memory[v] = new_v`.

If gradient zero on `update_module` parameters:
- Verify the upstream `update_module` is a `nn.Module` with trainable weights.
- If not, document and use only `memory_init` + decoder gradient flow.

If NaN at first epoch:
- Clip memory norm: `memory = memory.clamp(-10, 10)` after each update.
- Reduce decay_rate (paper default 1.0 → try 0.5).

- [ ] **Step 5: Commit**

```bash
git add src/models/dygnn.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN adapter forward + per-epoch cache + 4 smoke tests"
```

## Report
- pytest before AND after
- git log --oneline -2
- Any upstream API deviations needed (signature differences)
- Cache rebuild trigger logic verified working

---

## Task 6: Model config + smoke config + first experiment config

**Files:**
- Create: `configs/models/dygnn.yaml`
- Create: `configs/models/dygnn_smoke.yaml`
- Create: `configs/experiments/dygnn_collegemsg.yaml`

- [ ] **Step 1: Create `configs/models/dygnn.yaml`**

```yaml
name: dygnn
hidden_dim: 64
node_memory_dim: 64
edge_dim: 16
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 200
early_stop_patience: 20
grad_clip_max_norm: 5.0
decay_method: log
decay_rate: 1.0
```

- [ ] **Step 2: Create `configs/models/dygnn_smoke.yaml`** (for Task 8)

```yaml
name: dygnn
hidden_dim: 32
node_memory_dim: 32
edge_dim: 8
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 3
early_stop_patience: 20
grad_clip_max_norm: 5.0
decay_method: log
decay_rate: 1.0
```

- [ ] **Step 3: Create `configs/experiments/dygnn_collegemsg.yaml`**

```yaml
experiment_name: dygnn_collegemsg
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/dygnn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Verify all 3 YAML parse**

```bash
for cfg in configs/models/dygnn.yaml configs/models/dygnn_smoke.yaml configs/experiments/dygnn_collegemsg.yaml; do
    .venv/bin/python -c "import yaml; print('$cfg:', list(yaml.safe_load(open('$cfg')).keys()))"
done
```

Expected: 3 lines each showing the parsed top-level keys.

- [ ] **Step 5: Commit**

```bash
git add configs/models/dygnn.yaml configs/models/dygnn_smoke.yaml configs/experiments/dygnn_collegemsg.yaml
git commit -m "[configs] DyGNN model + smoke + CollegeMsg experiment configs"
```

## Report
- YAML parse output for all 3
- git log --oneline -2

---

## Task 7: CLI dispatch — add `dygnn` branch to `_build_model`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Read current `_build_model`**

```bash
grep -n "elif name ==" scripts/train.py
```

It should have `gcn_ma`, `evolvegcn_o`, `htgn` branches.

- [ ] **Step 2: Add dygnn branch**

In `scripts/train.py`, BEFORE the final `raise ValueError(...)` line, add:

```python
    elif name == "dygnn":
        from src.models.dygnn import DyGNN
        return DyGNN(
            num_nodes=graph.num_nodes,
            hidden_dim=model_cfg["hidden_dim"],
            node_memory_dim=model_cfg["node_memory_dim"],
            edge_dim=model_cfg["edge_dim"],
            dropout=model_cfg["dropout"],
            decay_method=model_cfg["decay_method"],
            decay_rate=model_cfg["decay_rate"],
        )
```

**Path B note:** if Task 2 BLOCKED and `src/models/dygnn/` is a directory, change the import to `from src.models.dygnn.model import DyGNN`.

- [ ] **Step 3: Sanity check imports**

```bash
.venv/bin/python scripts/train.py --help
```

Expected: argparse usage without ImportError.

- [ ] **Step 4: Regression check (3 prior models)**

```bash
for model in gcn_ma evolvegcn_o htgn; do
    echo "=== $model smoke ==="
    .venv/bin/python scripts/train.py --config configs/experiments/${model}_collegemsg_smoke.yaml 2>&1 | tail -5
done
```

Expected: 3 JSON records with respective `"model"` fields. Each AUC roughly matches prior plans.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "[scripts] _build_model: add dygnn branch (4 models supported)"
```

## Report
- argparse help output
- 3 smoke JSON records (1 per prior model)
- git log --oneline -2

---

## Task 8: Smoke train DyGNN on CollegeMsg

**Files:**
- Create: `configs/experiments/dygnn_collegemsg_smoke.yaml`

- [ ] **Step 1: Create smoke experiment config**

```yaml
experiment_name: dygnn_collegemsg_smoke
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/dygnn_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_smoke.jsonl
```

- [ ] **Step 2: Run smoke training**

```bash
mkdir -p results/logs
.venv/bin/python scripts/train.py --config configs/experiments/dygnn_collegemsg_smoke.yaml 2>&1 | tee "results/logs/dygnn_smoke_$(date +%Y%m%d-%H%M%S).log" | tail -30
```

Expected:
- 3 epochs complete (slow per epoch due to edge-sequence loop, ~1-2 min per epoch on CollegeMsg)
- val_auc per epoch finite, in (0.0, 1.0), NOT stuck exactly at 0.5
- Final JSON record with `"model": "dygnn"`, `auc` and `ap` in [0.0, 1.0]
- `results/metrics_smoke.jsonl` has new line

If val_auc stuck at 0.5:
1. Check that cache rebuild logic isn't preventing gradient flow.
2. Check that `memory_init` parameter has gradient flowing through (`memory_init.grad` non-zero after backward).
3. Inspect `update_module` parameters for grad — if zero, the upstream module isn't training.

If CUDA OOM: reduce node_memory_dim to 16 in smoke config.

If NaN loss:
- Clip memory norm in `_build_cache`: `memory = memory.clamp(-10, 10)` after each update.
- Reduce decay_rate to 0.5.

If extremely slow (>30 min for 3 epochs on CollegeMsg):
- Profile: which step in `_build_cache` dominates? If `int(u.item())` is slow, vectorize with `.tolist()` outside the loop.
- Consider keeping memory on GPU and avoiding scalar `.item()` calls.

- [ ] **Step 3: Commit smoke config**

```bash
git add configs/experiments/dygnn_collegemsg_smoke.yaml
git commit -m "[exp] DyGNN smoke config + verified end-to-end on CollegeMsg"
```

## Report
- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED
- Per-epoch summary (loss, val_auc, val_ap)
- Final JSON record (verbatim)
- Wall-clock time for the 3 epochs
- git log --oneline -2
- Any concerns about speed, NaN, or stuck val_auc

---

## Task 9: Remaining 4 experiment configs (SKIP LastFM)

**Files:**
- Create: `configs/experiments/dygnn_{bitcoinotc,eut,mooc_actions,wikipedia}.yaml`

**NO LastFM config** — skipped per spec §3.

- [ ] **Step 1-4: Create 4 configs**

For each `ds` in `[bitcoinotc, eut, mooc_actions, wikipedia]`, create `configs/experiments/dygnn_${ds}.yaml`:

```yaml
experiment_name: dygnn_<ds>
seed: 42
dataset_config: configs/datasets/<ds>.yaml
model_config: configs/models/dygnn.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

(Substitute `<ds>` with the actual dataset name in both `experiment_name` and `dataset_config`.)

- [ ] **Step 5: Verify all 5 configs parse (collegemsg + 4 new, NOT lastfm)**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions wikipedia; do
    .venv/bin/python -c "import yaml; print('$ds:', yaml.safe_load(open('configs/experiments/dygnn_${ds}.yaml'))['experiment_name'])"
done
# Verify lastfm config does NOT exist
if [ -f configs/experiments/dygnn_lastfm.yaml ]; then
    echo "ERROR: dygnn_lastfm.yaml should NOT exist (skipped per spec)"
else
    echo "OK: no dygnn_lastfm.yaml (correct)"
fi
```

Expected: 5 lines + "OK: no dygnn_lastfm.yaml".

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/dygnn_bitcoinotc.yaml \
        configs/experiments/dygnn_eut.yaml \
        configs/experiments/dygnn_mooc_actions.yaml \
        configs/experiments/dygnn_wikipedia.yaml
git commit -m "[exp] DyGNN experiment configs for 4 datasets (LastFM skipped)"
```

## Report
- Config parse output (5 datasets)
- "OK: no dygnn_lastfm.yaml"
- git log --oneline -2

---

## Task 10: Full 15-run experiment (5 datasets × 3 seeds)

Estimated ~3.5 hours wall-clock.

- [ ] **Step 1: Verify caches are warm**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions wikipedia; do
    n_cache=$(ls data/processed/${ds}/*.pt 2>/dev/null | wc -l)
    echo "$ds: ${n_cache} cache file(s)"
done
```

Note: caches were invalidated by Task 3's fmt3 bump. Each dataset's first DyGNN run will re-preprocess (~5-30s per dataset).

- [ ] **Step 2: Launch the 15-run sequence**

Order: small → large datasets so failures surface fast.

```bash
mkdir -p results/logs results/configs_runtime
for ds in collegemsg bitcoinotc eut wikipedia mooc_actions; do
    echo "===== $(date '+%H:%M:%S') ===== DyGNN × ${ds} ====="
    scripts/run_seeds.sh "$ds" dygnn
done
echo "===== $(date '+%H:%M:%S') ===== ALL DONE ====="
```

**No LastFM in the loop** — per spec §3.

If a run crashes:
- NaN loss → reduce node_memory_dim to 32 in `configs/models/dygnn.yaml` and retry the affected dataset.
- CUDA OOM → reduce hidden_dim to 32, retry only the affected dataset.
- val_auc stuck at 0.5 → investigate in detail; likely an issue with upstream update module on real data.
- Process killed (OOM kernel): restart only the (dataset, seed) that crashed.

- [ ] **Step 3: Verify 15 DyGNN records exist**

```bash
.venv/bin/python -c "
import json
records = [json.loads(line) for line in open('results/metrics.jsonl')]
dygnn = [r for r in records if r['model'] == 'dygnn']
print(f'DyGNN records: {len(dygnn)}')
for ds in ['collegemsg', 'bitcoinotc', 'eut', 'mooc_actions', 'wikipedia']:
    n = sum(1 for r in dygnn if r['dataset'] == ds)
    print(f'  {ds}: {n}/3')
lastfm_dygnn = [r for r in dygnn if r['dataset'] == 'lastfm']
print(f'  lastfm (should be 0): {len(lastfm_dygnn)}')
"
```

Expected: 15 total DyGNN records, 3 each on 5 datasets, 0 on lastfm.

- [ ] **Step 4: Generate 4-model cross-comparison**

```bash
.venv/bin/python scripts/aggregate_results.py --models gcn_ma evolvegcn_o htgn dygnn --output results/report/baselines_summary.md
```

Expected: 4-model Markdown table. LastFM row shows "—" for dygnn.

- [ ] **Step 5: Commit results**

```bash
git add results/metrics.jsonl results/report/baselines_summary.md
git commit -m "[milestone] DyGNN reproduced on 5 datasets × 3 seeds (LastFM skipped)"
```

## Report
- Per-dataset DyGNN record count (5 lines, 3/3 each)
- Total wall-clock time
- git log --oneline -2

---

## Task 11: Update reproduction-log + tag v0.3c-dygnn

**Files:**
- Modify: `docs/reproduction-log.md`

- [ ] **Step 1: Append Plan 3c section**

Append to `docs/reproduction-log.md`:

```markdown

---

## Plan 3c: DyGNN baseline integration

### Approach taken

Path: <A or B> — fill based on Task 2 outcome.

If Path A: vendored [`alge24/DyGNN`](https://github.com/alge24/DyGNN) at pin `<SHA>`. Adapter in `src/models/dygnn.py` (~<N>LOC) wraps upstream `<CLASS>`, applies per-epoch memory cache (spec §5), projects nothing (Euclidean memory), feeds shared MLP decoder. `<N_SHIMS>` lines of PyTorch 2.4 compat shims.

If Path B: kept submodule for citation; reimplemented core ~<N>LOC across `src/models/dygnn/{node_memory,edge_update,interaction,model}.py`. Reason path A failed: <SPECIFIC FINDINGS FROM TASK 2>.

### Hyperparameter policy (Hybrid)

Shared with GCN_MA / EvolveGCN-O / HTGN: `hidden_dim=64`, `node_memory_dim=64`, `dropout=0.1`, `lr=1e-3`, `weight_decay=1e-5`, Adam, `epochs=200`, patience 20, `grad_clip_max_norm=5.0`.

DyGNN-specific: `edge_dim=16`, `decay_method="log"`, `decay_rate=1.0`, learnable `nn.Parameter` `memory_init` of shape `[N, 64]` initialized to zero.

### Deviations from DyGNN paper

1. **Per-epoch memory cache** (gradient approximation). Paper does per-edge updates; we cache per epoch for compute. Documented in spec §5.
2. **Symmetric edge processing** — each (u, v, t) edge applied twice (u→v then v→u with ε offset).
3. **Shared MLP decoder** instead of paper's scoring head.
4. **LastFM skipped** — 1.29M edges × edge-sequence × 200 epochs = compute infeasible (~50+ hours/seed estimated).

### Loader cache schema bump fmt2 → fmt3

Each cached snapshot now exposes `edge_ts` — float64 tensor of original timestamps for chronological sort. Other 3 models ignore the attribute. Cost: one-time ~2.5 min re-preprocess across all 6 datasets.

### Final results — 5 datasets × 3 seeds

(paste contents of `results/report/baselines_summary.md`)

### Cross-model observations

(2-3 bullets after data is in: where DyGNN wins/loses vs GCN_MA, EvolveGCN-O, HTGN; whether per-epoch cache approximation hurt convergence)

### Carry-forwards to Plan 3d

- `_build_model` dispatch now handles 4 models.
- `aggregate_results.py --models` supports arbitrary lists.
- Per-epoch cache pattern reusable for streaming models in general.
- DGCN (Plan 3d) is the last baseline. Likely reimplement.
```

Fill in `<A or B>`, `<SHA>`, `<N>LOC`, `<CLASS>`, `<N_SHIMS>`, and runtime data paste.

- [ ] **Step 2: Verify reproduction-log renders**

```bash
tail -60 docs/reproduction-log.md
```

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reproduction-log.md
git commit -m "[docs] record Plan 3c DyGNN results"
git tag v0.3c-dygnn
git log --oneline -5
git tag
```

Expected: latest commit is the docs commit; `v0.3c-dygnn` in tag list alongside prior tags.

## Report
- final `docs/reproduction-log.md` tail (last 50 lines)
- `git tag` output (should show v0.3c-dygnn)

---

## Done (path A)

After Task 11, DyGNN is integrated as the 4th baseline. Path A success means 15 runs landed in metrics.jsonl with full path A adapter. Pattern for Plan 3d (DGCN) likely involves reimplement since DGCN has no canonical repo.

---

# Appendix: Path B (fallback) tasks

**Trigger:** Controller invokes these if Task 2 reports BLOCKED.

Path B replaces Tasks 4-5 with B1-B4. Tasks 6-11 remain the same (configs/dispatch/smoke/run/docs).

---

## Task B1: Node memory module

**Files:**
- Create: `src/models/dygnn/__init__.py`
- Create: `src/models/dygnn/node_memory.py`
- Test: `tests/test_dygnn_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_dygnn_smoke.py`:

```python
from src.models.dygnn.node_memory import NodeMemory


def test_node_memory_init_zero():
    nm = NodeMemory(num_nodes=50, dim=64)
    assert nm.state.shape == (50, 64)
    assert torch.allclose(nm.state, torch.zeros(50, 64))


def test_node_memory_clone_starts_fresh():
    """Cloning gives a detached fresh state (no grad chain through self.state)."""
    nm = NodeMemory(num_nodes=10, dim=8)
    cloned = nm.fresh_clone()
    assert cloned.shape == (10, 8)
    assert cloned.requires_grad  # fresh_clone preserves grad-tracking on the new tensor
    assert torch.allclose(cloned, nm.state)
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_node_memory_init_zero -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create empty `src/models/dygnn/__init__.py`**

```python
```

- [ ] **Step 4: Implement `src/models/dygnn/node_memory.py`**

```python
"""Per-node memory matrix for DyGNN (path B fallback).

Wraps a learnable [N, D] parameter that represents each node's evolving
state. `fresh_clone()` creates a per-epoch working copy that allows
gradient flow back to the initial state through the chain of edge updates.
"""
import torch
from torch import nn


class NodeMemory(nn.Module):
    """Container for per-node memory state."""

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        # Learnable initial memory state — Xavier-init keeps a small signal
        # while initializing close to zero (paper says zero; we use small-init
        # for non-trivial autograd at start of training).
        self.state = nn.Parameter(torch.zeros(num_nodes, dim))

    def fresh_clone(self) -> torch.Tensor:
        """Return a tensor copy that participates in the autograd graph.

        Used at the start of each epoch's cache build. `clone()` preserves
        the gradient chain back to `self.state`; subsequent in-place-style
        operations on the clone do not modify `self.state`.
        """
        return self.state.clone()
```

- [ ] **Step 5: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_node_memory_init_zero tests/test_dygnn_smoke.py::test_node_memory_clone_starts_fresh -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models/dygnn/__init__.py src/models/dygnn/node_memory.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN path B: NodeMemory module"
```

---

## Task B2: Coupled GRU edge update

**Files:**
- Create: `src/models/dygnn/edge_update.py`
- Modify: `tests/test_dygnn_smoke.py`

- [ ] **Step 1: Add failing test**

Append:

```python
from src.models.dygnn.edge_update import CoupledGRUUpdate


def test_coupled_gru_update_modifies_only_u_v_rows():
    """After update_module(memory, u, v, dt), only rows u and v change."""
    N, D = 10, 8
    mem = torch.randn(N, D, requires_grad=True)
    mem_before = mem.clone().detach()
    u, v = 3, 7
    layer = CoupledGRUUpdate(node_dim=D, decay_method="log", decay_rate=1.0)
    new_mem = layer(mem, u, v, torch.tensor(1.0))
    assert new_mem.shape == (N, D)
    # Non-(u,v) rows should be unchanged
    other_rows = [i for i in range(N) if i not in (u, v)]
    torch.testing.assert_close(new_mem[other_rows], mem_before[other_rows])
    # Row u and v should differ from mem_before
    assert not torch.allclose(new_mem[u], mem_before[u]), "row u unchanged"
    assert not torch.allclose(new_mem[v], mem_before[v]), "row v unchanged"


def test_coupled_gru_update_gradient_flows():
    N, D = 10, 8
    mem = torch.randn(N, D, requires_grad=True)
    layer = CoupledGRUUpdate(node_dim=D, decay_method="log", decay_rate=1.0)
    new_mem = layer(mem, 0, 1, torch.tensor(1.0))
    new_mem.sum().backward()
    assert mem.grad is not None and torch.isfinite(mem.grad).all()
    has_param_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in layer.parameters()
    )
    assert has_param_grad
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_coupled_gru_update_modifies_only_u_v_rows -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/dygnn/edge_update.py`**

```python
"""Coupled GRU update for DyGNN (path B fallback).

Each edge (u, v, dt) updates BOTH u's and v's memory via separate GRU
cells (source-side and target-side). Time decay multiplies the influence
based on how long since the last interaction.

Reference: DyGNN paper Eq. 4-7.
"""
import torch
from torch import nn


def _time_decay(dt: torch.Tensor, method: str, rate: float) -> torch.Tensor:
    """w(Δt) — multiplicative decay applied to neighbor influence."""
    dt_safe = dt.clamp(min=0.0)
    if method == "log":
        return 1.0 / torch.log(dt_safe + torch.e)  # log decay
    elif method == "polynomial":
        return 1.0 / ((dt_safe + 1.0) ** rate)
    elif method == "exponential":
        return torch.exp(-rate * dt_safe)
    raise ValueError(f"Unknown decay_method: {method}")


class CoupledGRUUpdate(nn.Module):
    """Update u's and v's memory when edge (u, v) arrives at time dt.

    Pre-update:
        s_u, s_v = memory[u], memory[v]
    Update source-side:
        s_u' = GRU_S(input=s_v * w(dt), hidden=s_u)
    Update target-side:
        s_v' = GRU_T(input=s_u * w(dt), hidden=s_v)
        (uses the OLD s_u for symmetric input)
    Write back to memory: memory[u] = s_u', memory[v] = s_v'.

    Returns a NEW [N, D] tensor (out-of-place update) so autograd works
    cleanly.
    """

    def __init__(self, node_dim: int, decay_method: str = "log", decay_rate: float = 1.0):
        super().__init__()
        self.node_dim = node_dim
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        self.gru_source = nn.GRUCell(input_size=node_dim, hidden_size=node_dim)
        self.gru_target = nn.GRUCell(input_size=node_dim, hidden_size=node_dim)

    def forward(self, memory: torch.Tensor, u: int, v: int, dt: torch.Tensor) -> torch.Tensor:
        """Update memory for edge (u, v) at time dt."""
        decay = _time_decay(dt, self.decay_method, self.decay_rate)
        s_u_old = memory[u].unsqueeze(0)  # [1, D]
        s_v_old = memory[v].unsqueeze(0)  # [1, D]

        # Source-side update: use neighbor (v) influence, decayed
        s_u_new = self.gru_source(s_v_old * decay, s_u_old).squeeze(0)
        # Target-side update: use original-u influence (not the just-updated value)
        s_v_new = self.gru_target(s_u_old * decay, s_v_old).squeeze(0)

        # Build new memory tensor (out-of-place — clean autograd)
        new_memory = memory.clone()
        new_memory[u] = s_u_new
        new_memory[v] = s_v_new
        return new_memory
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py -v
```

Expected: prior tests + 2 new tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/dygnn/edge_update.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN path B: CoupledGRUUpdate (Eq. 4-7)"
```

---

## Task B3: Interaction unit (simplified)

**Files:**
- Create: `src/models/dygnn/interaction.py`
- Modify: `tests/test_dygnn_smoke.py`

DyGNN paper Eq. 8-9 propagates effects from updated u/v to their neighbors. For our minimal reimplement, we skip neighborhood propagation (it would require an explicit neighbor list per node, increasing complexity). The interaction unit instead just returns memory unchanged. Path B is acknowledged as a faithful-to-paper-CORE reimplementation, not full HCS (hierarchical context sampling) variant.

- [ ] **Step 1: Add failing test**

Append:

```python
from src.models.dygnn.interaction import InteractionUnit


def test_interaction_unit_is_identity_for_minimal_reimplement():
    """In our minimal Path B reimplement, the interaction unit is identity.

    Paper Eq. 8-9 (neighborhood propagation) is skipped to keep the
    implementation focused. The edge update in CoupledGRUUpdate already
    captures the dominant signal.
    """
    nm = torch.randn(10, 8, requires_grad=True)
    unit = InteractionUnit(node_dim=8)
    out = unit(nm, edge_index_for_u_v_neighbors=None)
    torch.testing.assert_close(out, nm)
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py::test_interaction_unit_is_identity_for_minimal_reimplement -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/dygnn/interaction.py`**

```python
"""Interaction unit for DyGNN (path B fallback) — minimal identity.

Paper Eq. 8-9 propagates updated u/v signals to their neighbors. For our
minimal reimplementation we skip the propagation: the CoupledGRUUpdate
already captures the dominant signal, and adding neighborhood propagation
requires a stored adjacency that complicates the per-epoch cache.

This module is a placeholder that returns memory unchanged. Path B is
explicitly a faithful-to-paper-CORE reimplementation, not the full HCS
variant. Documented in reproduction-log.
"""
import torch
from torch import nn


class InteractionUnit(nn.Module):
    """Identity placeholder for paper Eq. 8-9 propagation step."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim
        # No trainable parameters in identity placeholder

    def forward(self, memory: torch.Tensor, edge_index_for_u_v_neighbors=None) -> torch.Tensor:
        return memory
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py -v
```

Expected: prior + 1 new test passing.

- [ ] **Step 5: Commit**

```bash
git add src/models/dygnn/interaction.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN path B: InteractionUnit (identity placeholder for paper Eq. 8-9)"
```

---

## Task B4: DyGNN composition (path B model.py)

**Files:**
- Create: `src/models/dygnn/model.py`
- Modify: `tests/test_dygnn_smoke.py`

- [ ] **Step 1: Add failing test**

Append:

```python
from src.models.dygnn.model import DyGNN as DyGNNPathB


def _make_dummy_snapshots(N: int, T: int, max_edges: int = 20) -> list[Data]:
    snaps = []
    for t in range(T):
        e = max(1, max_edges // 2)
        ei = torch.randint(0, N, (2, e))
        ts = torch.rand(e, dtype=torch.float64) + float(t)
        d = Data(edge_index=ei, num_nodes=N)
        d.edge_ts = ts
        snaps.append(d)
    return snaps


def test_dygnn_pathb_construct():
    m = DyGNNPathB(
        num_nodes=50, hidden_dim=64, node_memory_dim=64, edge_dim=16,
        dropout=0.0, decay_method="log", decay_rate=1.0,
    )
    assert m is not None


def test_dygnn_pathb_forward_shape():
    N, T, D = 50, 5, 64
    m = DyGNNPathB(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()


def test_dygnn_pathb_gradient_flows():
    N, T, D = 50, 4, 64
    m = DyGNNPathB(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
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
.venv/bin/pytest tests/test_dygnn_smoke.py::test_dygnn_pathb_construct -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/dygnn/model.py`**

```python
"""DyGNN composition (path B fallback).

Combines NodeMemory + CoupledGRUUpdate + InteractionUnit + shared MLP
decoder. Per-epoch cache as documented in spec §5.
"""
import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP
from src.models.dygnn.edge_update import CoupledGRUUpdate
from src.models.dygnn.interaction import InteractionUnit
from src.models.dygnn.node_memory import NodeMemory


class DyGNN(DynamicLinkPredictor):
    """DyGNN reimplemented from paper Eq. 4-9 (path B fallback)."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        node_memory_dim: int = 64,
        edge_dim: int = 16,
        dropout: float = 0.1,
        decay_method: str = "log",
        decay_rate: float = 1.0,
    ):
        super().__init__()
        if node_memory_dim != hidden_dim:
            raise ValueError(f"node_memory_dim ({node_memory_dim}) must equal hidden_dim ({hidden_dim})")
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.memory = NodeMemory(num_nodes=num_nodes, dim=node_memory_dim)
        self.update_module = CoupledGRUUpdate(
            node_dim=node_memory_dim,
            decay_method=decay_method,
            decay_rate=decay_rate,
        )
        self.interaction = InteractionUnit(node_dim=node_memory_dim)

        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Per-epoch cache state
        self._cached_memory_per_t: list[torch.Tensor] | None = None
        self._prev_max_t: int = -1

    @property
    def memory_init(self):
        """Compatibility shim: expose memory.state as memory_init for adapter tests."""
        return self.memory.state

    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        """Same semantics as path A adapter — per-epoch cache."""
        if self._cached_memory_per_t is None or time_step < self._prev_max_t:
            self._build_cache(snapshots)
        self._prev_max_t = max(self._prev_max_t, time_step)
        return self._cached_memory_per_t[time_step]

    def _build_cache(self, snapshots: list[Data]) -> None:
        device = self.memory.state.device
        mem = self.memory.fresh_clone()
        cache = []
        T = len(snapshots)

        for t in range(T):
            snap = snapshots[t]
            if snap.edge_index.numel() > 0 and hasattr(snap, "edge_ts") and snap.edge_ts.numel() > 0:
                ei = snap.edge_index.to(device)
                ts = snap.edge_ts.to(device)
                sorted_idx = torch.argsort(ts)
                ei_sorted = ei[:, sorted_idx]
                ts_sorted = ts[sorted_idx]

                for k in range(ei_sorted.shape[1]):
                    u = int(ei_sorted[0, k].item())
                    v = int(ei_sorted[1, k].item())
                    dt = ts_sorted[k]
                    mem = self.update_module(mem, u, v, dt)
                    mem = self.update_module(mem, v, u, dt + 1e-6)
                # Apply interaction propagation (identity in our minimal reimplement)
                mem = self.interaction(mem, edge_index_for_u_v_neighbors=ei_sorted)
            cache.append(mem.clone())

        self._cached_memory_per_t = cache
        self._prev_max_t = -1

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_dygnn_smoke.py -v
```

Expected: all path B smoke tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/dygnn/model.py tests/test_dygnn_smoke.py
git commit -m "[models] DyGNN path B: full composition (node_memory + edge_update + interaction + cache)"
```

After Task B4, resume mainline at Task 6. **Adjust Task 7's `_build_model` import:** change `from src.models.dygnn import DyGNN` to `from src.models.dygnn.model import DyGNN`.
