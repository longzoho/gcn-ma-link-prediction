# EvolveGCN-O Baseline Integration (Plan 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EvolveGCN-O as a baseline running end-to-end on all 6 datasets × 3 seeds, sharing the existing trainer/evaluator/loader pipeline with GCN_MA.

**Architecture:** Vendor `IBM/EvolveGCN` as a git submodule at a pinned commit. Implement a thin adapter `src/models/evolvegcn.py` (~120 LOC) extending `DynamicLinkPredictor`. Adapter translates between our `DynamicGraph` and upstream's `(A_list, X_list, mask_list)` format and provides a learnable node embedding in place of one-hot identity (RAM justification — one-hot is 34-59 GB on the largest datasets).

**Tech Stack:** Python 3.11, PyTorch 2.4, PyTorch Geometric 2.6. Reuses everything from Plan 1/2.

**Spec:** `docs/superpowers/specs/2026-05-16-evolvegcn-o-integration-design.md`

**Predecessor:** Plan 2 (tag `v0.2-gcn-ma-full`, commit `848ed31`).

**Out of scope:** HTGN, DyGNN, DGCN (Plans 3b/3c/3d). EvolveGCN-H variant. Hyperparameter tuning beyond Hybrid policy.

---

## File map

```
gcn-ma-link-prediction/
├── .gitmodules                                  # NEW
├── third_party/EvolveGCN/                       # NEW (submodule, pinned)
├── configs/
│   ├── models/evolvegcn_o.yaml                  # NEW
│   └── experiments/
│       ├── evolvegcn_o_collegemsg.yaml          # NEW
│       ├── evolvegcn_o_bitcoinotc.yaml          # NEW
│       ├── evolvegcn_o_eut.yaml                 # NEW
│       ├── evolvegcn_o_mooc_actions.yaml        # NEW
│       ├── evolvegcn_o_lastfm.yaml              # NEW
│       └── evolvegcn_o_wikipedia.yaml           # NEW
├── src/models/evolvegcn.py                      # NEW — adapter (~120 LOC)
├── scripts/
│   ├── train.py                                 # MODIFY — _build_model helper
│   ├── run_seeds.sh                             # MODIFY — accept (model, dataset)
│   └── aggregate_results.py                     # NEW — generalize aggregate_gcn_ma_results.py
├── tests/test_evolvegcn_smoke.py                # NEW — 4 unit tests
└── docs/reproduction-log.md                     # MODIFY — append Plan 3a section
```

---

## Task 1: Add IBM/EvolveGCN as git submodule

**Files:**
- Create: `.gitmodules`
- Create: `third_party/EvolveGCN/` (submodule)

- [ ] **Step 1: Add submodule**

```bash
mkdir -p third_party
git submodule add https://github.com/IBM/EvolveGCN third_party/EvolveGCN
```

- [ ] **Step 2: Pin commit and record SHA**

```bash
cd third_party/EvolveGCN
# Use HEAD on default branch as the pin
PIN_SHA=$(git rev-parse HEAD)
echo "EvolveGCN pinned commit: $PIN_SHA"
cd ../..
```

Record the SHA in your report — it goes into the reproduction log in Task 14.

- [ ] **Step 3: Verify submodule structure**

```bash
ls third_party/EvolveGCN/ | head -20
```

Expected: files include `egcn_o.py` (or similar), `models.py`, `utils.py`, configuration YAMLs, README. If `egcn_o.py` is not present at the top level, inspect to find the EvolveGCN-O class file.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party/EvolveGCN
git commit -m "[deps] vendor IBM/EvolveGCN as submodule, pinned to <SHA>"
```

Replace `<SHA>` with the pin SHA from Step 2.

---

## Task 2: Smoke-test upstream import + document API

**Files:**
- Create: `tests/test_evolvegcn_smoke.py` (initial version with import test only)

This task discovers the upstream API. Subsequent tasks depend on what is documented here.

- [ ] **Step 1: Inspect EvolveGCN-O class location and signature**

```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'third_party/EvolveGCN')
# Try common module names
for name in ['egcn_o', 'models.egcn_o', 'src.models.egcn_o']:
    try:
        mod = __import__(name, fromlist=['*'])
        print(f'OK: import {name} → {dir(mod)}')
        break
    except ImportError as e:
        print(f'FAIL: {name}: {e}')
"
```

Expected: identify the module containing the EvolveGCN-O class.

- [ ] **Step 2: Print the class init signature**

```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'third_party/EvolveGCN')
from <MODULE> import <CLASS_NAME>  # use names found in Step 1
import inspect
print('Init signature:', inspect.signature(<CLASS_NAME>.__init__))
print('Forward signature:', inspect.signature(<CLASS_NAME>.forward))
print('Class docstring:', <CLASS_NAME>.__doc__)
"
```

Record the exact signatures in your report.

- [ ] **Step 3: Write `tests/test_evolvegcn_smoke.py` initial version**

```python
"""Smoke tests for EvolveGCN-O upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "EvolveGCN"))


def test_can_import_upstream_egcn_o():
    """Upstream EGCN class must import without PyTorch 2.4 errors."""
    # Replace MODULE_NAME and CLASS_NAME with what Task 2 Step 1-2 discovered
    from <MODULE_NAME> import <CLASS_NAME>  # noqa: F401
    assert <CLASS_NAME> is not None
```

Substitute the actual module + class names found in Step 1.

- [ ] **Step 4: Run the smoke test**

```bash
.venv/bin/pytest tests/test_evolvegcn_smoke.py::test_can_import_upstream_egcn_o -v
```

Expected: PASS. Common failure modes and fixes:
- `ImportError: cannot import name 'inf' from 'torch._six'` → add shim at top of test file: `import torch; from types import SimpleNamespace; torch._six = getattr(torch, '_six', SimpleNamespace(inf=float('inf'), nan=float('nan')))`
- `ModuleNotFoundError: No module named 'X'` → the upstream depends on a package; install with `.venv/bin/uv pip install X` and retry.

If after 1 hour of shim attempts the import still fails (more than ~5 lines of shims needed), STOP and report BLOCKED. The controller will switch to a reimplementation path (separate plan revision).

- [ ] **Step 5: Document findings**

In your subagent report, include verbatim:
1. Module path used (e.g. `egcn_o` or `models.egcn_o`)
2. Class name (e.g. `EGCN` or `EGCN_o`)
3. `__init__` signature
4. `forward` signature
5. Any shims needed (count of lines)

- [ ] **Step 6: Commit**

```bash
git add tests/test_evolvegcn_smoke.py
git commit -m "[tests] smoke import test for upstream EvolveGCN-O

Module: <MODULE_NAME>
Class: <CLASS_NAME>
PyTorch 2.4 compat: <N lines of shims>"
```

---

## Task 3: Adapter scaffold + construction test

**Files:**
- Create: `src/models/evolvegcn.py` (initial scaffold)
- Modify: `tests/test_evolvegcn_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evolvegcn_smoke.py`:

```python
import torch

from src.models.evolvegcn import EvolveGCN_O


def test_can_construct_evolvegcn_o():
    """Adapter constructor must succeed with realistic shapes."""
    m = EvolveGCN_O(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    )
    assert m is not None
    # Node embedding should be initialized
    assert m.node_emb.weight.shape == (50, 64)
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_evolvegcn_smoke.py::test_can_construct_evolvegcn_o -v
```

Expected: `ImportError: No module named 'src.models.evolvegcn'`.

- [ ] **Step 3: Implement `src/models/evolvegcn.py` scaffold**

Adjust the module path and class name in the imports to match Task 2's findings.

```python
"""Adapter for IBM/EvolveGCN EvolveGCN-O baseline.

Wraps the upstream EGCN class as a DynamicLinkPredictor. Replaces the
one-hot identity feature convention (spec §6.6) with a learnable
nn.Embedding to keep RAM tractable on large datasets.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# Upstream submodule path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM = _REPO_ROOT / "third_party" / "EvolveGCN"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

# PyTorch 2.4 compatibility shims for upstream (2019-2020 code).
# If Task 2 needed shims, replicate them here.
if not hasattr(torch, "_six"):
    torch._six = SimpleNamespace(inf=float("inf"), nan=float("nan"))

from <UPSTREAM_MODULE> import <UPSTREAM_CLASS>  # set per Task 2 findings  # noqa: E501

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class EvolveGCN_O(DynamicLinkPredictor):
    """EvolveGCN-O wrapped as a DynamicLinkPredictor.

    Differences from upstream:
        - Uses our learnable nn.Embedding (not one-hot identity).
        - MLP decoder shared with GCN_MA for fair comparison.
        - Adapter consumes our DynamicGraph format.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Learnable node embedding replaces one-hot identity
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Upstream EGCN-O construction
        upstream_args = SimpleNamespace(
            num_layers=num_layers,
            feats_per_node=feat_dim,
            layer_1_feats=hidden_dim,
            layer_2_feats=hidden_dim,
        )
        self.core = <UPSTREAM_CLASS>(
            args=upstream_args,
            activation=nn.RReLU(),
            device=torch.device("cpu"),
        )

        # Shared MLP decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step):
        # Stub — fills in Task 4
        raise NotImplementedError("Implemented in Task 4")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
```

Adjust `<UPSTREAM_MODULE>` and `<UPSTREAM_CLASS>` according to Task 2 findings. If the upstream's `__init__` requires additional `args` fields beyond what is listed, add them to `upstream_args` with sensible defaults (consult upstream's own `run_exp.py` for default values).

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_evolvegcn_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/evolvegcn.py tests/test_evolvegcn_smoke.py
git commit -m "[models] EvolveGCN-O adapter scaffold + construction test"
```

---

## Task 4: Adapter forward — shape + gradient flow

**Files:**
- Modify: `src/models/evolvegcn.py`
- Modify: `tests/test_evolvegcn_smoke.py`

- [ ] **Step 1: Add failing shape + gradient tests**

Append to `tests/test_evolvegcn_smoke.py`:

```python
from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int) -> list[Data]:
    snaps = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, N * 2))
        d = Data(edge_index=ei, num_nodes=N)
        snaps.append(d)
    return snaps


def test_evolvegcn_forward_shape():
    N, T, D = 50, 5, 64
    m = EvolveGCN_O(num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D), f"expected ({N}, {D}), got {Z.shape}"


def test_evolvegcn_gradient_flows():
    N, T, D = 50, 4, 64
    m = EvolveGCN_O(num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    # At least one parameter must have finite gradient
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no finite non-zero gradient on any parameter"
```

- [ ] **Step 2: Run, verify fail**

```bash
.venv/bin/pytest tests/test_evolvegcn_smoke.py::test_evolvegcn_forward_shape -v
```

Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `forward` in `src/models/evolvegcn.py`**

Replace the `raise NotImplementedError(...)` line with the full forward:

```python
    def forward(self, snapshots, time_step):
        """Run upstream EGCN-O through snapshots [0..time_step].

        Returns Z^{time_step} ∈ R^{num_nodes × hidden_dim}.
        """
        device = self.node_emb.weight.device
        N = self.num_nodes
        node_ids = torch.arange(N, device=device)

        A_list, X_list, mask_list = [], [], []
        for tau in range(time_step + 1):
            snap = snapshots[tau]
            ei = snap.edge_index.to(device)
            if ei.numel() == 0:
                # Empty snapshot — add self-loops only so adjacency is well-defined.
                ei = torch.stack([node_ids, node_ids], dim=0)
            vals = torch.ones(ei.shape[1], device=device)
            A = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce()
            A_list.append(A)
            X_list.append(self.node_emb(node_ids))
            mask_list.append(torch.ones(N, device=device))

        # Upstream forward — argument order depends on Task 2 findings.
        # Most common: forward(A_list, Nodes_list, nodes_mask_list).
        Z = self.core(A_list, X_list, mask_list)
        return Z
```

If Task 2 found a different forward signature (e.g., `(Nodes_list, A_list, nodes_mask_list)` or just `(A_list, X_list)`), reorder arguments accordingly.

If upstream `forward` returns a tuple (some impls return `(Z, attn_weights)` or similar), unpack: `Z, *_ = self.core(...)`.

- [ ] **Step 4: Run, verify pass**

```bash
.venv/bin/pytest tests/test_evolvegcn_smoke.py -v
```

Expected: 4 passed.

If shape is wrong or gradients are NaN:
- Check upstream's expected dtype (some require float32 explicitly)
- Verify A_sparse → dense conversion isn't required by upstream
- Try moving to CPU first if CUDA sparse ops crash

- [ ] **Step 5: Commit**

```bash
git add src/models/evolvegcn.py tests/test_evolvegcn_smoke.py
git commit -m "[models] EvolveGCN-O adapter forward + smoke tests"
```

---

## Task 5: Model config + first experiment config

**Files:**
- Create: `configs/models/evolvegcn_o.yaml`
- Create: `configs/experiments/evolvegcn_o_collegemsg.yaml`

- [ ] **Step 1: Create `configs/models/evolvegcn_o.yaml`**

```yaml
name: evolvegcn_o
feat_dim: 64
hidden_dim: 64
num_layers: 2
activation: rrelu
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 200
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

- [ ] **Step 2: Create `configs/experiments/evolvegcn_o_collegemsg.yaml`**

```yaml
experiment_name: evolvegcn_o_collegemsg
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 3: Verify YAML parses**

```bash
.venv/bin/python -c "
import yaml
cfg = yaml.safe_load(open('configs/experiments/evolvegcn_o_collegemsg.yaml'))
mcfg = yaml.safe_load(open('configs/models/evolvegcn_o.yaml'))
print('Experiment:', cfg)
print('Model:', mcfg)
"
```

Expected: both configs print as dicts with the listed fields.

- [ ] **Step 4: Commit**

```bash
git add configs/models/evolvegcn_o.yaml configs/experiments/evolvegcn_o_collegemsg.yaml
git commit -m "[configs] EvolveGCN-O model config + CollegeMsg experiment"
```

---

## Task 6: CLI dispatch — `_build_model` helper

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Read current `scripts/train.py` around the `GCN_MA(...)` construction**

```bash
grep -n "GCN_MA(" scripts/train.py
```

Find the line (~line 95-105 in current state) that constructs the model. Note the surrounding context.

- [ ] **Step 2: Add `_build_model` helper at module level**

In `scripts/train.py`, BEFORE the `def main()` function, add:

```python
def _build_model(model_cfg: dict, graph):
    """Construct a DynamicLinkPredictor by model name."""
    name = model_cfg["name"]
    if name == "gcn_ma":
        from src.models.gcn_ma.model import GCN_MA
        return GCN_MA(
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["num_heads"],
            dropout=model_cfg["dropout"],
        )
    elif name == "evolvegcn_o":
        from src.models.evolvegcn import EvolveGCN_O
        return EvolveGCN_O(
            num_nodes=graph.num_nodes,
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unknown model name: {name!r}")
```

The existing `from src.models.gcn_ma.model import GCN_MA` at the top of the file can stay or be moved inside `_build_model` — both work. If it stays, the local import inside `_build_model` is redundant but harmless.

- [ ] **Step 3: Replace direct GCN_MA construction with `_build_model` call**

Find the line that currently reads (roughly):

```python
    model = GCN_MA(
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    )
```

Replace it with:

```python
    model = _build_model(model_cfg, graph)
```

- [ ] **Step 4: Update the metrics record's `model` field**

Find the line that currently reads:

```python
        "model": "gcn_ma",
```

Replace with:

```python
        "model": model_cfg["name"],
```

- [ ] **Step 5: Verify GCN_MA path still works (no regression)**

```bash
.venv/bin/python scripts/train.py --help
```

Expected: argparse usage prints without ImportError.

- [ ] **Step 6: Smoke-test GCN_MA still works via dispatcher**

```bash
.venv/bin/python scripts/train.py --config configs/experiments/gcn_ma_collegemsg_smoke.yaml 2>&1 | tail -10
```

Expected: training runs for 3 epochs, prints JSON record with `"model": "gcn_ma"`.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py
git commit -m "[scripts] _build_model dispatch supports gcn_ma + evolvegcn_o"
```

---

## Task 7: Multi-seed runner generalization

**Files:**
- Modify: `scripts/run_seeds.sh`

- [ ] **Step 1: Read current `scripts/run_seeds.sh`**

```bash
cat scripts/run_seeds.sh
```

Confirm it currently takes one positional arg (`$1`) for dataset.

- [ ] **Step 2: Replace with generalized version**

```bash
#!/usr/bin/env bash
# Run a model × dataset experiment across 3 seeds.
#
# Usage: scripts/run_seeds.sh <dataset> [<model>]
# Default model: gcn_ma (backward compatible).
# Example: scripts/run_seeds.sh collegemsg evolvegcn_o
#
# Reads configs/experiments/<model>_<dataset>.yaml, overrides seed for
# each run, appends to results/metrics.jsonl.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset> [<model>]" >&2; exit 1
fi
DATASET="$1"
MODEL="${2:-gcn_ma}"
SEEDS=(42 123 2024)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CFG="configs/experiments/${MODEL}_${DATASET}.yaml"
if [ ! -f "$BASE_CFG" ]; then
    echo "Missing experiment config: $BASE_CFG" >&2; exit 1
fi

mkdir -p results/logs results/configs_runtime
for SEED in "${SEEDS[@]}"; do
    RUN_CFG="results/configs_runtime/${MODEL}_${DATASET}_seed${SEED}.yaml"
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s/^experiment_name:.*/experiment_name: ${MODEL}_${DATASET}_seed${SEED}/" \
        "$BASE_CFG" > "$RUN_CFG"
    LOG="results/logs/${MODEL}_${DATASET}_seed${SEED}_$(date +%Y%m%d-%H%M%S).log"
    echo "=== Running ${MODEL}/${DATASET} seed=${SEED} → $LOG ==="
    .venv/bin/python scripts/train.py --config "$RUN_CFG" 2>&1 | tee "$LOG"
done
echo "Done: ${MODEL}/${DATASET} × 3 seeds. metrics.jsonl appended."
```

- [ ] **Step 3: Verify shell syntax**

```bash
bash -n scripts/run_seeds.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 4: Verify rendered config (manual test)**

```bash
RUN_CFG=$(mktemp)
sed -e 's/^seed:.*/seed: 123/' \
    -e 's/^experiment_name:.*/experiment_name: evolvegcn_o_collegemsg_seed123/' \
    configs/experiments/evolvegcn_o_collegemsg.yaml > "$RUN_CFG"
cat "$RUN_CFG"
rm "$RUN_CFG"
```

Expected: `seed: 123` and `experiment_name: evolvegcn_o_collegemsg_seed123` are substituted; other fields preserved.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_seeds.sh
git commit -m "[scripts] run_seeds.sh accepts (dataset, model) — backward compat default gcn_ma"
```

---

## Task 8: Smoke-train EvolveGCN-O on CollegeMsg

**Files:**
- Create: `configs/experiments/evolvegcn_o_collegemsg_smoke.yaml`
- Create: `configs/models/evolvegcn_o_smoke.yaml`

- [ ] **Step 1: Create smoke configs (reduced epochs)**

`configs/models/evolvegcn_o_smoke.yaml`:

```yaml
name: evolvegcn_o
feat_dim: 32
hidden_dim: 32
num_layers: 2
activation: rrelu
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 3
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

`configs/experiments/evolvegcn_o_collegemsg_smoke.yaml`:

```yaml
experiment_name: evolvegcn_o_collegemsg_smoke
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/evolvegcn_o_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_smoke.jsonl
```

- [ ] **Step 2: Run smoke training**

```bash
mkdir -p results/logs
.venv/bin/python scripts/train.py --config configs/experiments/evolvegcn_o_collegemsg_smoke.yaml 2>&1 | tail -25
```

Expected:
- 3 epochs complete without Python errors
- val_auc per epoch is a finite number in (0.0, 1.0), not stuck exactly at 0.5
- Final JSON record prints with `auc` and `ap` in [0.0, 1.0] and `"model": "evolvegcn_o"`
- `results/metrics_smoke.jsonl` has 1 line for evolvegcn_o (plus any prior GCN_MA smoke entries)

If validation crashes with empty-snapshot error: trainer fix from Plan 2 should handle it. If it doesn't, report BLOCKED with the traceback.

If val_auc is stuck exactly at 0.5: model isn't learning. Likely cause: forward pass returns embeddings disconnected from any gradient (e.g., upstream's `forward` returns a tensor we forgot to keep in the graph). Check `Z.requires_grad` in a debug print.

If CUDA OOM: reduce `feat_dim` and `hidden_dim` further in the smoke config (e.g., 16) and retry; the full config will use 64 on actual hardware.

- [ ] **Step 3: Commit smoke configs**

```bash
git add configs/models/evolvegcn_o_smoke.yaml configs/experiments/evolvegcn_o_collegemsg_smoke.yaml
git commit -m "[exp] EvolveGCN-O smoke configs + verified end-to-end on CollegeMsg"
```

---

## Task 9: Remaining 5 experiment configs

**Files:**
- Create: `configs/experiments/evolvegcn_o_{bitcoinotc,eut,mooc_actions,lastfm,wikipedia}.yaml`

All 5 follow the same template — only `dataset_config` differs.

- [ ] **Step 1: Create `configs/experiments/evolvegcn_o_bitcoinotc.yaml`**

```yaml
experiment_name: evolvegcn_o_bitcoinotc
seed: 42
dataset_config: configs/datasets/bitcoinotc.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 2: Create `configs/experiments/evolvegcn_o_eut.yaml`**

```yaml
experiment_name: evolvegcn_o_eut
seed: 42
dataset_config: configs/datasets/eut.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 3: Create `configs/experiments/evolvegcn_o_mooc_actions.yaml`**

```yaml
experiment_name: evolvegcn_o_mooc_actions
seed: 42
dataset_config: configs/datasets/mooc_actions.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Create `configs/experiments/evolvegcn_o_lastfm.yaml`**

```yaml
experiment_name: evolvegcn_o_lastfm
seed: 42
dataset_config: configs/datasets/lastfm.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 5: Create `configs/experiments/evolvegcn_o_wikipedia.yaml`**

```yaml
experiment_name: evolvegcn_o_wikipedia
seed: 42
dataset_config: configs/datasets/wikipedia.yaml
model_config: configs/models/evolvegcn_o.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 6: Verify all 6 experiment configs parse**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    .venv/bin/python -c "import yaml; print('$ds:', yaml.safe_load(open('configs/experiments/evolvegcn_o_${ds}.yaml'))['experiment_name'])"
done
```

Expected: 6 lines printing each `experiment_name`.

- [ ] **Step 7: Commit**

```bash
git add configs/experiments/evolvegcn_o_bitcoinotc.yaml \
        configs/experiments/evolvegcn_o_eut.yaml \
        configs/experiments/evolvegcn_o_mooc_actions.yaml \
        configs/experiments/evolvegcn_o_lastfm.yaml \
        configs/experiments/evolvegcn_o_wikipedia.yaml
git commit -m "[exp] EvolveGCN-O experiment configs for remaining 5 datasets"
```

---

## Task 10: Generalize aggregation script

**Files:**
- Create: `scripts/aggregate_results.py` (new, generalizes `aggregate_gcn_ma_results.py`)

- [ ] **Step 1: Implement `scripts/aggregate_results.py`**

```python
"""Aggregate model results across seeds and emit Markdown tables.

Usage:
  Single-model summary:
    python scripts/aggregate_results.py --model gcn_ma
  Cross-model comparison:
    python scripts/aggregate_results.py --models gcn_ma evolvegcn_o
"""
import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER_TABLE2_GCN_MA = {
    "collegemsg":   {"auc": 0.9149, "ap": 0.8926},
    "bitcoinotc":   {"auc": 0.9120, "ap": 0.8943},
    "eut":          {"auc": 0.9222, "ap": 0.9082},
    "mooc_actions": {"auc": 0.9880, "ap": 0.9863},
    "lastfm":       {"auc": 0.8757, "ap": 0.8704},
    "wikipedia":    {"auc": 0.8742, "ap": 0.8575},
}

DATASETS = ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]


def _mean_std(values):
    if not values:
        return None, None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _load_records(metrics_path: Path):
    by_model_dataset = {}
    with metrics_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["dataset"])
            by_model_dataset.setdefault(key, []).append(r)
    return by_model_dataset


def _single_model_table(by_md, model: str) -> str:
    lines = [f"# {model.upper()} Reproduction — Per-Dataset Summary\n"]
    lines.append("| Dataset | n seeds | AUC (mean ± std) | AP (mean ± std) | Paper AUC | Paper AP | Δ AUC | Δ AP |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        recs = by_md.get((model, ds), [])
        paper = PAPER_TABLE2_GCN_MA.get(ds, {"auc": float("nan"), "ap": float("nan")})
        if not recs:
            lines.append(f"| {ds} | 0 | — | — | {paper['auc']:.4f} | {paper['ap']:.4f} | — | — |")
            continue
        auc_m, auc_s = _mean_std([r["auc"] for r in recs])
        ap_m, ap_s = _mean_std([r["ap"] for r in recs])
        lines.append(
            f"| {ds} | {len(recs)} | {auc_m:.4f} ± {auc_s:.4f} | {ap_m:.4f} ± {ap_s:.4f} "
            f"| {paper['auc']:.4f} | {paper['ap']:.4f} | {auc_m - paper['auc']:+.4f} | {ap_m - paper['ap']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def _cross_model_table(by_md, models: list[str]) -> str:
    lines = [f"# Cross-Model Comparison — {', '.join(models)}\n"]
    header_cols = ["Dataset"]
    for m in models:
        header_cols.append(f"{m} AUC")
        header_cols.append(f"{m} AP")
    header_cols.append("Paper GCN_MA AUC")
    header_cols.append("Paper GCN_MA AP")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    for ds in DATASETS:
        row = [ds]
        for m in models:
            recs = by_md.get((m, ds), [])
            auc_m, auc_s = _mean_std([r["auc"] for r in recs])
            ap_m, ap_s = _mean_std([r["ap"] for r in recs])
            if auc_m is None:
                row.extend(["—", "—"])
            else:
                row.append(f"{auc_m:.4f} ± {auc_s:.4f}")
                row.append(f"{ap_m:.4f} ± {ap_s:.4f}")
        paper = PAPER_TABLE2_GCN_MA.get(ds, {"auc": float("nan"), "ap": float("nan")})
        row.append(f"{paper['auc']:.4f}")
        row.append(f"{paper['ap']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/report/results_summary.md", type=Path)
    parser.add_argument("--model", default=None, help="single-model mode")
    parser.add_argument("--models", nargs="+", default=None, help="cross-model mode (≥2 names)")
    args = parser.parse_args()

    by_md = _load_records(REPO_ROOT / args.metrics)

    if args.models and len(args.models) >= 2:
        out = _cross_model_table(by_md, args.models)
    elif args.model:
        out = _single_model_table(by_md, args.model)
    else:
        parser.error("Specify --model NAME or --models NAME1 NAME2 [...]")

    print(out)
    (REPO_ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output).write_text(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test against Plan 2 GCN_MA results**

```bash
.venv/bin/python scripts/aggregate_results.py --model gcn_ma --output results/report/gcn_ma_check.md
```

Expected: prints Markdown table identical to Plan 2's `results/report/gcn_ma_summary.md` content (same numbers).

- [ ] **Step 3: Test cross-model mode (no EvolveGCN data yet → 0 seeds for evolvegcn_o)**

```bash
.venv/bin/python scripts/aggregate_results.py --models gcn_ma evolvegcn_o --output results/report/cross_check.md
```

Expected: 2-model table with all rows showing GCN_MA values and EvolveGCN-O as "—".

- [ ] **Step 4: Commit**

```bash
git add scripts/aggregate_results.py
git commit -m "[scripts] generalized aggregate_results.py supporting single/cross-model tables"
```

---

## Task 11: Full 18-run experiment

This is the big run. Estimated ~2.5 hours wall-clock.

- [ ] **Step 1: Verify cache is warm for all 6 datasets**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    n_cache=$(ls data/processed/${ds}/*.pt 2>/dev/null | wc -l)
    echo "$ds: ${n_cache} cache file(s)"
done
```

Expected: each dataset shows ≥ 1 cache file (set up by Plan 2). If any is missing, run the loader smoke check from Plan 2 to rebuild:

```bash
.venv/bin/python -c "
from pathlib import Path
from src.data.loaders.<NAME> import <Loader>
<Loader>().build(Path('data/raw/<NAME>/<RAWFILE>'), Path('data/processed/<NAME>'), num_time_steps=<T>, beta=0.8)
"
```

- [ ] **Step 2: Launch the 18-run sequence**

Order: small → large datasets so any failure surfaces fast.

```bash
mkdir -p results/logs results/configs_runtime
for ds in collegemsg bitcoinotc eut wikipedia lastfm mooc_actions; do
    echo "===== $(date '+%H:%M:%S') ===== EvolveGCN-O × ${ds} ====="
    scripts/run_seeds.sh "$ds" evolvegcn_o
done
echo "===== $(date '+%H:%M:%S') ===== ALL DONE ====="
```

This runs in the foreground but can be backgrounded if desired. Expected total ~2.5 hours.

If a run crashes mid-way:
- EUT empty-snapshot issue should be handled by Plan 2's trainer fix
- Bipartite dataset OOM: shouldn't happen at h=64, but if so, reduce to h=32 in the model config and re-run only the affected dataset
- Other crash: collect log, report BLOCKED with traceback

- [ ] **Step 3: Verify 18 EvolveGCN-O records exist**

```bash
.venv/bin/python -c "
import json
n = 0
with open('results/metrics.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['model'] == 'evolvegcn_o':
            n += 1
print(f'EvolveGCN-O records: {n}')
"
```

Expected: 18. If less, identify which (dataset, seed) is missing from the logs and re-run that specific (dataset, seed) via:

```bash
RUN_CFG="results/configs_runtime/evolvegcn_o_<DATASET>_seed<SEED>.yaml"
.venv/bin/python scripts/train.py --config "$RUN_CFG"
```

- [ ] **Step 4: Run aggregation in cross-model mode**

```bash
.venv/bin/python scripts/aggregate_results.py --models gcn_ma evolvegcn_o --output results/report/baselines_summary.md
```

Expected: Markdown table with both GCN_MA and EvolveGCN-O mean±std across all 6 datasets.

- [ ] **Step 5: Commit results**

```bash
git add results/metrics.jsonl results/report/baselines_summary.md
git commit -m "[milestone] EvolveGCN-O reproduced on 6 datasets × 3 seeds"
```

---

## Task 12: Update reproduction-log + tag v0.3a

**Files:**
- Modify: `docs/reproduction-log.md`

- [ ] **Step 1: Append Plan 3a section**

Open `docs/reproduction-log.md` and append at the end:

```markdown

---

## Plan 3a: EvolveGCN-O baseline integration

### Approach

Vendored [`IBM/EvolveGCN`](https://github.com/IBM/EvolveGCN) as a git submodule at pinned commit `<SHA>` (see `.gitmodules`). Wrote a thin adapter `src/models/evolvegcn.py` (~<N>LOC) extending `DynamicLinkPredictor`, sharing the trainer/evaluator/loaders with GCN_MA from Plan 1/2.

### Hyperparameter policy (Hybrid)

| Parameter | EvolveGCN-O | GCN_MA |
|---|---|---|
| feat_dim | 64 (learnable embedding) | 3 ([degree, CC, AS]) |
| hidden_dim | 64 | 64 |
| num_layers | 2 (EvolveGCN paper default) | 1 GCN layer |
| activation | RReLU (EvolveGCN paper default) | ReLU |
| lr, weight_decay, optimizer, epochs, patience, dropout, grad_clip | shared | — |

Shared values: lr=1e-3, weight_decay=1e-5, Adam, epochs=200 patience 20, dropout=0.1, grad_clip_max_norm=5.0.

### Deviation from main spec §6.6

Spec called for one-hot identity `I_N` as the baseline input feature, but RAM cost is prohibitive on the larger datasets (one-hot is 34-59 GB for Bitcoinotc, Mooc, and Wikipedia at our T values). Replaced with `nn.Embedding(N, feat_dim)` Xavier-initialized — same convention as IBM/EvolveGCN's own code for large-N cases. Documented justification: §6.6 was a Plan 1 oversight that did not budget for full-dataset memory.

### PyTorch 2.4 compatibility shims

Upstream IBM/EvolveGCN was written for PyTorch 1.x. Adapter applies the following shims at import time (in `src/models/evolvegcn.py`):

<paste the shim lines actually used, even if zero — note "no shims needed" if so>

### Final results — 6 datasets × 3 seeds

(paste contents of `results/report/baselines_summary.md`)

### Cross-model observations

(2-3 bullets: in which datasets does EvolveGCN-O beat / lose to GCN_MA, do those gaps match the paper's Table 2)

### Carry-forwards to Plans 3b/3c/3d

- The `_build_model` dispatch in `scripts/train.py` is now extensible — Plan 3b adds HTGN with the same pattern.
- The `aggregate_results.py` cross-model table supports any number of models — Plan 4 will produce the full 5-column comparison.
- If subsequent baselines also need learnable embeddings, factor the embedding into a shared utility.
```

Replace `<SHA>`, `<N>LOC`, and the paste-marker blocks with actual values.

- [ ] **Step 2: Verify reproduction-log renders cleanly**

```bash
head -20 docs/reproduction-log.md
tail -40 docs/reproduction-log.md
```

Expected: top of file is Plan 1 header; bottom is the new Plan 3a section.

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reproduction-log.md
git commit -m "[docs] record Plan 3a EvolveGCN-O results"
git tag v0.3a-evolvegcn-o
git log --oneline -5
git tag
```

Expected: latest commit is the docs commit; `v0.3a-evolvegcn-o` appears in tag list alongside `v0.1-foundation`, `v0.1-foundation-smoke`, `v0.2-gcn-ma-full`.

---

## Done

Plan 3a ships EvolveGCN-O as a working baseline on all 6 datasets with multi-seed mean ± std vs paper Table 2 and vs GCN_MA. Pattern for adding the next baseline (HTGN in Plan 3b): submodule → smoke import → adapter scaffold → forward → configs → no further changes to trainer/evaluator/CLI.
