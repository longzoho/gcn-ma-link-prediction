# GCN_MA Foundation (CollegeMsg end-to-end) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working GCN_MA model that trains and evaluates end-to-end on the CollegeMsg dataset, producing AUC and AP numbers to compare against the paper.

**Architecture:** Monorepo with `src/` modules for data, models, training, and evaluation. PyTorch Geometric for graph operations. `DynamicGraph` dataclass standardizes data flow. GCN_MA composes 4 sub-modules: spectral GCN with NRNAE-enhanced adjacency, LSTM weight updater across time, multi-head self-attention within a snapshot, MLP link decoder.

**Tech Stack:** Python 3.11, uv, PyTorch 2.4, PyTorch Geometric 2.6, NetworkX, scikit-learn, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-05-16-gcn-ma-link-prediction-design.md`

**Out of scope for this plan:** baselines (EvolveGCN, HTGN, DyGNN, DGCN), datasets other than CollegeMsg, hyperparameter grid search, aggregation/plotting scripts, thesis assets. All covered in subsequent plans.

---

## File map

```
gcn-ma-link-prediction/
├── pyproject.toml                # NEW
├── .python-version               # NEW
├── README.md                     # NEW
├── main.py                       # DELETE (PyCharm template)
├── configs/
│   ├── datasets/collegemsg.yaml  # NEW
│   └── models/gcn_ma.yaml        # NEW
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py               # DynamicGraph dataclass + split helpers
│   │   ├── preprocess.py         # NRNAE: CC, AS, S_ij, Ŝ
│   │   └── loaders/
│   │       ├── __init__.py
│   │       └── collegemsg.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # DynamicLinkPredictor ABC
│   │   └── gcn_ma/
│   │       ├── __init__.py
│   │       ├── gcn_layer.py      # spectral GCN w/ enhanced adjacency
│   │       ├── lstm_weight.py    # LSTM weight updater
│   │       ├── attention.py      # multi-head self-attention
│   │       ├── link_decoder.py   # MLP decoder
│   │       └── model.py          # composition
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── negative_sampling.py
│   │   └── trainer.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py            # AUC, AP
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── logging.py
├── scripts/
│   ├── download_datasets.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_dynamic_graph.py
│   ├── test_nrnae.py
│   ├── test_collegemsg_loader.py
│   ├── test_data_split.py
│   ├── test_negative_sampling.py
│   ├── test_metrics.py
│   ├── test_evaluator.py
│   ├── test_seed.py
│   └── test_models_smoke.py
└── docs/
    ├── paper-notes.md
    └── reproduction-log.md
```

**Testing strategy:** Strict TDD for data layer, metrics, negative sampling, split (computable ground truth). Smoke tests (shape + grad flow) for model components — research model correctness is validated by matching paper numbers, not unit tests. Manual integration test at the end (Task 22).

---

## Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`, `.python-version`, `README.md`
- Delete: `main.py`

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "gcn-ma-link-prediction"
version = "0.1.0"
description = "Reproduction of GCN_MA (Mei & Zhao 2024) for dynamic network link prediction"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch==2.4.0",
    "torch-geometric==2.6.1",
    "networkx>=3.2",
    "pandas>=2.0",
    "numpy>=1.26,<2.0",
    "scikit-learn>=1.4",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1", "ruff>=0.5"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: integration tests > 1 minute"]
```

- [ ] **Step 3: Create `README.md`**

```markdown
# GCN_MA Link Prediction (Reproduction)

Reproduction of *Dynamic network link prediction with node representation learning from graph convolutional networks* (Mei & Zhao, Scientific Reports 2024).

## Setup

Requires Python 3.11 and CUDA 12.1 capable GPU.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Quickstart

```bash
python scripts/download_datasets.py --dataset collegemsg
python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml
```

## Layout

See `docs/superpowers/specs/2026-05-16-gcn-ma-link-prediction-design.md` for design.
```

- [ ] **Step 4: Delete `main.py`** (PyCharm template, unused)

```bash
rm main.py
```

- [ ] **Step 5: Install env**

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -c "import torch; print(torch.cuda.is_available())"
```

Expected: `True` printed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version README.md
git rm main.py
git commit -m "[setup] bootstrap project: deps, README, remove PyCharm template"
```

---

## Task 2: Test infrastructure

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `src/__init__.py`

- [ ] **Step 1: Create empty `src/__init__.py`**

```python
```

- [ ] **Step 2: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_all():
    """Make every test deterministic."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def tiny_graph_edges():
    """5-node graph with known clustering structure.

    Edges: (0,1), (0,2), (1,2), (2,3), (3,4)
    Triangles: {0,1,2}
    """
    return [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
```

- [ ] **Step 4: Verify pytest discovers tests**

```bash
pytest --collect-only
```

Expected: `0 tests collected` (no test files yet), no errors.

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "[tests] bootstrap pytest with shared fixtures"
```

---

## Task 3: Seed utility

**Files:**
- Create: `src/utils/__init__.py`, `src/utils/seed.py`
- Test: `tests/test_seed.py`

- [ ] **Step 1: Write the failing test**

`tests/test_seed.py`:
```python
import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproduces_numpy():
    set_seed(42)
    a = np.random.randn(3)
    set_seed(42)
    b = np.random.randn(3)
    np.testing.assert_array_equal(a, b)


def test_set_seed_reproduces_torch():
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    torch.testing.assert_close(a, b)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_seed.py -v
```

Expected: `ImportError: cannot import name 'set_seed'`.

- [ ] **Step 3: Create `src/utils/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `src/utils/seed.py`**

```python
"""Seed all RNGs for reproducibility."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_seed.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/utils/__init__.py src/utils/seed.py tests/test_seed.py
git commit -m "[utils] add deterministic seeding helper"
```

---

## Task 4: DynamicGraph dataclass + split

**Files:**
- Create: `src/data/__init__.py`, `src/data/base.py`
- Test: `tests/test_dynamic_graph.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dynamic_graph.py`:
```python
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph, temporal_split


def _make_snapshot(num_edges: int, num_nodes: int = 5) -> Data:
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def test_dynamic_graph_fields():
    snaps = [_make_snapshot(3) for _ in range(10)]
    g = DynamicGraph(
        snapshots=snaps,
        node_features=torch.zeros(5, 3),
        num_nodes=5,
        num_time_steps=10,
        dataset_name="dummy",
    )
    assert g.num_time_steps == 10
    assert len(g.snapshots) == 10
    assert g.node_features.shape == (5, 3)


def test_temporal_split_8_1_rest():
    """Spec §6.4: Val target is snapshot ⌊0.8T⌋, test targets are snapshots
    [⌊0.8T⌋+1, T-1]. Returned tuple is (train_end, val_step, test_start) where
    `val_step` is the t value (model sees [0..val_step], predicts val_step+1)
    and `test_start` is the first t for test iteration."""
    train_end, val_step, test_start = temporal_split(num_time_steps=10, train_ratio=0.8)
    # floor(0.8 * 10) = 8
    # val target = snapshot 8 → val_step (t input) = 7
    # test targets = snapshots [9, 9] (i.e. just snapshot 9) → test t = 8
    assert train_end == 8
    assert val_step == 7
    assert test_start == 8


def test_temporal_split_non_round():
    # 47 time steps (CollegeMsg)
    train_end, val_step, test_start = temporal_split(num_time_steps=47, train_ratio=0.8)
    # floor(0.8 * 47) = 37
    # val target = snapshot 37, val_step = 36, test_start = 37
    assert train_end == 37
    assert val_step == 36
    assert test_start == 37
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_dynamic_graph.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/data/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `src/data/base.py`**

```python
"""Standard dynamic-graph container and temporal split helper."""
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class DynamicGraph:
    """A sequence of graph snapshots over time.

    Attributes:
        snapshots: list of PyG Data objects, one per time step.
        node_features: shared node feature matrix [N, F] (may be per-snapshot
            in a separate field — see GCN_MA usage; this is the canonical
            fallback used by baselines).
        num_nodes: N.
        num_time_steps: T = len(snapshots).
        dataset_name: identifier for logging.
    """

    snapshots: list[Data]
    node_features: torch.Tensor
    num_nodes: int
    num_time_steps: int
    dataset_name: str


def temporal_split(num_time_steps: int, train_ratio: float = 0.8) -> tuple[int, int, int]:
    """Compute (train_end, val_step, test_start) per spec §6.4.

    Semantics of returned indices:
        - Training iterates `t in [0, val_step)` and predicts snapshot t+1.
        - Validation uses `t = val_step` (model sees snapshots [0..val_step],
          predicts snapshot val_step+1 == train_end).
        - Test iterates `t in [test_start, num_time_steps - 1)` and predicts t+1.

    Args:
        num_time_steps: T.
        train_ratio: fraction of snapshots reserved for training+val.

    Returns:
        train_end: snapshot index at the train/test boundary (val target).
        val_step: t input for validation (= train_end - 1).
        test_start: first t for test iteration (= train_end).
    """
    train_end = int(num_time_steps * train_ratio)
    val_step = train_end - 1
    test_start = train_end
    return train_end, val_step, test_start
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_dynamic_graph.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/data/__init__.py src/data/base.py tests/test_dynamic_graph.py
git commit -m "[data] DynamicGraph dataclass and temporal split helper"
```

---

## Task 5: NRNAE preprocessing — CC and AS

**Files:**
- Create: `src/data/preprocess.py`
- Test: `tests/test_nrnae.py`

- [ ] **Step 1: Write the failing test**

`tests/test_nrnae.py`:
```python
import networkx as nx
import torch

from src.data.preprocess import aggregation_strength, clustering_coefficient


def _triangle_plus_tail():
    """Nodes 0,1,2 form a triangle; node 3 attached to 2; node 4 attached to 3.

    Degrees: 0:2, 1:2, 2:3, 3:2, 4:1
    Triangles passing through node: 0:1, 1:1, 2:1, 3:0, 4:0
    CC(i) = 2*R(i) / (K(i)*(K(i)-1))
        CC(0) = 2*1 / (2*1) = 1.0
        CC(1) = 2*1 / (2*1) = 1.0
        CC(2) = 2*1 / (3*2) = 0.333...
        CC(3) = 0   (degree 2 but no triangle)
        CC(4) = 0   (degree 1 → undefined; convention 0)
    AS(i) = degree(i) * CC(i)
        AS(0) = 2.0, AS(1) = 2.0, AS(2) = 1.0, AS(3) = 0.0, AS(4) = 0.0
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def test_clustering_coefficient_matches_paper_formula():
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    assert cc.shape == (5,)
    torch.testing.assert_close(cc[0], torch.tensor(1.0))
    torch.testing.assert_close(cc[1], torch.tensor(1.0))
    torch.testing.assert_close(cc[2], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(cc[3], torch.tensor(0.0))
    torch.testing.assert_close(cc[4], torch.tensor(0.0))


def test_aggregation_strength():
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    assert as_.shape == (5,)
    torch.testing.assert_close(as_[0], torch.tensor(2.0))
    torch.testing.assert_close(as_[1], torch.tensor(2.0))
    torch.testing.assert_close(as_[2], torch.tensor(1.0))
    torch.testing.assert_close(as_[3], torch.tensor(0.0))
    torch.testing.assert_close(as_[4], torch.tensor(0.0))


def test_handles_isolated_node():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    cc = clustering_coefficient(G, num_nodes=2)
    assert cc.shape == (2,)
    torch.testing.assert_close(cc, torch.zeros(2))
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_nrnae.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/data/preprocess.py` (partial — CC + AS)**

```python
"""NRNAE preprocessing per paper §3.

For each snapshot:
    CC(i) = 2*R(i) / (K(i)*(K(i)-1))     # clustering coefficient
    AS(i) = degree(i) * CC(i)             # aggregation strength
    S(i,j) = |N(i) ∩ N(j)| * AS(i)        # pairwise aggregation
    Ŝ = A + β·S + I                       # enhanced adjacency
"""
import networkx as nx
import torch


def clustering_coefficient(G: nx.Graph, num_nodes: int) -> torch.Tensor:
    """Return per-node CC as a [num_nodes] tensor.

    Convention: CC = 0 for isolated nodes or nodes with degree < 2.
    """
    cc_dict = nx.clustering(G)
    cc = torch.zeros(num_nodes)
    for i, val in cc_dict.items():
        cc[i] = val
    return cc


def aggregation_strength(G: nx.Graph, cc: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return AS(i) = degree(i) * CC(i) as a [num_nodes] tensor."""
    deg = torch.zeros(num_nodes)
    for i, d in G.degree():
        deg[i] = d
    return deg * cc
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_nrnae.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_nrnae.py
git commit -m "[data] NRNAE: clustering coefficient and aggregation strength"
```

---

## Task 6: NRNAE preprocessing — S and Ŝ

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/test_nrnae.py`

- [ ] **Step 1: Add failing tests for S and Ŝ**

Append to `tests/test_nrnae.py`:
```python
from src.data.preprocess import enhanced_adjacency, pairwise_aggregation


def test_pairwise_aggregation_shape_and_values():
    """For the triangle-plus-tail graph:
    N(0)={1,2}, N(1)={0,2}, N(2)={0,1,3}, N(3)={2,4}, N(4)={3}
    |N(0) ∩ N(1)| = |{2}| = 1
    |N(0) ∩ N(2)| = |{1}| = 1
    |N(1) ∩ N(2)| = |{0}| = 1
    |N(2) ∩ N(3)| = |{}| = 0
    S(i,j) = |N(i)∩N(j)| * AS(i)
        S(0,1) = 1 * 2 = 2
        S(0,2) = 1 * 2 = 2
        S(1,0) = 1 * 2 = 2
        S(1,2) = 1 * 2 = 2
        S(2,0) = 1 * 1 = 1
        S(2,1) = 1 * 1 = 1
    """
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    assert S.shape == (5, 5)
    assert S[0, 1].item() == 2.0
    assert S[1, 0].item() == 2.0
    assert S[2, 0].item() == 1.0
    assert S[0, 0].item() == 0.0  # no self
    assert S[2, 3].item() == 0.0  # no common neighbor


def test_enhanced_adjacency_includes_identity():
    """Ŝ = A + β·S + I."""
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    A = torch.zeros(5, 5)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    S_hat = enhanced_adjacency(A, S, beta=0.8)
    expected_diag = torch.ones(5)  # identity contribution
    torch.testing.assert_close(torch.diag(S_hat), expected_diag)
    # Off-diagonal at (0,1): A=1, S=2, beta=0.8 → 1 + 0.8*2 = 2.6
    torch.testing.assert_close(S_hat[0, 1], torch.tensor(2.6))
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_nrnae.py::test_pairwise_aggregation_shape_and_values -v
```

Expected: `ImportError`.

- [ ] **Step 3: Extend `src/data/preprocess.py`**

Append to the existing file:
```python
def pairwise_aggregation(G: nx.Graph, as_: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute S(i,j) = |N(i) ∩ N(j)| * AS(i) as a dense [N, N] tensor.

    Diagonal is zero (no self-loop contribution).
    """
    S = torch.zeros(num_nodes, num_nodes)
    neighbors: dict[int, set[int]] = {n: set(G.neighbors(n)) for n in G.nodes()}
    for i in G.nodes():
        for j in G.nodes():
            if i == j:
                continue
            common = neighbors.get(i, set()) & neighbors.get(j, set())
            if not common:
                continue
            S[i, j] = len(common) * as_[i].item()
    return S


def enhanced_adjacency(A: torch.Tensor, S: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute Ŝ = A + β·S + I per paper Eq. 5."""
    n = A.shape[0]
    return A + beta * S + torch.eye(n, dtype=A.dtype, device=A.device)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_nrnae.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_nrnae.py
git commit -m "[data] NRNAE: pairwise aggregation S and enhanced adjacency"
```

---

## Task 7: NRNAE — per-snapshot feature pipeline

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/test_nrnae.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_nrnae.py`:
```python
from src.data.preprocess import compute_snapshot_features


def test_compute_snapshot_features_returns_3dim_node_features_and_S_hat():
    """compute_snapshot_features(edges, num_nodes, beta) →
       node_features [N, 3] = [degree, CC, AS], S_hat [N, N] sparse-friendly dense.
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    features, S_hat = compute_snapshot_features(edges, num_nodes=5, beta=0.8)
    assert features.shape == (5, 3)
    # node 2: degree=3, CC=1/3, AS=1
    torch.testing.assert_close(features[2, 0], torch.tensor(3.0))
    torch.testing.assert_close(features[2, 1], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(features[2, 2], torch.tensor(1.0))
    assert S_hat.shape == (5, 5)


def test_compute_snapshot_features_empty_edges():
    """Snapshot with no edges → all zero features, S_hat = I."""
    features, S_hat = compute_snapshot_features([], num_nodes=5, beta=0.8)
    torch.testing.assert_close(features, torch.zeros(5, 3))
    torch.testing.assert_close(S_hat, torch.eye(5))
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_nrnae.py::test_compute_snapshot_features_returns_3dim_node_features_and_S_hat -v
```

Expected: `ImportError`.

- [ ] **Step 3: Extend `src/data/preprocess.py`**

Append:
```python
def compute_snapshot_features(
    edges: list[tuple[int, int]],
    num_nodes: int,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-snapshot node features [N, 3] = [degree, CC, AS] and Ŝ [N, N].

    Args:
        edges: undirected edge list. Duplicates and self-loops are ignored.
        num_nodes: total nodes in the network (constant across snapshots).
        beta: NRNAE mixing factor.

    Returns:
        features: torch.Tensor [N, 3]
        S_hat: torch.Tensor [N, N], dense
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((u, v) for u, v in edges if u != v)

    cc = clustering_coefficient(G, num_nodes)
    as_ = aggregation_strength(G, cc, num_nodes)

    deg = torch.zeros(num_nodes)
    for i, d in G.degree():
        deg[i] = d

    features = torch.stack([deg, cc, as_], dim=1)

    A = torch.zeros(num_nodes, num_nodes)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    S = pairwise_aggregation(G, as_, num_nodes)
    S_hat = enhanced_adjacency(A, S, beta)
    return features, S_hat
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_nrnae.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_nrnae.py
git commit -m "[data] NRNAE: per-snapshot feature computation entry point"
```

---

## Task 8: CollegeMsg dataset loader

**Files:**
- Create: `src/data/loaders/__init__.py`, `src/data/loaders/collegemsg.py`
- Create: `configs/datasets/collegemsg.yaml`
- Test: `tests/test_collegemsg_loader.py`

- [ ] **Step 1: Create `configs/datasets/collegemsg.yaml`**

```yaml
name: collegemsg
raw_url: https://snap.stanford.edu/data/CollegeMsg.txt.gz
raw_filename: CollegeMsg.txt.gz
num_time_steps: 47
beta: 0.8
train_ratio: 0.8
```

- [ ] **Step 2: Write the failing test**

`tests/test_collegemsg_loader.py`:
```python
import gzip
from pathlib import Path

import pytest
import torch

from src.data.loaders.collegemsg import build_dynamic_graph, parse_collegemsg_file


def test_parse_collegemsg_file(tmp_path: Path):
    """CollegeMsg format: src dst timestamp (whitespace separated)."""
    raw = "1 2 1000\n1 3 1100\n2 3 1200\n4 5 1500\n"
    gz_path = tmp_path / "CollegeMsg.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(raw)

    df = parse_collegemsg_file(gz_path)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 4
    assert df.iloc[0].tolist() == [1, 2, 1000]


def test_build_dynamic_graph_produces_snapshots(tmp_path: Path):
    raw = "\n".join(
        [f"{i % 5} {(i + 1) % 5} {1000 + i}" for i in range(20)]
    )
    gz_path = tmp_path / "CollegeMsg.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(raw)

    g = build_dynamic_graph(gz_path, num_time_steps=4, beta=0.8)
    assert g.dataset_name == "collegemsg"
    assert g.num_time_steps == 4
    assert len(g.snapshots) == 4
    assert g.num_nodes == 5
    # Each snapshot has Data fields; node features per snapshot in .x
    for snap in g.snapshots:
        assert snap.x.shape == (5, 3)
        assert hasattr(snap, "S_hat")
        assert snap.S_hat.shape == (5, 5)


def test_node_id_zero_indexed_and_dense(tmp_path: Path):
    """Original CollegeMsg uses 1-indexed sparse IDs; loader must remap to 0..N-1."""
    raw = "10 20 1000\n10 30 1100\n20 30 1200\n"
    gz_path = tmp_path / "CollegeMsg.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(raw)

    g = build_dynamic_graph(gz_path, num_time_steps=1, beta=0.8)
    assert g.num_nodes == 3
    for snap in g.snapshots:
        assert snap.edge_index.max().item() < 3
        assert snap.edge_index.min().item() >= 0
```

- [ ] **Step 3: Run, verify fail**

```bash
pytest tests/test_collegemsg_loader.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Create empty `src/data/loaders/__init__.py`**

```python
```

- [ ] **Step 5: Implement `src/data/loaders/collegemsg.py`**

```python
"""Loader for SNAP CollegeMsg temporal network."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.data.preprocess import compute_snapshot_features


def parse_collegemsg_file(path: Path) -> pd.DataFrame:
    """Parse SNAP CollegeMsg `.txt.gz` (whitespace `src dst ts`) → DataFrame."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
    return df.astype({"src": int, "dst": int, "ts": int})


def _remap_to_dense_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remap arbitrary node IDs to 0..N-1. Returns (remapped_df, N)."""
    unique_ids = sorted(set(df.src.unique()) | set(df.dst.unique()))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    df = df.copy()
    df["src"] = df["src"].map(id_map)
    df["dst"] = df["dst"].map(id_map)
    return df, len(unique_ids)


def _snapshot_bin_edges(ts: pd.Series, num_time_steps: int) -> np.ndarray:
    """Equal-time-window bin edges from t_min to t_max inclusive."""
    t_min, t_max = float(ts.min()), float(ts.max())
    return np.linspace(t_min, t_max + 1e-6, num_time_steps + 1)


def build_dynamic_graph(
    raw_gz_path: Path,
    num_time_steps: int,
    beta: float,
) -> DynamicGraph:
    """Parse CollegeMsg gzip file and produce a fully-preprocessed DynamicGraph.

    Each snapshot's `.x` contains [degree, CC, AS] per node and `.S_hat`
    contains the enhanced adjacency from NRNAE.
    """
    df = parse_collegemsg_file(raw_gz_path)
    df, num_nodes = _remap_to_dense_ids(df)

    bins = _snapshot_bin_edges(df.ts, num_time_steps)
    snapshots: list[Data] = []
    for t in range(num_time_steps):
        mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
        sub = df.loc[mask, ["src", "dst"]].values
        edges_list = [(int(u), int(v)) for u, v in sub]

        features, S_hat = compute_snapshot_features(edges_list, num_nodes, beta)
        if len(edges_list) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data.x = features
        data.S_hat = S_hat
        snapshots.append(data)

    # node_features as fallback: one-hot identity (used by baselines, not GCN_MA)
    return DynamicGraph(
        snapshots=snapshots,
        node_features=torch.eye(num_nodes),
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        dataset_name="collegemsg",
    )
```

- [ ] **Step 6: Run, verify pass**

```bash
pytest tests/test_collegemsg_loader.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/__init__.py src/data/loaders/collegemsg.py configs/datasets/collegemsg.yaml tests/test_collegemsg_loader.py
git commit -m "[data] CollegeMsg loader with per-snapshot NRNAE features"
```

---

## Task 9: Dataset download script

**Files:**
- Create: `scripts/download_datasets.py`

This is a utility script — no TDD, manual verification.

- [ ] **Step 1: Implement `scripts/download_datasets.py`**

```python
"""Download dataset raw files from SNAP / other sources with retry.

Usage:
    python scripts/download_datasets.py --dataset collegemsg
"""
import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_config(name: str) -> dict:
    path = REPO_ROOT / "configs" / "datasets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def download_with_retry(url: str, dest: Path, max_attempts: int = 3) -> None:
    """Stream-download URL → dest with up to max_attempts retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[{attempt}/{max_attempts}] Downloading {url} → {dest}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            print(f"OK ({dest.stat().st_size} bytes)")
            return
        except Exception as e:
            print(f"  failed: {e}")
            if attempt < max_attempts:
                time.sleep(2**attempt)
    print(f"ERROR: failed after {max_attempts} attempts", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset name (e.g. collegemsg)")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)
    raw_dir = REPO_ROOT / "data" / "raw" / args.dataset
    dest = raw_dir / cfg["raw_filename"]
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    download_with_retry(cfg["raw_url"], dest)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test**

```bash
python scripts/download_datasets.py --dataset collegemsg
ls -la data/raw/collegemsg/
```

Expected: file `data/raw/collegemsg/CollegeMsg.txt.gz` ~250 KB.

- [ ] **Step 3: Verify loader works on real data**

```bash
python -c "
from pathlib import Path
from src.data.loaders.collegemsg import build_dynamic_graph
g = build_dynamic_graph(Path('data/raw/collegemsg/CollegeMsg.txt.gz'), num_time_steps=47, beta=0.8)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
print(f'first snapshot x.shape={g.snapshots[0].x.shape}, S_hat.shape={g.snapshots[0].S_hat.shape}')
"
```

Expected output similar to:
```
nodes=1899, T=47, edges_total=59835
first snapshot x.shape=torch.Size([1899, 3]), S_hat.shape=torch.Size([1899, 1899])
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_datasets.py
git commit -m "[scripts] CollegeMsg downloader with retry"
```

---

## Task 10: Negative sampling

**Files:**
- Create: `src/training/__init__.py`, `src/training/negative_sampling.py`
- Test: `tests/test_negative_sampling.py`

- [ ] **Step 1: Write the failing test**

`tests/test_negative_sampling.py`:
```python
import torch

from src.training.negative_sampling import sample_negative_edges


def test_returns_correct_count():
    pos = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges
    neg = sample_negative_edges(pos, num_nodes=10, num_samples=3, seed=42)
    assert neg.shape == (2, 3)


def test_no_self_loops():
    pos = torch.tensor([[0, 1], [1, 2]])
    neg = sample_negative_edges(pos, num_nodes=10, num_samples=100, seed=42)
    assert (neg[0] != neg[1]).all()


def test_does_not_contain_positive_edges():
    pos = torch.tensor([[0, 1, 2], [1, 2, 3]])
    pos_set = {(int(u), int(v)) for u, v in pos.t().tolist()}
    neg = sample_negative_edges(pos, num_nodes=20, num_samples=100, seed=42)
    neg_set = {(int(u), int(v)) for u, v in neg.t().tolist()}
    assert neg_set.isdisjoint(pos_set)


def test_deterministic_with_seed():
    pos = torch.tensor([[0, 1], [2, 3]])
    a = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    b = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    pos = torch.tensor([[0, 1], [2, 3]])
    a = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    b = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=999)
    assert not torch.equal(a, b)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_negative_sampling.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create empty `src/training/__init__.py`**

```python
```

- [ ] **Step 4: Implement `src/training/negative_sampling.py`**

```python
"""Uniform negative edge sampling with rejection.

Rejects edges that are self-loops or appear in the positive set.
"""
import torch


def sample_negative_edges(
    pos_edges: torch.Tensor,
    num_nodes: int,
    num_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample `num_samples` negative edges as a [2, num_samples] tensor.

    Args:
        pos_edges: [2, P] positive edges to exclude.
        num_nodes: vocabulary size for sampling.
        num_samples: how many negatives to return.
        seed: per-call RNG seed for reproducibility (None to use ambient).

    Returns:
        neg_edges: [2, num_samples] long tensor.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    pos_set: set[tuple[int, int]] = {
        (int(u), int(v)) for u, v in pos_edges.t().tolist()
    }

    sampled: list[tuple[int, int]] = []
    max_attempts = num_samples * 20
    attempts = 0
    while len(sampled) < num_samples and attempts < max_attempts:
        batch = num_samples - len(sampled)
        candidates = torch.randint(0, num_nodes, (2, batch * 2), generator=generator)
        for k in range(candidates.shape[1]):
            u, v = int(candidates[0, k]), int(candidates[1, k])
            if u == v:
                continue
            if (u, v) in pos_set:
                continue
            sampled.append((u, v))
            if len(sampled) == num_samples:
                break
        attempts += 1

    if len(sampled) < num_samples:
        raise RuntimeError(
            f"Could not sample {num_samples} negatives after {max_attempts} attempts "
            f"(got {len(sampled)}). Graph too dense?"
        )

    return torch.tensor(sampled, dtype=torch.long).t().contiguous()
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_negative_sampling.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/training/__init__.py src/training/negative_sampling.py tests/test_negative_sampling.py
git commit -m "[training] uniform-rejection negative edge sampling"
```

---

## Task 11: Metrics (AUC, AP)

**Files:**
- Create: `src/eval/__init__.py`, `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest

from src.eval.metrics import compute_ap, compute_auc, compute_link_prediction_metrics


def test_auc_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc(y_true, y_score) == pytest.approx(1.0)


def test_auc_random_balanced():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.random(size=1000)
    auc = compute_auc(y_true, y_score)
    assert 0.4 < auc < 0.6  # near 0.5


def test_ap_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_ap(y_true, y_score) == pytest.approx(1.0)


def test_compute_link_prediction_metrics_returns_dict():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.6])
    metrics = compute_link_prediction_metrics(y_true, y_score)
    assert set(metrics.keys()) == {"auc", "ap"}
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create empty `src/eval/__init__.py`**

```python
```

- [ ] **Step 4: Implement `src/eval/metrics.py`**

```python
"""Link-prediction metrics (AUC, AP) via sklearn."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def compute_link_prediction_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    return {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_ap(y_true, y_score),
    }
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/eval/__init__.py src/eval/metrics.py tests/test_metrics.py
git commit -m "[eval] AUC and AP metrics via sklearn"
```

---

## Task 12: Model base interface

**Files:**
- Create: `src/models/__init__.py`, `src/models/base.py`

No TDD — defining an ABC. Verified by usage in subsequent tasks.

- [ ] **Step 1: Create empty `src/models/__init__.py`**

```python
```

- [ ] **Step 2: Implement `src/models/base.py`**

```python
"""Abstract base class for dynamic link-prediction models."""
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.data import Data


class DynamicLinkPredictor(nn.Module, ABC):
    """Common interface for GCN_MA and baselines.

    Subclasses implement `forward(snapshots, t)` returning node embeddings Z^t.
    `predict_link` is shared and may be overridden.
    """

    @abstractmethod
    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        """Return Z^t ∈ R^{N×D} given snapshots [0..time_step]."""

    def predict_link(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Default decoder: dot product of source and target embeddings.

        Subclasses with a learned MLP decoder should override and call their
        internal decoder. This default is for sanity testing only.

        Args:
            Z: [N, D] embeddings.
            edges: [2, E] edge index.
        Returns:
            logits: [E] tensor.
        """
        src, dst = edges[0], edges[1]
        return (Z[src] * Z[dst]).sum(dim=-1)
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from src.models.base import DynamicLinkPredictor; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/models/__init__.py src/models/base.py
git commit -m "[models] DynamicLinkPredictor abstract interface"
```

---

## Task 13: GCN layer with enhanced adjacency

**Files:**
- Create: `src/models/gcn_ma/__init__.py`, `src/models/gcn_ma/gcn_layer.py`
- Test: `tests/test_models_smoke.py` (start)

- [ ] **Step 1: Write the failing smoke test**

`tests/test_models_smoke.py`:
```python
import torch

from src.models.gcn_ma.gcn_layer import EnhancedGCNLayer


def test_gcn_layer_output_shape():
    N, F_in, D = 10, 3, 8
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
    W = torch.randn(F_in, D, requires_grad=True)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    assert H.shape == (N, D)


def test_gcn_layer_gradient_flows():
    N, F_in, D = 10, 3, 8
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
    W = torch.randn(F_in, D, requires_grad=True)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    loss = H.sum()
    loss.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_gcn_layer_handles_isolated_nodes():
    """Zero-degree node should not produce NaN (degree normalization edge case)."""
    N, F_in, D = 5, 3, 4
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N)  # only identity → all "degree 1" via self-loop
    W = torch.randn(F_in, D)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    assert torch.isfinite(H).all()
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create empty `src/models/gcn_ma/__init__.py`**

```python
```

- [ ] **Step 4: Implement `src/models/gcn_ma/gcn_layer.py`**

```python
"""Spectral GCN layer with NRNAE-enhanced adjacency.

H^t = ReLU(D̂^(-1/2) · Ŝ^t · D̂^(-1/2) · X^t · W^t)

Where Ŝ^t comes from preprocessing (Ŝ = A + β·S + I) and D̂ is the row-sum
degree matrix of Ŝ.
"""
import torch
from torch import nn


class EnhancedGCNLayer(nn.Module):
    """Stateless spectral GCN layer. Weights W^t are provided externally
    (the LSTM weight updater owns them).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, X: torch.Tensor, S_hat: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        """X: [N, F], S_hat: [N, N], W: [F, D] → H: [N, D]."""
        deg = S_hat.sum(dim=1).clamp(min=self.eps)
        d_inv_sqrt = deg.pow(-0.5)
        # Symmetric normalization
        S_norm = d_inv_sqrt.unsqueeze(1) * S_hat * d_inv_sqrt.unsqueeze(0)
        H = S_norm @ X @ W
        return torch.relu(H)
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models/gcn_ma/__init__.py src/models/gcn_ma/gcn_layer.py tests/test_models_smoke.py
git commit -m "[models] GCN_MA: spectral GCN layer with enhanced adjacency"
```

---

## Task 14: LSTM weight updater

**Files:**
- Create: `src/models/gcn_ma/lstm_weight.py`
- Modify: `tests/test_models_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_models_smoke.py`:
```python
from src.models.gcn_ma.lstm_weight import LSTMWeightUpdater


def test_lstm_weight_updater_preserves_shape():
    F_in, D = 3, 8
    updater = LSTMWeightUpdater(in_features=F_in, out_features=D)
    W_t = torch.randn(F_in, D)
    h_t, c_t = updater.init_state(W_t.device)
    W_next, h_next, c_next = updater(W_t, h_t, c_t)
    assert W_next.shape == (F_in, D)
    assert h_next.shape == h_t.shape
    assert c_next.shape == c_t.shape


def test_lstm_weight_updater_gradients_flow():
    F_in, D = 3, 4
    updater = LSTMWeightUpdater(in_features=F_in, out_features=D)
    W_t = torch.randn(F_in, D, requires_grad=True)
    h_t, c_t = updater.init_state(W_t.device)
    W_next, _, _ = updater(W_t, h_t, c_t)
    loss = W_next.sum()
    loss.backward()
    assert W_t.grad is not None
    assert torch.isfinite(W_t.grad).all()
    # The LSTM cell should also accumulate grads on its own params
    for p in updater.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_models_smoke.py::test_lstm_weight_updater_preserves_shape -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/gcn_ma/lstm_weight.py`**

```python
"""LSTM-based weight evolver for GCN_MA.

W^t = LSTMCell(flatten(W^{t-1}), state^{t-1})

Treats the weight matrix W ∈ R^{F×D} as a flattened vector of size F*D.
"""
import torch
from torch import nn


class LSTMWeightUpdater(nn.Module):
    """LSTM cell that evolves a [F, D] weight matrix across time steps."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = in_features * out_features
        self.cell = nn.LSTMCell(input_size=self.dim, hidden_size=self.dim)

    def init_state(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, self.dim, device=device)
        c = torch.zeros(1, self.dim, device=device)
        return h, c

    def forward(
        self, W: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = W.reshape(1, self.dim)
        h_next, c_next = self.cell(flat, (h, c))
        W_next = h_next.reshape(self.in_features, self.out_features)
        return W_next, h_next, c_next
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/gcn_ma/lstm_weight.py tests/test_models_smoke.py
git commit -m "[models] GCN_MA: LSTM weight updater"
```

---

## Task 15: Multi-head self-attention

**Files:**
- Create: `src/models/gcn_ma/attention.py`
- Modify: `tests/test_models_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_models_smoke.py`:
```python
from src.models.gcn_ma.attention import MultiHeadSelfAttention


def test_attention_preserves_shape():
    N, D = 10, 16
    attn = MultiHeadSelfAttention(embed_dim=D, num_heads=4, dropout=0.0)
    H = torch.randn(N, D)
    Z = attn(H)
    assert Z.shape == (N, D)


def test_attention_embed_dim_divisible_by_heads_validated():
    import pytest
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=10, num_heads=3, dropout=0.0)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_models_smoke.py::test_attention_preserves_shape -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/gcn_ma/attention.py`**

```python
"""Multi-head self-attention with residual + LayerNorm."""
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention applied to a single snapshot's embeddings.

    Z = LayerNorm(H + MultiHead(H, H, H))
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: [N, D] → Z: [N, D]."""
        h_batched = H.unsqueeze(0)  # [1, N, D]
        attn_out, _ = self.mha(h_batched, h_batched, h_batched)
        z = self.norm(h_batched + attn_out)
        return z.squeeze(0)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/gcn_ma/attention.py tests/test_models_smoke.py
git commit -m "[models] GCN_MA: multi-head self-attention"
```

---

## Task 16: Link decoder MLP

**Files:**
- Create: `src/models/gcn_ma/link_decoder.py`
- Modify: `tests/test_models_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_models_smoke.py`:
```python
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


def test_link_decoder_output_shape():
    N, D, E = 10, 16, 7
    Z = torch.randn(N, D)
    edges = torch.randint(0, N, (2, E))
    decoder = LinkDecoderMLP(embed_dim=D, hidden_dim=D, dropout=0.0)
    logits = decoder(Z, edges)
    assert logits.shape == (E,)


def test_link_decoder_logit_not_probability():
    """Decoder returns raw logits, NOT sigmoid (we use BCEWithLogitsLoss)."""
    Z = torch.randn(5, 4)
    edges = torch.tensor([[0, 1], [2, 3]])
    decoder = LinkDecoderMLP(embed_dim=4, hidden_dim=4, dropout=0.0)
    logits = decoder(Z, edges)
    # If output were already sigmoid'd we'd expect all to be in [0,1].
    # A 2-layer MLP with random init can produce values outside this range.
    # Statistically over 100 calls at least one should be outside [0,1].
    found_outside = False
    for _ in range(100):
        decoder = LinkDecoderMLP(embed_dim=4, hidden_dim=4, dropout=0.0)
        out = decoder(Z, edges)
        if (out < 0).any() or (out > 1).any():
            found_outside = True
            break
    assert found_outside, "decoder appears to apply sigmoid internally"
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_models_smoke.py::test_link_decoder_output_shape -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/models/gcn_ma/link_decoder.py`**

```python
"""MLP link decoder.

logit(u, v) = MLP_2-layer([Z[u] ⊕ Z[v]])
"""
import torch
from torch import nn


class LinkDecoderMLP(nn.Module):
    """2-layer MLP returning raw logits over edge pairs."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Z: [N, D], edges: [2, E] → logits: [E]."""
        src, dst = edges[0], edges[1]
        pair = torch.cat([Z[src], Z[dst]], dim=-1)
        return self.net(pair).squeeze(-1)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/gcn_ma/link_decoder.py tests/test_models_smoke.py
git commit -m "[models] GCN_MA: MLP link decoder"
```

---

## Task 17: GCN_MA composition

**Files:**
- Create: `src/models/gcn_ma/model.py`, `configs/models/gcn_ma.yaml`
- Modify: `tests/test_models_smoke.py`

- [ ] **Step 1: Create `configs/models/gcn_ma.yaml`**

```yaml
name: gcn_ma
feat_dim: 3            # [degree, CC, AS]
hidden_dim: 128
num_heads: 8
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 200
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

- [ ] **Step 2: Add failing smoke test for full model**

Append to `tests/test_models_smoke.py`:
```python
from torch_geometric.data import Data

from src.models.gcn_ma.model import GCN_MA


def _make_dummy_snapshots(N: int, T: int, F_in: int = 3) -> list[Data]:
    snaps = []
    for _ in range(T):
        d = Data(edge_index=torch.randint(0, N, (2, N * 2)), num_nodes=N)
        d.x = torch.randn(N, F_in)
        d.S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
        snaps.append(d)
    return snaps


def test_gcn_ma_forward_shape():
    N, T, D = 8, 5, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    assert Z.shape == (N, D)


def test_gcn_ma_gradient_flows():
    N, T, D = 8, 4, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_gcn_ma_predict_link():
    N, T, D = 8, 3, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    edges = torch.tensor([[0, 1, 2], [3, 4, 5]])
    logits = model.predict_link(Z, edges)
    assert logits.shape == (3,)
```

- [ ] **Step 3: Run, verify fail**

```bash
pytest tests/test_models_smoke.py::test_gcn_ma_forward_shape -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/models/gcn_ma/model.py`**

```python
"""GCN_MA: composition of NRNAE-enhanced GCN, LSTM weight evolver,
multi-head self-attention, and an MLP link decoder.

Per paper §3:
    H^t = GCNLayer(X^t, Ŝ^t, W^t)
    W^t = LSTMCell(W^{t-1})
    Z^t = MultiHeadSelfAttn(H^t)
    P^t = σ(MLP([Z^t[u] ⊕ Z^t[v]]))
"""
import math

import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.attention import MultiHeadSelfAttention
from src.models.gcn_ma.gcn_layer import EnhancedGCNLayer
from src.models.gcn_ma.link_decoder import LinkDecoderMLP
from src.models.gcn_ma.lstm_weight import LSTMWeightUpdater


class GCN_MA(DynamicLinkPredictor):
    def __init__(
        self,
        feat_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.gcn = EnhancedGCNLayer()
        self.lstm_w = LSTMWeightUpdater(in_features=feat_dim, out_features=hidden_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Initial W^0 via Xavier
        bound = 1.0 / math.sqrt(feat_dim)
        self.W0 = nn.Parameter(torch.empty(feat_dim, hidden_dim).uniform_(-bound, bound))

    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        """Run the model up to and including snapshot `time_step`.

        Returns Z^{time_step} ∈ R^{N×D}.
        """
        device = self.W0.device
        W = self.W0
        h, c = self.lstm_w.init_state(device)

        H = None
        for tau in range(time_step + 1):
            snap = snapshots[tau]
            X = snap.x.to(device)
            S_hat = snap.S_hat.to(device)
            H = self.gcn(X, S_hat, W)
            if tau < time_step:
                W, h, c = self.lstm_w(W, h, c)

        # apply attention only on the final snapshot's embeddings
        assert H is not None
        return self.attn(H)

    def predict_link(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z, edges)
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_models_smoke.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models/gcn_ma/model.py configs/models/gcn_ma.yaml tests/test_models_smoke.py
git commit -m "[models] GCN_MA: full model composition + config"
```

---

## Task 18: BCE loss wrapper

**Files:**
- Create: `src/training/losses.py`

Thin wrapper around `BCEWithLogitsLoss`. No new tests — exercised by trainer integration.

- [ ] **Step 1: Implement `src/training/losses.py`**

```python
"""Binary cross-entropy loss with logits for link prediction."""
import torch
from torch import nn


def link_prediction_loss(
    pos_logits: torch.Tensor, neg_logits: torch.Tensor
) -> torch.Tensor:
    """Compute BCE-with-logits over concatenated positive and negative logits.

    Args:
        pos_logits: [P] logits for positive edges.
        neg_logits: [Q] logits for negative edges.
    Returns:
        scalar loss.
    """
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat(
        [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]
    )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.training.losses import link_prediction_loss; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/training/losses.py
git commit -m "[training] BCE-with-logits loss wrapper"
```

---

## Task 19: Evaluator

**Files:**
- Create: `src/eval/evaluator.py`
- Test: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluator.py`:
```python
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.eval.evaluator import evaluate_dynamic
from src.models.base import DynamicLinkPredictor


class _ConstantModel(DynamicLinkPredictor):
    """Returns deterministic embeddings independent of input.

    With identity embeddings and dot-product decoder, AUC ≈ 0.5 on random
    positives/negatives.
    """

    def __init__(self, num_nodes: int, dim: int = 4):
        super().__init__()
        self.Z = torch.nn.Parameter(torch.randn(num_nodes, dim), requires_grad=False)

    def forward(self, snapshots, time_step):
        return self.Z


def _dummy_graph(N: int, T: int) -> DynamicGraph:
    snaps = []
    for _ in range(T):
        d = Data(edge_index=torch.randint(0, N, (2, N)), num_nodes=N)
        d.x = torch.randn(N, 3)
        d.S_hat = torch.eye(N)
        snaps.append(d)
    return DynamicGraph(
        snapshots=snaps,
        node_features=torch.eye(N),
        num_nodes=N,
        num_time_steps=T,
        dataset_name="dummy",
    )


def test_evaluator_returns_auc_and_ap():
    N, T = 20, 5
    graph = _dummy_graph(N, T)
    model = _ConstantModel(N, dim=4)
    test_pairs = {
        t + 1: {
            "pos": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "neg": torch.tensor([[6, 7, 8], [9, 10, 11]]),
        }
        for t in range(T - 2, T - 1)  # one test step
    }
    metrics = evaluate_dynamic(
        model, graph, time_steps=[T - 2], test_pairs=test_pairs
    )
    assert "auc" in metrics and "ap" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0


def test_evaluator_pools_across_time_steps():
    """A single AUC computed over concatenated scores from all evaluated steps."""
    N, T = 20, 5
    graph = _dummy_graph(N, T)
    model = _ConstantModel(N, dim=4)
    test_pairs = {
        t + 1: {
            "pos": torch.tensor([[0, 1], [2, 3]]),
            "neg": torch.tensor([[4, 5], [6, 7]]),
        }
        for t in range(T - 3, T - 1)  # two test steps
    }
    metrics = evaluate_dynamic(
        model, graph, time_steps=list(range(T - 3, T - 1)), test_pairs=test_pairs
    )
    assert "auc" in metrics
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_evaluator.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/eval/evaluator.py`**

```python
"""Pooled link-prediction evaluator over a list of test time steps."""
import numpy as np
import torch

from src.data.base import DynamicGraph
from src.eval.metrics import compute_link_prediction_metrics
from src.models.base import DynamicLinkPredictor

TestPairs = dict[int, dict[str, torch.Tensor]]
# test_pairs[t+1] = {"pos": [2, P], "neg": [2, Q]}


@torch.no_grad()
def evaluate_dynamic(
    model: DynamicLinkPredictor,
    graph: DynamicGraph,
    time_steps: list[int],
    test_pairs: TestPairs,
) -> dict[str, float]:
    """Evaluate the model on each step t ∈ time_steps; pool scores across all
    steps before computing AUC and AP.

    At step t, the model sees snapshots [0..t] and predicts edges of
    snapshot t+1 stored in test_pairs[t+1].
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for t in time_steps:
        Z = model(graph.snapshots, time_step=t)
        pairs = test_pairs[t + 1]
        pos_logits = model.predict_link(Z, pairs["pos"])
        neg_logits = model.predict_link(Z, pairs["neg"])
        scores = torch.sigmoid(torch.cat([pos_logits, neg_logits])).cpu().numpy()
        labels = np.concatenate(
            [
                np.ones(pairs["pos"].shape[1], dtype=np.int64),
                np.zeros(pairs["neg"].shape[1], dtype=np.int64),
            ]
        )
        all_scores.append(scores)
        all_labels.append(labels)

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    return compute_link_prediction_metrics(y_true, y_score)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_evaluator.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/evaluator.py tests/test_evaluator.py
git commit -m "[eval] pooled dynamic-LP evaluator"
```

---

## Task 20: Trainer

**Files:**
- Create: `src/training/trainer.py`

The trainer is exercised by the end-to-end run in Task 22. Adding a unit smoke test here would require so many fixtures that the integration test is more valuable.

- [ ] **Step 1: Implement `src/training/trainer.py`**

```python
"""Training loop for dynamic link prediction."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.base import DynamicGraph, temporal_split
from src.eval.evaluator import evaluate_dynamic
from src.models.base import DynamicLinkPredictor
from src.training.losses import link_prediction_loss
from src.training.negative_sampling import sample_negative_edges


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stop_patience: int = 20
    grad_clip_max_norm: float = 5.0
    neg_sampling_seed_base: int = 0


def train_dynamic(
    model: DynamicLinkPredictor,
    graph: DynamicGraph,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict:
    """Train `model` on `graph` and return the best validation result.

    Returns:
        {
            "best_val_auc": float,
            "best_val_ap": float,
            "best_epoch": int,
            "history": list[dict] (per-epoch loss + val_auc),
        }
    """
    train_end, val_step, _ = temporal_split(graph.num_time_steps, train_ratio=0.8)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)

    # Validation: predict snapshot val_step+1 (== train_end) given snapshots [0..val_step].
    val_pos = graph.snapshots[val_step + 1].edge_index
    val_neg = sample_negative_edges(
        val_pos, num_nodes=graph.num_nodes,
        num_samples=val_pos.shape[1], seed=config.neg_sampling_seed_base + 99,
    )
    val_pairs = {val_step + 1: {"pos": val_pos, "neg": val_neg}}

    history: list[dict] = []
    best_val_auc = -1.0
    best_val_ap = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        # Training predicts snapshots [1..val_step] from inputs [0..val_step-1].
        # Never targets snapshots train_end..T-1 (those belong to val/test).
        for t in tqdm(range(val_step), desc=f"Epoch {epoch}", leave=False):
            if t + 1 >= graph.num_time_steps:
                break
            Z_t = model(graph.snapshots, time_step=t)
            pos = graph.snapshots[t + 1].edge_index.to(device)
            if pos.shape[1] == 0:
                continue
            neg = sample_negative_edges(
                pos, num_nodes=graph.num_nodes,
                num_samples=pos.shape[1],
                seed=config.neg_sampling_seed_base + epoch * 1000 + t,
            ).to(device)

            pos_logits = model.predict_link(Z_t, pos)
            neg_logits = model.predict_link(Z_t, neg)
            loss = link_prediction_loss(pos_logits, neg_logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip_max_norm
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        val_metrics = evaluate_dynamic(
            model, graph, time_steps=[val_step], test_pairs=val_pairs
        )
        history.append(
            {"epoch": epoch, "loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        )
        print(
            f"Epoch {epoch:3d}: loss={avg_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_ap={val_metrics['ap']:.4f}"
        )

        scheduler.step(val_metrics["auc"])
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_ap = val_metrics["ap"]
            best_epoch = epoch
            patience = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_auc": best_val_auc},
                    checkpoint_path,
                )
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience}).")
                break

    return {
        "best_val_auc": best_val_auc,
        "best_val_ap": best_val_ap,
        "best_epoch": best_epoch,
        "history": history,
    }
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.training.trainer import train_dynamic, TrainConfig; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer.py
git commit -m "[training] dynamic LP trainer with early stop and checkpointing"
```

---

## Task 21: CLI train entrypoint

**Files:**
- Create: `scripts/train.py`, `configs/experiments/gcn_ma_collegemsg.yaml`
- Create: `src/utils/logging.py`

- [ ] **Step 1: Create `configs/experiments/gcn_ma_collegemsg.yaml`**

```yaml
experiment_name: gcn_ma_collegemsg
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 2: Implement `src/utils/logging.py`**

```python
"""Append-only JSONL metrics writer."""
import json
from pathlib import Path
from typing import Any


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 3: Implement `scripts/train.py`**

```python
"""CLI: train one model on one dataset with one seed.

Usage:
    python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.data.loaders.collegemsg import build_dynamic_graph
from src.eval.evaluator import evaluate_dynamic
from src.models.gcn_ma.model import GCN_MA
from src.training.negative_sampling import sample_negative_edges
from src.training.trainer import TrainConfig, train_dynamic
from src.utils.logging import append_metrics
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _config_hash(*configs: dict) -> str:
    blob = json.dumps(configs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_test_pairs(graph, test_start: int, seed: int):
    """Build pos/neg test pairs for each test time step.

    Iterates t in [test_start, T-1). At each t, model sees snapshots [0..t]
    and predicts snapshot t+1. So this yields test_pairs keyed by t+1, i.e.
    test target snapshots [test_start+1, ..., T-1].
    """
    test_pairs = {}
    for t in range(test_start, graph.num_time_steps - 1):
        pos = graph.snapshots[t + 1].edge_index
        if pos.shape[1] == 0:
            continue
        neg = sample_negative_edges(
            pos, num_nodes=graph.num_nodes,
            num_samples=pos.shape[1], seed=seed + t,
        )
        test_pairs[t + 1] = {"pos": pos, "neg": neg}
    return test_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    exp = _load_yaml(args.config)
    ds_cfg = _load_yaml(REPO_ROOT / exp["dataset_config"])
    model_cfg = _load_yaml(REPO_ROOT / exp["model_config"])
    config_hash = _config_hash(exp, ds_cfg, model_cfg)

    set_seed(exp["seed"])
    device = torch.device(exp["device"] if torch.cuda.is_available() else "cpu")

    # Load dataset
    raw_path = REPO_ROOT / "data" / "raw" / ds_cfg["name"] / ds_cfg["raw_filename"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Run: python scripts/download_datasets.py --dataset {ds_cfg['name']}"
        )
    graph = build_dynamic_graph(
        raw_path,
        num_time_steps=ds_cfg["num_time_steps"],
        beta=ds_cfg["beta"],
    )

    # Model
    model = GCN_MA(
        feat_dim=model_cfg["feat_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    )

    # Train
    train_cfg = TrainConfig(
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        epochs=model_cfg["epochs"],
        early_stop_patience=model_cfg["early_stop_patience"],
        grad_clip_max_norm=model_cfg["grad_clip_max_norm"],
        neg_sampling_seed_base=exp["seed"],
    )
    ckpt = REPO_ROOT / exp["checkpoint_dir"] / f"{exp['experiment_name']}_seed{exp['seed']}_best.pt"

    t0 = time.time()
    train_result = train_dynamic(model, graph, train_cfg, device, checkpoint_path=ckpt)

    # Load best checkpoint and evaluate on test
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    from src.data.base import temporal_split
    _, _, test_start = temporal_split(graph.num_time_steps, train_ratio=0.8)
    test_pairs = _build_test_pairs(graph, test_start, seed=999)
    test_time_steps = [t - 1 for t in sorted(test_pairs.keys())]
    test_metrics = evaluate_dynamic(model, graph, test_time_steps, test_pairs)
    runtime_s = time.time() - t0

    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "experiment_name": exp["experiment_name"],
        "model": "gcn_ma",
        "dataset": ds_cfg["name"],
        "seed": exp["seed"],
        "auc": test_metrics["auc"],
        "ap": test_metrics["ap"],
        "val_auc": train_result["best_val_auc"],
        "val_ap": train_result["best_val_ap"],
        "best_epoch": train_result["best_epoch"],
        "runtime_s": runtime_s,
        "config_hash": config_hash,
        "git_sha": _git_sha(),
    }
    print(json.dumps(record, indent=2))
    append_metrics(REPO_ROOT / exp["metrics_path"], record)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Sanity-check imports**

```bash
python -c "import scripts.train" 2>&1 | tail -3
```

Expected: no error, or one ImportError indicating missing module that we can fix.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py configs/experiments/gcn_ma_collegemsg.yaml src/utils/logging.py
git commit -m "[scripts] CLI train.py with metrics.jsonl logging"
```

---

## Task 22: End-to-end smoke run on CollegeMsg

This is the Tier-3 integration test: run the full pipeline with reduced epochs to verify the system works before launching the real run.

- [ ] **Step 1: Create a smoke-test experiment config**

`configs/experiments/gcn_ma_collegemsg_smoke.yaml`:
```yaml
experiment_name: gcn_ma_collegemsg_smoke
seed: 42
dataset_config: configs/datasets/collegemsg.yaml
model_config: configs/models/gcn_ma_smoke.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics_smoke.jsonl
```

`configs/models/gcn_ma_smoke.yaml` (same shape as gcn_ma.yaml, just fewer epochs):
```yaml
name: gcn_ma
feat_dim: 3
hidden_dim: 64
num_heads: 4
dropout: 0.1
lr: 1.0e-3
weight_decay: 1.0e-5
optimizer: adam
epochs: 3
early_stop_patience: 20
grad_clip_max_norm: 5.0
```

- [ ] **Step 2: Ensure CollegeMsg is downloaded**

```bash
python scripts/download_datasets.py --dataset collegemsg
```

Expected: file exists at `data/raw/collegemsg/CollegeMsg.txt.gz`.

- [ ] **Step 3: Run smoke training**

```bash
python scripts/train.py --config configs/experiments/gcn_ma_collegemsg_smoke.yaml
```

Expected:
- Runs through 3 epochs without crashing.
- Prints per-epoch loss and val_auc.
- Final JSON record printed with `auc` and `ap` in `[0.0, 1.0]`.
- File `results/metrics_smoke.jsonl` contains exactly 1 line.

If it crashes, investigate and fix before proceeding. Likely categories of issue:
- CUDA OOM → reduce `hidden_dim` further.
- NaN loss → check NRNAE preprocessing for divide-by-zero (look at the test snapshot before training).
- Tensor shape mismatch → re-run `pytest tests/test_models_smoke.py` to localize.

- [ ] **Step 4: Run the full training**

```bash
python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml
```

Expected:
- Trains up to 200 epochs or early stops.
- Final test AUC roughly in the ballpark of the paper's 91.49% (acceptable range: 88-94% — exact match unlikely on first try).
- `results/metrics.jsonl` contains 1 line with the result.

- [ ] **Step 5: Document deviations in `docs/reproduction-log.md`**

Create the file with initial entries:
```markdown
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

### Known limitations

- Hyperparameters tuned on CollegeMsg only; later plans will validate on additional datasets.
- TBPTT not yet implemented; will be added when reaching EUT (T=127) in Plan 2.
```

`docs/paper-notes.md`:
```markdown
# Paper notes — Mei & Zhao 2024

DOI: 10.1038/s41598-023-50977-6

## Equations

### NRNAE
- Clustering coefficient: `CC(i) = 2·R(i) / (K(i)·(K(i)-1))`
- Aggregation strength: `AS(i) = degree(i) · CC(i)`
- Pairwise aggregation: `S(i,j) = |N(i) ∩ N(j)| · AS(i)`
- Enhanced adjacency: `Ŝ = A + β·S + I`, β ∈ [0.7, 0.9] reported optimal

### Spectral GCN
`H^t = σ(D̂^(-1/2) · Ŝ^t · D̂^(-1/2) · X^t · W^t)`

### LSTM weight evolution
`W^t = LSTMCell(W^{t-1}, state^{t-1})`

### Attention
`Z^t = MultiHeadSelfAttn(H^t)`

### Decoder
`P^t = σ(MLP([Z^t[u] ⊕ Z^t[v]]))`

### Loss
`L = -1/|B| · Σ [Y·log(P) + (1-Y)·log(1-P)]`

## Reported scores (Table 2)

| Dataset | AUC | AP |
|---|---|---|
| Mooc-action | 98.80 | 98.63 |
| CollegeMsg | 91.49 | 89.26 |
| EUT | 92.22 | 90.82 |
| Bitcoinotc | 91.20 | 89.43 |
| LastFM | 87.57 | 87.04 |
| Wikipedia | 87.42 | 85.75 |

## Things the paper omits

- Learning rate, optimizer, batch size, epochs.
- Embedding/hidden dim.
- Number of attention heads.
- Negative sampling strategy and ratio.
- Initialization of W^0.
- Code repository.
```

- [ ] **Step 6: Commit**

```bash
git add docs/reproduction-log.md docs/paper-notes.md configs/experiments/gcn_ma_collegemsg_smoke.yaml configs/models/gcn_ma_smoke.yaml results/metrics.jsonl
git commit -m "[milestone] GCN_MA end-to-end on CollegeMsg with paper-notes and reproduction-log"
git tag v0.1-foundation
```

---

## Done

After Task 22, GCN_MA trains and evaluates on CollegeMsg. Compare the reported test AUC against the paper's 91.49%. Expect a gap; document it in `docs/reproduction-log.md`. The next plan extends the loader pipeline to the remaining 5 datasets and adds β grid search.
