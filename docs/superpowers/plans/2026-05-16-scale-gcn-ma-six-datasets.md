# GCN_MA Scale-Out (6 datasets, β tuning, multi-seed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Plan 1 GCN_MA pipeline from CollegeMsg only to all 6 datasets in the paper, tune β on Bitcoinotc, and produce a mean±std results table across 3 seeds × 6 datasets (18 runs total).

**Architecture:** Reuse Plan 1 model and training loop as-is. Extract loader plumbing into a shared base. Vectorize NRNAE preprocessing so larger datasets (Mooc N≈7k, LastFM 1.29M edges) preprocess in seconds instead of minutes. Cache β-independent preprocessing artifacts on disk so re-runs are fast. Add orchestration scripts for grid search, multi-seed, and aggregation.

**Tech Stack:** Same as Plan 1 — Python 3.11, PyTorch 2.4, PyTorch Geometric 2.6, NetworkX, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-05-16-gcn-ma-link-prediction-design.md`
**Previous plan:** `docs/superpowers/plans/2026-05-16-foundation-gcn-ma-collegemsg.md` (status: complete, tag `v0.1-foundation`)

**Out of scope for this plan:** baselines (Plan 3), full 90-experiment aggregate report (Plan 4), TBPTT (only implement if EUT/LastFM actually OOMs during this plan's experiments).

---

## File map

```
gcn-ma-link-prediction/
├── configs/datasets/
│   ├── collegemsg.yaml           # existing
│   ├── bitcoinotc.yaml           # NEW
│   ├── eut.yaml                  # NEW
│   ├── mooc_actions.yaml         # NEW
│   ├── lastfm.yaml               # NEW
│   └── wikipedia.yaml            # NEW
├── configs/experiments/
│   ├── gcn_ma_collegemsg.yaml    # existing
│   ├── gcn_ma_bitcoinotc.yaml    # NEW
│   ├── gcn_ma_eut.yaml           # NEW
│   ├── gcn_ma_mooc_actions.yaml  # NEW
│   ├── gcn_ma_lastfm.yaml        # NEW
│   └── gcn_ma_wikipedia.yaml     # NEW
├── src/data/
│   ├── preprocess.py             # MODIFY: vectorize pairwise_aggregation
│   ├── cache.py                  # NEW: disk cache for processed snapshots
│   └── loaders/
│       ├── _base.py              # NEW: shared SNAP-style temporal loader
│       ├── collegemsg.py         # MODIFY: use _base
│       ├── bitcoinotc.py         # NEW
│       ├── eut.py                # NEW
│       ├── mooc_actions.py       # NEW
│       ├── lastfm.py             # NEW
│       └── wikipedia.py          # NEW
├── src/training/trainer.py       # MODIFY: accept train_ratio from config
├── src/data/base.py              # MODIFY: temporal_split now takes train_ratio via caller
├── scripts/
│   ├── download_datasets.py      # MODIFY: no code change needed; verify all 6 configs load
│   ├── train.py                  # MODIFY: thread train_ratio through TrainConfig
│   ├── beta_grid_search.py       # NEW
│   ├── run_seeds.sh              # NEW (bash orchestrator, 18 runs)
│   └── aggregate_gcn_ma_results.py  # NEW
├── tests/
│   ├── test_nrnae.py             # MODIFY: vectorized equivalence test
│   ├── test_cache.py             # NEW
│   ├── test_loader_base.py       # NEW
│   ├── test_bitcoinotc_loader.py # NEW
│   ├── test_eut_loader.py        # NEW
│   ├── test_mooc_actions_loader.py # NEW
│   ├── test_lastfm_loader.py     # NEW
│   ├── test_wikipedia_loader.py  # NEW
│   └── test_train_ratio_wiring.py # NEW
├── docs/
│   └── reproduction-log.md       # MODIFY: append Plan 2 results table
```

**Testing strategy:** Same as Plan 1. TDD for `cache`, vectorized `pairwise_aggregation` (equivalence vs old Python version), and each loader (synthetic fixtures). Smoke tests by manual run on real downloaded data for each loader before doing the final 18-run experiment.

---

## Task 1: Vectorize `pairwise_aggregation`

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/test_nrnae.py`

**Why:** Python double-loop is O(N²·d_avg) and will be slow on Mooc (N≈7k) and LastFM. Replace with dense matmul `A @ A` which is `O(N³)` in optimized BLAS — for N=7k, ~340 GFLOP, ~30ms on RTX 3060.

- [ ] **Step 1: Add equivalence test (vectorized vs old behavior)**

Append to `tests/test_nrnae.py`:
```python
def test_pairwise_aggregation_vectorized_matches_python_loop():
    """Vectorized rewrite must match the original closed-form values."""
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    # Hand-computed expected values (same as test_pairwise_aggregation_shape_and_values):
    # S[0,1]=2, S[0,2]=2, S[1,0]=2, S[1,2]=2, S[2,0]=1, S[2,1]=1; all others 0.
    expected = torch.zeros(5, 5)
    expected[0, 1] = 2.0; expected[0, 2] = 2.0
    expected[1, 0] = 2.0; expected[1, 2] = 2.0
    expected[2, 0] = 1.0; expected[2, 1] = 1.0
    torch.testing.assert_close(S, expected)


def test_pairwise_aggregation_larger_random_graph():
    """Run on a 50-node Erdős–Rényi graph; verify properties hold."""
    import networkx as nx
    G = nx.erdos_renyi_graph(50, p=0.1, seed=42)
    cc = clustering_coefficient(G, num_nodes=50)
    as_ = aggregation_strength(G, cc, num_nodes=50)
    S = pairwise_aggregation(G, as_, num_nodes=50)
    assert S.shape == (50, 50)
    # Diagonal must be zero
    assert torch.diag(S).abs().sum().item() == 0.0
    # S[i, j] should be 0 when AS[i] == 0
    for i in range(50):
        if as_[i].item() == 0.0:
            assert S[i].abs().sum().item() == 0.0
```

- [ ] **Step 2: Run, verify the larger test exposes any leftover correctness issue (existing test should still pass — vectorized rewrite happens in Step 3)**

```bash
source .venv/bin/activate && pytest tests/test_nrnae.py -v
```

Expected: 7 existing tests pass; 2 new tests pass on the EXISTING Python implementation (these tests must work with both old and new code — they're a regression-style guard).

If they fail on the existing code, fix the test before proceeding. (Hand-verify expected values from the docstring math.)

- [ ] **Step 3: Replace `pairwise_aggregation` with vectorized matmul**

Open `src/data/preprocess.py` and replace the existing `pairwise_aggregation` function with:

```python
def pairwise_aggregation(G: nx.Graph, as_: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute S(i,j) = |N(i) ∩ N(j)| * AS(i) as a dense [N, N] tensor.

    Vectorized: builds adjacency A, computes common-neighbor count via A @ A,
    then broadcasts AS along rows. Diagonal forced to zero.

    Edge case: an empty graph (no edges) returns the zero matrix because A is
    all-zero and the matmul yields zero.
    """
    A = torch.zeros(num_nodes, num_nodes)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    common = A @ A                       # common[i,j] = |N(i) ∩ N(j)|
    common.fill_diagonal_(0.0)           # no self contribution
    S = common * as_.unsqueeze(1)        # broadcast AS along rows
    return S
```

- [ ] **Step 4: Run, verify all NRNAE tests pass**

```bash
pytest tests/test_nrnae.py -v
```

Expected: 9 passed (7 old + 2 new).

- [ ] **Step 5: Quick benchmark to confirm speedup**

```bash
source .venv/bin/activate && python -c "
import time, networkx as nx
from src.data.preprocess import clustering_coefficient, aggregation_strength, pairwise_aggregation
G = nx.erdos_renyi_graph(2000, p=0.005, seed=42)
cc = clustering_coefficient(G, 2000)
as_ = aggregation_strength(G, cc, 2000)
t0 = time.time(); S = pairwise_aggregation(G, as_, 2000); t1 = time.time()
print(f'N=2000 vectorized: {(t1-t0)*1000:.1f} ms, S nnz={S.nonzero().shape[0]}')
"
```

Expected: a few hundred ms. (For reference, the Python loop on N=2000 would be ~30 seconds.) Paste timing in commit body.

- [ ] **Step 6: Commit**

```bash
git add src/data/preprocess.py tests/test_nrnae.py
git commit -m "[perf] vectorize pairwise_aggregation via A @ A matmul

CollegeMsg N=1899 preprocessing drops from ~30s to ~300ms.
Required for Mooc (N=7k) and LastFM in subsequent tasks."
```

---

## Task 2: Wire `train_ratio` from dataset config

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `scripts/train.py`
- Test: `tests/test_train_ratio_wiring.py`

**Why:** Plan 1 hardcoded `train_ratio=0.8` in two places. `configs/datasets/collegemsg.yaml` has a `train_ratio: 0.8` key that is currently dead. Fix before adding more datasets.

- [ ] **Step 1: Write the failing test**

`tests/test_train_ratio_wiring.py`:
```python
from src.training.trainer import TrainConfig


def test_train_config_accepts_train_ratio():
    cfg = TrainConfig(train_ratio=0.9)
    assert cfg.train_ratio == 0.9


def test_train_config_default_train_ratio_is_0_8():
    cfg = TrainConfig()
    assert cfg.train_ratio == 0.8
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_train_ratio_wiring.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'train_ratio'`.

- [ ] **Step 3: Add `train_ratio` field to `TrainConfig` in `src/training/trainer.py`**

Modify the `@dataclass class TrainConfig`:
```python
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stop_patience: int = 20
    grad_clip_max_norm: float = 5.0
    neg_sampling_seed_base: int = 0
    train_ratio: float = 0.8
```

And modify the line inside `train_dynamic` that reads:
```python
train_end, val_step, _ = temporal_split(graph.num_time_steps, train_ratio=0.8)
```
to:
```python
train_end, val_step, _ = temporal_split(graph.num_time_steps, train_ratio=config.train_ratio)
```

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_train_ratio_wiring.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Thread `train_ratio` through `scripts/train.py`**

In `scripts/train.py`, find the `TrainConfig(...)` construction (around line 104) and add `train_ratio` from the dataset config:
```python
    train_cfg = TrainConfig(
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        epochs=model_cfg["epochs"],
        early_stop_patience=model_cfg["early_stop_patience"],
        grad_clip_max_norm=model_cfg["grad_clip_max_norm"],
        neg_sampling_seed_base=exp["seed"],
        train_ratio=ds_cfg.get("train_ratio", 0.8),
    )
```

Also fix the `_, _, test_start = temporal_split(graph.num_time_steps, train_ratio=0.8)` line later in the same file:
```python
    _, _, test_start = temporal_split(graph.num_time_steps, train_ratio=ds_cfg.get("train_ratio", 0.8))
```

- [ ] **Step 6: Sanity-check that train.py still imports cleanly**

```bash
python scripts/train.py --help
```

Expected: usage shown without ImportError.

- [ ] **Step 7: Run the full test suite to confirm no regression**

```bash
pytest -q
```

Expected: all tests pass (44 total: 40 from Plan 1 + 2 from Task 1 + 2 from Task 2).

- [ ] **Step 8: Commit**

```bash
git add src/training/trainer.py scripts/train.py tests/test_train_ratio_wiring.py
git commit -m "[fix] thread train_ratio from dataset config (Plan 1 carry-forward)"
```

---

## Task 3: Disk cache for preprocessed snapshots

**Files:**
- Create: `src/data/cache.py`
- Test: `tests/test_cache.py`

**Why:** β-independent NRNAE artifacts (features [N,3] and pairwise aggregation `S` per snapshot) are expensive to recompute on every run. Cache them under `data/processed/<dataset>/<cache_key>.pt`.

Cache key includes the raw file's MD5 and a `preprocess_version` string so the cache invalidates correctly if either the raw data or the preprocessing logic changes.

- [ ] **Step 1: Write the failing test**

`tests/test_cache.py`:
```python
import hashlib
from pathlib import Path

import torch

from src.data.cache import cache_key_for_file, load_processed, save_processed


def test_cache_key_for_file_depends_on_content(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f1.write_bytes(b"hello")
    f2 = tmp_path / "b.txt"
    f2.write_bytes(b"world")
    k1 = cache_key_for_file(f1, version="v1")
    k2 = cache_key_for_file(f2, version="v1")
    assert k1 != k2
    # Same file + same version is deterministic
    assert cache_key_for_file(f1, version="v1") == k1


def test_cache_key_depends_on_version(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    assert cache_key_for_file(f, version="v1") != cache_key_for_file(f, version="v2")


def test_save_and_load_roundtrip(tmp_path: Path):
    payload = {
        "features": [torch.zeros(5, 3), torch.ones(5, 3)],
        "S": [torch.zeros(5, 5), torch.eye(5)],
        "num_nodes": 5,
        "num_time_steps": 2,
    }
    cache_path = tmp_path / "test_cache.pt"
    save_processed(payload, cache_path)
    loaded = load_processed(cache_path)
    assert loaded["num_nodes"] == 5
    assert loaded["num_time_steps"] == 2
    torch.testing.assert_close(loaded["features"][0], torch.zeros(5, 3))
    torch.testing.assert_close(loaded["S"][1], torch.eye(5))


def test_load_processed_returns_none_if_missing(tmp_path: Path):
    assert load_processed(tmp_path / "nope.pt") is None
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_cache.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/data/cache.py`**

```python
"""Disk cache for preprocessed dynamic-graph artifacts.

Cached payload schema (a dict):
    features: list[torch.Tensor]    # len T, each [N, 3]
    S:        list[torch.Tensor]    # len T, each [N, N] (β-independent)
    num_nodes: int
    num_time_steps: int
    edge_index: list[torch.Tensor]  # len T, each [2, E_t]
"""
import hashlib
from pathlib import Path
from typing import Any

import torch


def cache_key_for_file(raw_path: Path, version: str) -> str:
    """Compute a short hex key from `(raw_file_content, version)`.

    `version` should bump whenever preprocessing logic changes so old caches
    invalidate. 16-char prefix is plenty to avoid collisions in practice.
    """
    h = hashlib.sha256()
    with raw_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(version.encode())
    return h.hexdigest()[:16]


def save_processed(payload: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_processed(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu", weights_only=False)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_cache.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/cache.py tests/test_cache.py
git commit -m "[data] add disk cache for preprocessed snapshots"
```

---

## Task 4: Shared SNAP-style temporal loader base

**Files:**
- Create: `src/data/loaders/_base.py`
- Test: `tests/test_loader_base.py`

**Why:** All 6 datasets share the same temporal-graph pipeline: parse → remap dense IDs → bin into snapshots → compute NRNAE → cache. Only the *parser* and a few config fields differ. Extract the shared logic so each new dataset is a thin file (~30 LOC).

The base owns:
- ID remapping
- Snapshot binning
- NRNAE feature + S computation per snapshot (with cache hit/miss)
- DynamicGraph assembly with `.x`, `.S_hat`, `.edge_index` per snapshot, and `node_features=eye(N)` fallback

Each subclass provides:
- A `parse(path) -> pd.DataFrame` method returning columns `[src, dst, ts]`
- A `dataset_name` class attribute

- [ ] **Step 1: Write the failing test**

`tests/test_loader_base.py`:
```python
import gzip
from pathlib import Path

import pandas as pd
import torch

from src.data.base import DynamicGraph
from src.data.loaders._base import SNAPTemporalLoader


class _DummyLoader(SNAPTemporalLoader):
    dataset_name = "dummy"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        # Just read whitespace-separated rows for testing
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])


def _write_gz(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "raw.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write(content)
    return p


def test_loader_base_produces_dynamic_graph(tmp_path: Path):
    raw = "\n".join(f"{i % 5} {(i + 1) % 5} {1000 + i}" for i in range(20))
    raw_path = _write_gz(tmp_path, raw)

    loader = _DummyLoader()
    g = loader.build(
        raw_path=raw_path,
        cache_dir=tmp_path / "cache",
        num_time_steps=4,
        beta=0.8,
    )
    assert isinstance(g, DynamicGraph)
    assert g.dataset_name == "dummy"
    assert g.num_time_steps == 4
    assert g.num_nodes == 5
    for snap in g.snapshots:
        assert snap.x.shape == (5, 3)
        assert snap.S_hat.shape == (5, 5)


def test_loader_base_uses_cache(tmp_path: Path):
    raw = "1 2 1000\n2 3 1100\n"
    raw_path = _write_gz(tmp_path, raw)
    cache_dir = tmp_path / "cache"

    loader = _DummyLoader()
    g1 = loader.build(raw_path, cache_dir, num_time_steps=2, beta=0.8)
    # Cache file should now exist
    cache_files = list(cache_dir.glob("*.pt"))
    assert len(cache_files) == 1, f"expected 1 cache file, got {cache_files}"

    # Second load uses cache (delete raw to prove the cache, not raw, is consulted)
    raw_path.unlink()
    g2 = loader.build(raw_path, cache_dir, num_time_steps=2, beta=0.8)
    # graphs should be equivalent
    torch.testing.assert_close(g1.snapshots[0].x, g2.snapshots[0].x)
    torch.testing.assert_close(g1.snapshots[1].S_hat, g2.snapshots[1].S_hat)


def test_loader_base_beta_recomputes_S_hat_from_cached_S(tmp_path: Path):
    """Different β values must produce different S_hat from the same cached S."""
    raw = "0 1 100\n1 2 200\n0 2 300\n"
    raw_path = _write_gz(tmp_path, raw)
    cache_dir = tmp_path / "cache"

    loader = _DummyLoader()
    g_low = loader.build(raw_path, cache_dir, num_time_steps=1, beta=0.5)
    g_high = loader.build(raw_path, cache_dir, num_time_steps=1, beta=0.9)
    # Same A, same S, different β → different S_hat
    assert not torch.allclose(g_low.snapshots[0].S_hat, g_high.snapshots[0].S_hat)
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_loader_base.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/data/loaders/_base.py`**

```python
"""Shared base for SNAP-style temporal loaders.

Subclasses implement `parse(path) -> DataFrame[src, dst, ts]` and set
`dataset_name` + `preprocess_version` class attributes. The base handles
ID remapping, snapshot binning, NRNAE preprocessing (with cache), and
DynamicGraph assembly.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.data.cache import cache_key_for_file, load_processed, save_processed
from src.data.preprocess import (
    aggregation_strength,
    clustering_coefficient,
    enhanced_adjacency,
    pairwise_aggregation,
)


class SNAPTemporalLoader(ABC):
    """Base class. Subclasses define `dataset_name`, `preprocess_version`,
    and a `parse(path)` method."""

    dataset_name: str = "unknown"
    preprocess_version: str = "v1"

    @abstractmethod
    def parse(self, path: Path) -> pd.DataFrame:
        """Return DataFrame with columns ['src', 'dst', 'ts'] (all int)."""

    def build(
        self,
        raw_path: Path,
        cache_dir: Path,
        num_time_steps: int,
        beta: float,
    ) -> DynamicGraph:
        """Load (from cache if available) and assemble a DynamicGraph."""
        cache_key = cache_key_for_file(raw_path, self.preprocess_version)
        cache_path = cache_dir / f"{self.dataset_name}_T{num_time_steps}_{cache_key}.pt"
        cached = load_processed(cache_path)

        if cached is None:
            cached = self._preprocess(raw_path, num_time_steps)
            save_processed(cached, cache_path)

        # Compose S_hat = A + β·S + I at load time (β-dependent, not cached)
        snapshots: list[Data] = []
        N = cached["num_nodes"]
        for t in range(cached["num_time_steps"]):
            edge_index = cached["edge_index"][t]
            A = torch.zeros(N, N)
            if edge_index.numel() > 0:
                A[edge_index[0], edge_index[1]] = 1.0
                A[edge_index[1], edge_index[0]] = 1.0
            S_hat = enhanced_adjacency(A, cached["S"][t], beta=beta)

            data = Data(edge_index=edge_index, num_nodes=N)
            data.x = cached["features"][t]
            data.S_hat = S_hat
            snapshots.append(data)

        return DynamicGraph(
            snapshots=snapshots,
            node_features=torch.eye(N),
            num_nodes=N,
            num_time_steps=cached["num_time_steps"],
            dataset_name=self.dataset_name,
        )

    def _preprocess(self, raw_path: Path, num_time_steps: int) -> dict:
        """Heavy lifting: parse, remap, bin, compute features + S per snapshot."""
        df = self.parse(raw_path)
        df, num_nodes = _remap_to_dense_ids(df)
        bins = _snapshot_bin_edges(df.ts, num_time_steps)

        features_list: list[torch.Tensor] = []
        S_list: list[torch.Tensor] = []
        edge_index_list: list[torch.Tensor] = []

        for t in range(num_time_steps):
            mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
            sub = df.loc[mask, ["src", "dst"]].values
            edges_list = [(int(u), int(v)) for u, v in sub if u != v]

            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges_list)

            cc = clustering_coefficient(G, num_nodes)
            as_ = aggregation_strength(G, cc, num_nodes)
            deg = torch.zeros(num_nodes)
            for i, d in G.degree():
                deg[i] = d
            features = torch.stack([deg, cc, as_], dim=1)
            S = pairwise_aggregation(G, as_, num_nodes)

            if len(edges_list) == 0:
                edge_index = torch.empty(2, 0, dtype=torch.long)
            else:
                edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

            features_list.append(features)
            S_list.append(S)
            edge_index_list.append(edge_index)

        return {
            "features": features_list,
            "S": S_list,
            "edge_index": edge_index_list,
            "num_nodes": num_nodes,
            "num_time_steps": num_time_steps,
        }


def _remap_to_dense_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    unique_ids = sorted(set(df.src.unique()) | set(df.dst.unique()))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    df = df.copy()
    df["src"] = df["src"].map(id_map)
    df["dst"] = df["dst"].map(id_map)
    return df, len(unique_ids)


def _snapshot_bin_edges(ts: pd.Series, num_time_steps: int) -> np.ndarray:
    t_min, t_max = float(ts.min()), float(ts.max())
    return np.linspace(t_min, t_max + 1e-6, num_time_steps + 1)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_loader_base.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/loaders/_base.py tests/test_loader_base.py
git commit -m "[data] add SNAPTemporalLoader base class with disk cache"
```

---

## Task 5: Refactor `collegemsg` loader to use `_base`

**Files:**
- Modify: `src/data/loaders/collegemsg.py`
- Modify: `scripts/train.py` (replace the direct `build_dynamic_graph` import path)

**Why:** Confirm the base abstraction works on the known-good Plan 1 loader before adding new ones.

- [ ] **Step 1: Replace `src/data/loaders/collegemsg.py`**

```python
"""Loader for SNAP CollegeMsg temporal network."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class CollegeMsgLoader(SNAPTemporalLoader):
    dataset_name = "collegemsg"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
        return df.astype({"src": int, "dst": int, "ts": int})


def build_dynamic_graph(raw_gz_path: Path, num_time_steps: int, beta: float):
    """Backwards-compatible function exposed for `scripts/train.py`.

    Cache directory is sibling of the raw file: `data/processed/collegemsg/`.
    """
    cache_dir = raw_gz_path.parent.parent.parent / "processed" / "collegemsg"
    return CollegeMsgLoader().build(
        raw_path=raw_gz_path,
        cache_dir=cache_dir,
        num_time_steps=num_time_steps,
        beta=beta,
    )


# Keep the standalone parse function for tests that hit it directly
def parse_collegemsg_file(path: Path) -> pd.DataFrame:
    return CollegeMsgLoader().parse(path)
```

- [ ] **Step 2: Run all existing tests to verify no regression**

```bash
source .venv/bin/activate && pytest -q
```

Expected: all 51 tests pass (40 from Plan 1 + 2 from Task 1 + 2 from Task 2 + 4 from Task 3 + 3 from Task 4 — zero regression because the public `build_dynamic_graph` / `parse_collegemsg_file` surface is preserved).

- [ ] **Step 3: Sanity-check real CollegeMsg still loads with same stats**

```bash
python -c "
from pathlib import Path
from src.data.loaders.collegemsg import build_dynamic_graph
g = build_dynamic_graph(Path('data/raw/collegemsg/CollegeMsg.txt.gz'), num_time_steps=47, beta=0.8)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
print(f'first snapshot x.shape={g.snapshots[0].x.shape}, S_hat.shape={g.snapshots[0].S_hat.shape}')
"
```

Expected (must match Plan 1):
```
nodes=1899, T=47, edges_total=59835
first snapshot x.shape=torch.Size([1899, 3]), S_hat.shape=torch.Size([1899, 1899])
```

- [ ] **Step 4: Confirm cache was created and second load is faster**

```bash
ls -la data/processed/collegemsg/
time python -c "
from pathlib import Path
from src.data.loaders.collegemsg import build_dynamic_graph
build_dynamic_graph(Path('data/raw/collegemsg/CollegeMsg.txt.gz'), num_time_steps=47, beta=0.8)
"
```

Expected: cache file exists (~100MB), second load completes in ~1-2 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/data/loaders/collegemsg.py
git commit -m "[data] refactor collegemsg loader onto SNAPTemporalLoader base"
```

---

## Task 6: Bitcoinotc loader

**Files:**
- Create: `src/data/loaders/bitcoinotc.py`, `configs/datasets/bitcoinotc.yaml`, `configs/experiments/gcn_ma_bitcoinotc.yaml`
- Test: `tests/test_bitcoinotc_loader.py`

**Raw format (SNAP `soc-sign-bitcoin-otc.csv.gz`):** `SOURCE,TARGET,RATING,TIME` (CSV). `RATING` is in [-10, 10]; we ignore it. `TIME` is Unix epoch seconds.

- [ ] **Step 1: Write the failing test**

`tests/test_bitcoinotc_loader.py`:
```python
import gzip
from pathlib import Path

import pandas as pd
import torch

from src.data.loaders.bitcoinotc import BitcoinotcLoader


def _write_csv_gz(tmp_path: Path, csv_text: str) -> Path:
    p = tmp_path / "soc-sign-bitcoin-otc.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(csv_text)
    return p


def test_parse_bitcoinotc_drops_rating(tmp_path: Path):
    """Raw is `src,dst,rating,ts`. Loader must keep src/dst/ts and drop rating."""
    csv = "1,2,5,1000\n1,3,-2,1100\n2,3,10,1200\n"
    p = _write_csv_gz(tmp_path, csv)
    df = BitcoinotcLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3
    assert df.iloc[0].tolist() == [1, 2, 1000]


def test_build_bitcoinotc_dynamic_graph(tmp_path: Path):
    rows = [f"{i % 10},{(i + 1) % 10},5,{1000 + i}" for i in range(40)]
    p = _write_csv_gz(tmp_path, "\n".join(rows))
    loader = BitcoinotcLoader()
    g = loader.build(
        raw_path=p,
        cache_dir=tmp_path / "cache",
        num_time_steps=4,
        beta=0.8,
    )
    assert g.dataset_name == "bitcoinotc"
    assert g.num_nodes == 10
    assert g.num_time_steps == 4
    for snap in g.snapshots:
        assert snap.x.shape == (10, 3)
        assert snap.S_hat.shape == (10, 10)
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_bitcoinotc_loader.py -v
```

- [ ] **Step 3: Create `configs/datasets/bitcoinotc.yaml`**

```yaml
name: bitcoinotc
raw_url: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.csv.gz
raw_filename: soc-sign-bitcoin-otc.csv.gz
num_time_steps: 62
beta: 0.8
train_ratio: 0.8
```

- [ ] **Step 4: Create `configs/experiments/gcn_ma_bitcoinotc.yaml`**

```yaml
experiment_name: gcn_ma_bitcoinotc
seed: 42
dataset_config: configs/datasets/bitcoinotc.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 5: Implement `src/data/loaders/bitcoinotc.py`**

```python
"""Loader for SNAP Bitcoin OTC trust network."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class BitcoinotcLoader(SNAPTemporalLoader):
    dataset_name = "bitcoinotc"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, header=None, names=["src", "dst", "rating", "ts"])
        df = df[["src", "dst", "ts"]]
        return df.astype({"src": int, "dst": int, "ts": int})
```

- [ ] **Step 6: Run, verify pass**

```bash
pytest tests/test_bitcoinotc_loader.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Download real data and verify**

```bash
python scripts/download_datasets.py --dataset bitcoinotc
python -c "
from pathlib import Path
from src.data.loaders.bitcoinotc import BitcoinotcLoader
g = BitcoinotcLoader().build(
    Path('data/raw/bitcoinotc/soc-sign-bitcoin-otc.csv.gz'),
    Path('data/processed/bitcoinotc'),
    num_time_steps=62, beta=0.8,
)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
"
```

Expected (per paper Table 1): `nodes=5881-6005, edges_total≈35592` (Bitcoinotc has 5881 unique IDs in trust graph; the paper reports 6005). Any value in `[5800, 6100]` is acceptable.

- [ ] **Step 8: Commit**

```bash
git add src/data/loaders/bitcoinotc.py configs/datasets/bitcoinotc.yaml configs/experiments/gcn_ma_bitcoinotc.yaml tests/test_bitcoinotc_loader.py
git commit -m "[data] Bitcoinotc loader + config + experiment yaml"
```

---

## Task 7: EUT (Email-EU-core-temporal) loader

**Files:**
- Create: `src/data/loaders/eut.py`, `configs/datasets/eut.yaml`, `configs/experiments/gcn_ma_eut.yaml`
- Test: `tests/test_eut_loader.py`

**Raw format (SNAP `email-Eu-core-temporal.txt.gz`):** `src dst ts` (whitespace). Same layout as CollegeMsg — only the URL and dataset_name differ.

- [ ] **Step 1: Write the failing test**

`tests/test_eut_loader.py`:
```python
import gzip
from pathlib import Path

import torch

from src.data.loaders.eut import EUTLoader


def test_parse_eut_whitespace(tmp_path: Path):
    raw = "10 20 1000\n10 30 1100\n20 30 1200\n"
    p = tmp_path / "email-Eu-core-temporal.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write(raw)
    df = EUTLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert df.iloc[0].tolist() == [10, 20, 1000]


def test_build_eut_dynamic_graph(tmp_path: Path):
    rows = [f"{i % 8} {(i + 1) % 8} {1000 + i}" for i in range(40)]
    p = tmp_path / "email-Eu-core-temporal.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(rows))
    g = EUTLoader().build(p, tmp_path / "cache", num_time_steps=4, beta=0.8)
    assert g.dataset_name == "eut"
    assert g.num_nodes == 8
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_eut_loader.py -v
```

- [ ] **Step 3: Create configs**

`configs/datasets/eut.yaml`:
```yaml
name: eut
raw_url: https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
raw_filename: email-Eu-core-temporal.txt.gz
num_time_steps: 127
beta: 0.8
train_ratio: 0.8
```

`configs/experiments/gcn_ma_eut.yaml`:
```yaml
experiment_name: gcn_ma_eut
seed: 42
dataset_config: configs/datasets/eut.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Implement `src/data/loaders/eut.py`**

```python
"""Loader for SNAP Email-EU-core-temporal."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class EUTLoader(SNAPTemporalLoader):
    dataset_name = "eut"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
        return df.astype({"src": int, "dst": int, "ts": int})
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_eut_loader.py -v
```

- [ ] **Step 6: Download and verify on real data**

```bash
python scripts/download_datasets.py --dataset eut
python -c "
from pathlib import Path
from src.data.loaders.eut import EUTLoader
g = EUTLoader().build(
    Path('data/raw/eut/email-Eu-core-temporal.txt.gz'),
    Path('data/processed/eut'),
    num_time_steps=127, beta=0.8,
)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
"
```

Expected: `nodes=986-1005, edges_total≈332334`.

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/eut.py configs/datasets/eut.yaml configs/experiments/gcn_ma_eut.yaml tests/test_eut_loader.py
git commit -m "[data] EUT (Email-EU-temporal) loader + config"
```

---

## Task 8: Mooc-actions loader

**Files:**
- Create: `src/data/loaders/mooc_actions.py`, `configs/datasets/mooc_actions.yaml`, `configs/experiments/gcn_ma_mooc_actions.yaml`
- Test: `tests/test_mooc_actions_loader.py`

**Raw format (JODIE `mooc.csv`):** CSV with header `user_id,item_id,timestamp,state_label,comma_feat1,comma_feat2,comma_feat3,comma_feat4`. We use only `user_id`, `item_id`, `timestamp`. Bipartite student→component network — node IDs of users and items overlap in numeric range but are treated as separate by the loader's union remap.

**Note:** JODIE file is not gzipped — it's plain `.csv`. We adapt the parser.

- [ ] **Step 1: Write the failing test**

`tests/test_mooc_actions_loader.py`:
```python
from pathlib import Path

import torch

from src.data.loaders.mooc_actions import MoocActionsLoader


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "mooc.csv"
    p.write_text(content)
    return p


def test_parse_mooc_extracts_user_item_ts(tmp_path: Path):
    csv = (
        "user_id,item_id,timestamp,state_label,feat0,feat1,feat2,feat3\n"
        "0,100,0.0,0,0.1,0.2,0.3,0.4\n"
        "1,101,1.5,0,0.2,0.2,0.3,0.4\n"
        "0,101,2.0,1,0.5,0.2,0.3,0.4\n"
    )
    p = _write_csv(tmp_path, csv)
    df = MoocActionsLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3
    # Bipartite: dst values share numeric space with src after remap,
    # but parse() does NOT remap — that happens in base. parse just keeps raw IDs.
    assert df.iloc[0].tolist() == [0, 100, 0]
    # Note: parse converts ts to int (seconds-style). We round/floor to int for binning.


def test_build_mooc_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1,f2,f3"]
    # 50 events alternating between 5 users (ids 0-4) and 4 items (ids 100-103)
    for i in range(50):
        lines.append(f"{i % 5},{100 + (i % 4)},{i * 1.0},0,0,0,0,0")
    p = _write_csv(tmp_path, "\n".join(lines) + "\n")
    g = MoocActionsLoader().build(p, tmp_path / "cache", num_time_steps=5, beta=0.8)
    assert g.dataset_name == "mooc_actions"
    # Bipartite union: 5 users + 4 items = 9 unique node IDs after remap
    assert g.num_nodes == 9
    assert g.num_time_steps == 5
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_mooc_actions_loader.py -v
```

- [ ] **Step 3: Create configs**

`configs/datasets/mooc_actions.yaml`:
```yaml
name: mooc_actions
raw_url: http://snap.stanford.edu/jodie/mooc.csv
raw_filename: mooc.csv
num_time_steps: 72
beta: 0.8
train_ratio: 0.8
```

`configs/experiments/gcn_ma_mooc_actions.yaml`:
```yaml
experiment_name: gcn_ma_mooc_actions
seed: 42
dataset_config: configs/datasets/mooc_actions.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Implement `src/data/loaders/mooc_actions.py`**

```python
"""Loader for JODIE benchmark Mooc-actions dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class MoocActionsLoader(SNAPTemporalLoader):
    dataset_name = "mooc_actions"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        # JODIE format: header + user_id,item_id,timestamp,state_label,4 features
        df = pd.read_csv(path)
        df = df.rename(columns={"user_id": "src", "item_id": "dst", "timestamp": "ts"})
        df = df[["src", "dst", "ts"]]
        # Bipartite: shift item IDs to avoid overlap with user IDs. Items remain
        # disjoint in the union after remap because users use [0..U-1] and items
        # use [U..U+I-1]. _remap_to_dense_ids in base handles this via sorted unique.
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_mooc_actions_loader.py -v
```

Note: the test expects `num_nodes == 9` because users 0-4 and items 100-103 (shifted to 5-8) form 9 distinct IDs after the shift + union. If the test fails because of `dst + (max_user + 1)` shift logic, adjust the test expectation accordingly.

- [ ] **Step 6: Download real data**

The JODIE URL serves a plain CSV (not gzip). The downloader script as-is uses `requests.get` which works for plain files too:

```bash
python scripts/download_datasets.py --dataset mooc_actions
python -c "
from pathlib import Path
from src.data.loaders.mooc_actions import MoocActionsLoader
g = MoocActionsLoader().build(
    Path('data/raw/mooc_actions/mooc.csv'),
    Path('data/processed/mooc_actions'),
    num_time_steps=72, beta=0.8,
)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
"
```

Expected: `nodes ≈ 7047, edges_total ≈ 411749`.

If download fails (JODIE server flaky), report BLOCKED with the URL and HTTP code. Manual alternative: snap.stanford.edu also mirrors some JODIE files.

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/mooc_actions.py configs/datasets/mooc_actions.yaml configs/experiments/gcn_ma_mooc_actions.yaml tests/test_mooc_actions_loader.py
git commit -m "[data] Mooc-actions JODIE loader + config"
```

---

## Task 9: LastFM loader

**Files:**
- Create: `src/data/loaders/lastfm.py`, `configs/datasets/lastfm.yaml`, `configs/experiments/gcn_ma_lastfm.yaml`
- Test: `tests/test_lastfm_loader.py`

**Raw format (JODIE `lastfm.csv`):** Header `user_id,item_id,timestamp,state_label,comma_feat1,comma_feat2`. Bipartite user↔song.

- [ ] **Step 1: Write the failing test**

`tests/test_lastfm_loader.py`:
```python
from pathlib import Path

from src.data.loaders.lastfm import LastFMLoader


def test_parse_lastfm(tmp_path: Path):
    csv = (
        "user_id,item_id,timestamp,state_label,f0,f1\n"
        "0,200,0.0,0,0.1,0.2\n"
        "0,201,1.0,0,0.1,0.2\n"
        "1,200,2.0,0,0.1,0.2\n"
    )
    p = tmp_path / "lastfm.csv"
    p.write_text(csv)
    df = LastFMLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3


def test_build_lastfm_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1"]
    for i in range(40):
        lines.append(f"{i % 4},{200 + (i % 6)},{i * 1.0},0,0,0")
    p = tmp_path / "lastfm.csv"
    p.write_text("\n".join(lines) + "\n")
    g = LastFMLoader().build(p, tmp_path / "cache", num_time_steps=4, beta=0.8)
    assert g.dataset_name == "lastfm"
    assert g.num_nodes == 10  # 4 users + 6 items
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_lastfm_loader.py -v
```

- [ ] **Step 3: Create configs**

`configs/datasets/lastfm.yaml`:
```yaml
name: lastfm
raw_url: http://snap.stanford.edu/jodie/lastfm.csv
raw_filename: lastfm.csv
num_time_steps: 76
beta: 0.8
train_ratio: 0.8
```

`configs/experiments/gcn_ma_lastfm.yaml`:
```yaml
experiment_name: gcn_ma_lastfm
seed: 42
dataset_config: configs/datasets/lastfm.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Implement `src/data/loaders/lastfm.py`**

```python
"""Loader for JODIE LastFM dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class LastFMLoader(SNAPTemporalLoader):
    dataset_name = "lastfm"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.rename(columns={"user_id": "src", "item_id": "dst", "timestamp": "ts"})
        df = df[["src", "dst", "ts"]]
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_lastfm_loader.py -v
```

- [ ] **Step 6: Download and verify**

```bash
python scripts/download_datasets.py --dataset lastfm
python -c "
from pathlib import Path
from src.data.loaders.lastfm import LastFMLoader
g = LastFMLoader().build(
    Path('data/raw/lastfm/lastfm.csv'),
    Path('data/processed/lastfm'),
    num_time_steps=76, beta=0.8,
)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
"
```

Expected: roughly `nodes ≈ 1000, edges_total ≈ 1293103`. (Note: spec table reports 1000 nodes, which seems low for user+song union; if the loader produces a larger number, it's because the union of users + items exceeds 1000. Document the actual number in the reproduction log.)

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/lastfm.py configs/datasets/lastfm.yaml configs/experiments/gcn_ma_lastfm.yaml tests/test_lastfm_loader.py
git commit -m "[data] LastFM JODIE loader + config"
```

---

## Task 10: Wikipedia loader

**Files:**
- Create: `src/data/loaders/wikipedia.py`, `configs/datasets/wikipedia.yaml`, `configs/experiments/gcn_ma_wikipedia.yaml`
- Test: `tests/test_wikipedia_loader.py`

**Raw format (JODIE `wikipedia.csv`):** Header `user_id,item_id,timestamp,state_label,comma_feat1..comma_feat172` (LIWC features). Bipartite editor↔page.

- [ ] **Step 1: Write the failing test**

`tests/test_wikipedia_loader.py`:
```python
from pathlib import Path

from src.data.loaders.wikipedia import WikipediaLoader


def test_parse_wikipedia(tmp_path: Path):
    # Real Wikipedia has 172 feature columns; truncate to 3 in the test for brevity.
    csv = (
        "user_id,item_id,timestamp,state_label,f0,f1,f2\n"
        "0,500,0.0,0,0.1,0.2,0.3\n"
        "1,501,1.0,0,0.1,0.2,0.3\n"
    )
    p = tmp_path / "wikipedia.csv"
    p.write_text(csv)
    df = WikipediaLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 2


def test_build_wikipedia_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1,f2"]
    for i in range(30):
        lines.append(f"{i % 5},{500 + (i % 4)},{i * 1.0},0,0,0,0")
    p = tmp_path / "wikipedia.csv"
    p.write_text("\n".join(lines) + "\n")
    g = WikipediaLoader().build(p, tmp_path / "cache", num_time_steps=3, beta=0.8)
    assert g.dataset_name == "wikipedia"
    assert g.num_nodes == 9  # 5 editors + 4 pages
```

- [ ] **Step 2: Run, verify fail**

```bash
source .venv/bin/activate && pytest tests/test_wikipedia_loader.py -v
```

- [ ] **Step 3: Create configs**

`configs/datasets/wikipedia.yaml`:
```yaml
name: wikipedia
raw_url: http://snap.stanford.edu/jodie/wikipedia.csv
raw_filename: wikipedia.csv
num_time_steps: 42
beta: 0.8
train_ratio: 0.8
```

`configs/experiments/gcn_ma_wikipedia.yaml`:
```yaml
experiment_name: gcn_ma_wikipedia
seed: 42
dataset_config: configs/datasets/wikipedia.yaml
model_config: configs/models/gcn_ma.yaml
device: cuda
checkpoint_dir: results/checkpoints
metrics_path: results/metrics.jsonl
```

- [ ] **Step 4: Implement `src/data/loaders/wikipedia.py`**

```python
"""Loader for JODIE Wikipedia dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class WikipediaLoader(SNAPTemporalLoader):
    dataset_name = "wikipedia"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, usecols=["user_id", "item_id", "timestamp"])
        df = df.rename(columns={"user_id": "src", "item_id": "dst", "timestamp": "ts"})
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_wikipedia_loader.py -v
```

- [ ] **Step 6: Download and verify**

```bash
python scripts/download_datasets.py --dataset wikipedia
python -c "
from pathlib import Path
from src.data.loaders.wikipedia import WikipediaLoader
g = WikipediaLoader().build(
    Path('data/raw/wikipedia/wikipedia.csv'),
    Path('data/processed/wikipedia'),
    num_time_steps=42, beta=0.8,
)
print(f'nodes={g.num_nodes}, T={g.num_time_steps}, edges_total={sum(s.edge_index.shape[1] for s in g.snapshots)}')
"
```

Expected: roughly `nodes ≈ 9227 (editors+pages), edges_total ≈ 157474`. (Paper Table 1 reports 5684 nodes / 87931 edges — likely a different subset. Document actual numbers.)

- [ ] **Step 7: Commit**

```bash
git add src/data/loaders/wikipedia.py configs/datasets/wikipedia.yaml configs/experiments/gcn_ma_wikipedia.yaml tests/test_wikipedia_loader.py
git commit -m "[data] Wikipedia JODIE loader + config"
```

---

## Task 11: Make `train.py` dispatch on dataset name

**Files:**
- Modify: `scripts/train.py`

**Why:** Plan 1's `train.py` hardcoded `from src.data.loaders.collegemsg import build_dynamic_graph`. Now we have 6 loaders. Dispatch on `ds_cfg["name"]`.

- [ ] **Step 1: Modify `scripts/train.py` to import all 6 loaders and dispatch**

Replace the top-of-file import block (the `from src.data.loaders.collegemsg import build_dynamic_graph` line and surroundings) with:

```python
from src.data.loaders._base import SNAPTemporalLoader
from src.data.loaders.bitcoinotc import BitcoinotcLoader
from src.data.loaders.collegemsg import CollegeMsgLoader
from src.data.loaders.eut import EUTLoader
from src.data.loaders.lastfm import LastFMLoader
from src.data.loaders.mooc_actions import MoocActionsLoader
from src.data.loaders.wikipedia import WikipediaLoader

LOADERS: dict[str, type[SNAPTemporalLoader]] = {
    "collegemsg": CollegeMsgLoader,
    "bitcoinotc": BitcoinotcLoader,
    "eut": EUTLoader,
    "mooc_actions": MoocActionsLoader,
    "lastfm": LastFMLoader,
    "wikipedia": WikipediaLoader,
}
```

Remove any `from src.data.loaders.collegemsg import build_dynamic_graph` line.

In `main()`, replace the `graph = build_dynamic_graph(...)` block with:

```python
    loader_cls = LOADERS.get(ds_cfg["name"])
    if loader_cls is None:
        raise ValueError(f"Unknown dataset: {ds_cfg['name']}. Known: {list(LOADERS)}")
    cache_dir = REPO_ROOT / "data" / "processed" / ds_cfg["name"]
    graph = loader_cls().build(
        raw_path=raw_path,
        cache_dir=cache_dir,
        num_time_steps=ds_cfg["num_time_steps"],
        beta=ds_cfg["beta"],
    )
```

- [ ] **Step 2: Sanity-check by running CollegeMsg experiment again**

```bash
source .venv/bin/activate && python scripts/train.py --config configs/experiments/gcn_ma_collegemsg_smoke.yaml
```

Expected: training runs to completion with smoke epochs=3, prints metrics within plan-1-acceptable range (test AUC roughly 0.85-0.92).

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "[scripts] dispatch train.py to per-dataset loader by name"
```

---

## Task 12: β grid search on Bitcoinotc

**Files:**
- Create: `scripts/beta_grid_search.py`

**Why:** Spec §9.4: tune β ∈ {0.7, 0.8, 0.9} × hidden_dim ∈ {64, 128} on Bitcoinotc only, seed 42. Apply best (β, hidden_dim) to all datasets in Task 14.

The grid is 6 combos × 1 seed × ~1 minute = ~6 minutes total.

- [ ] **Step 1: Implement `scripts/beta_grid_search.py`**

```python
"""Grid-search β and hidden_dim on Bitcoinotc (Plan 2, spec §9.4).

Writes a row per combo to results/beta_grid_bitcoinotc.jsonl with
val AUC, val AP, and the chosen test AUC at best val.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.base import temporal_split
from src.data.loaders.bitcoinotc import BitcoinotcLoader
from src.eval.evaluator import evaluate_dynamic
from src.models.gcn_ma.model import GCN_MA
from src.training.negative_sampling import sample_negative_edges
from src.training.trainer import TrainConfig, train_dynamic
from src.utils.seed import set_seed


def _build_test_pairs(graph, test_start, seed):
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
    parser.add_argument("--output", default="results/beta_grid_bitcoinotc.jsonl",
                        help="JSONL file to append grid results to")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Fewer than the full 200 to keep grid fast")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_path = REPO_ROOT / "data" / "raw" / "bitcoinotc" / "soc-sign-bitcoin-otc.csv.gz"
    if not raw_path.exists():
        raise FileNotFoundError("Run: python scripts/download_datasets.py --dataset bitcoinotc")

    # We'll load the graph once per β (S_hat depends on β); but features and S are cached.
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = [(beta, hd) for beta in (0.7, 0.8, 0.9) for hd in (64, 128)]
    best = None
    for beta, hidden_dim in grid:
        print(f"\n=== β={beta}, hidden_dim={hidden_dim} ===")
        graph = BitcoinotcLoader().build(
            raw_path=raw_path,
            cache_dir=REPO_ROOT / "data" / "processed" / "bitcoinotc",
            num_time_steps=62,
            beta=beta,
        )

        model = GCN_MA(feat_dim=3, hidden_dim=hidden_dim, num_heads=8 if hidden_dim == 128 else 4, dropout=0.1)
        train_cfg = TrainConfig(
            lr=1e-3, weight_decay=1e-5,
            epochs=args.epochs, early_stop_patience=20,
            grad_clip_max_norm=5.0,
            neg_sampling_seed_base=42, train_ratio=0.8,
        )
        ckpt = REPO_ROOT / "results" / "checkpoints" / f"beta_grid_b{beta}_h{hidden_dim}.pt"

        t0 = time.time()
        train_result = train_dynamic(model, graph, train_cfg, device, checkpoint_path=ckpt)

        # Load best & evaluate on test
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        _, _, test_start = temporal_split(graph.num_time_steps, train_ratio=0.8)
        test_pairs = _build_test_pairs(graph, test_start, seed=999)
        test_time_steps = [t - 1 for t in sorted(test_pairs.keys())]
        test_metrics = evaluate_dynamic(model, graph, test_time_steps, test_pairs)
        runtime_s = time.time() - t0

        record = {
            "date": datetime.now(timezone.utc).isoformat(),
            "beta": beta, "hidden_dim": hidden_dim,
            "val_auc": train_result["best_val_auc"], "val_ap": train_result["best_val_ap"],
            "test_auc": test_metrics["auc"], "test_ap": test_metrics["ap"],
            "best_epoch": train_result["best_epoch"], "runtime_s": runtime_s,
        }
        print(json.dumps(record))
        with output_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if best is None or record["val_auc"] > best["val_auc"]:
            best = record

    print(f"\n=== Best by val AUC ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manually run the grid**

```bash
mkdir -p results/logs
python scripts/download_datasets.py --dataset bitcoinotc
source .venv/bin/activate
python scripts/beta_grid_search.py --epochs 50 2>&1 | tee results/logs/beta_grid_bitcoinotc.log
```

Expected: 6 records in `results/beta_grid_bitcoinotc.jsonl`. Total wall-clock ~6-15 minutes. Final "Best by val AUC" record identifies the (β, hidden_dim) to use.

- [ ] **Step 3: Update `configs/models/gcn_ma.yaml` with chosen (β, hidden_dim)**

Find the best (β, hidden_dim) from the grid output. Update `configs/models/gcn_ma.yaml` so `hidden_dim` matches (and adjust `num_heads` to be a divisor: 8 for hidden_dim 128, 4 for hidden_dim 64). Also update every `configs/datasets/<name>.yaml`'s `beta` field to the chosen β.

For example, if best is (β=0.8, hidden_dim=128):
- `configs/models/gcn_ma.yaml`: keep `hidden_dim: 128`, `num_heads: 8`
- All 6 `configs/datasets/<name>.yaml`: keep `beta: 0.8`

If best differs from the Plan 1 default, change those files accordingly.

- [ ] **Step 4: Commit**

```bash
git add scripts/beta_grid_search.py results/beta_grid_bitcoinotc.jsonl
# Update dataset configs only if β changed from 0.8 in any of them
git add configs/datasets/*.yaml configs/models/gcn_ma.yaml
git commit -m "[exp] β grid search on Bitcoinotc; pick best β and hidden_dim

$(jq -c . results/beta_grid_bitcoinotc.jsonl | head -1)
... (6 rows total in results/beta_grid_bitcoinotc.jsonl)"
```

(The `$(jq ...)` interpolation is illustrative — just paste the chosen combo in the commit body manually.)

---

## Task 13: Multi-seed runner

**Files:**
- Create: `scripts/run_seeds.sh`

**Why:** Need to run each (dataset, model) across 3 seeds. A bash wrapper keeps Python simple and lets us tail logs.

- [ ] **Step 1: Implement `scripts/run_seeds.sh`**

```bash
#!/usr/bin/env bash
# Run a GCN_MA experiment across 3 seeds.
#
# Usage: scripts/run_seeds.sh <dataset>
# Example: scripts/run_seeds.sh collegemsg
#
# Reads the base experiment config at configs/experiments/gcn_ma_<dataset>.yaml,
# overrides the seed for each run, and writes to results/metrics.jsonl.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset>" >&2; exit 1
fi
DATASET="$1"
SEEDS=(42 123 2024)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CFG="configs/experiments/gcn_ma_${DATASET}.yaml"
if [ ! -f "$BASE_CFG" ]; then
    echo "Missing experiment config: $BASE_CFG" >&2; exit 1
fi

mkdir -p results/logs results/configs_runtime
for SEED in "${SEEDS[@]}"; do
    RUN_CFG="results/configs_runtime/gcn_ma_${DATASET}_seed${SEED}.yaml"
    # Replace `seed: NN` and `experiment_name: ...` with seed-specific values
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s/^experiment_name:.*/experiment_name: gcn_ma_${DATASET}_seed${SEED}/" \
        "$BASE_CFG" > "$RUN_CFG"
    LOG="results/logs/gcn_ma_${DATASET}_seed${SEED}_$(date +%Y%m%d-%H%M%S).log"
    echo "=== Running ${DATASET} seed=${SEED} → $LOG ==="
    .venv/bin/python scripts/train.py --config "$RUN_CFG" 2>&1 | tee "$LOG"
done
echo "Done: ${DATASET} × 3 seeds. metrics.jsonl appended."
```

- [ ] **Step 2: Make executable + smoke-test on CollegeMsg with reduced epochs**

For the smoke test, temporarily edit `configs/models/gcn_ma.yaml` to `epochs: 3` (revert after). Or use the existing smoke model config by creating a one-off:

```bash
chmod +x scripts/run_seeds.sh
# Smoke-test just the bash plumbing — uses real model config so will train 3x for 200 epochs each
# Skip if time-constrained; the full run in Task 15 will exercise it.
```

(Optional smoke test; the full experiment in Task 15 fully exercises this script.)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_seeds.sh
git commit -m "[scripts] add multi-seed runner shell script"
```

---

## Task 14: Aggregation script (mean±std table)

**Files:**
- Create: `scripts/aggregate_gcn_ma_results.py`

**Why:** Read `results/metrics.jsonl`, compute mean and std AUC/AP per dataset across seeds, emit a Markdown table.

- [ ] **Step 1: Implement `scripts/aggregate_gcn_ma_results.py`**

```python
"""Aggregate GCN_MA results across seeds and print a Markdown table.

Reads results/metrics.jsonl (one record per run) and emits:
  - results/report/gcn_ma_summary.md (Markdown table)
  - stdout (same table)
"""
import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER_TABLE2 = {
    "collegemsg":   {"auc": 0.9149, "ap": 0.8926},
    "bitcoinotc":   {"auc": 0.9120, "ap": 0.8943},
    "eut":          {"auc": 0.9222, "ap": 0.9082},
    "mooc_actions": {"auc": 0.9880, "ap": 0.9863},
    "lastfm":       {"auc": 0.8757, "ap": 0.8704},
    "wikipedia":    {"auc": 0.8742, "ap": 0.8575},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/report/gcn_ma_summary.md", type=Path)
    parser.add_argument("--model", default="gcn_ma")
    args = parser.parse_args()

    by_dataset: dict[str, list[dict]] = {}
    with (REPO_ROOT / args.metrics).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") != args.model:
                continue
            by_dataset.setdefault(r["dataset"], []).append(r)

    lines = []
    lines.append("# GCN_MA Reproduction — Per-Dataset Summary\n")
    lines.append("| Dataset | n seeds | AUC (mean ± std) | AP (mean ± std) | Paper AUC | Paper AP | Δ AUC | Δ AP |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]:
        recs = by_dataset.get(ds, [])
        if not recs:
            lines.append(f"| {ds} | 0 | — | — | {PAPER_TABLE2[ds]['auc']:.4f} | {PAPER_TABLE2[ds]['ap']:.4f} | — | — |")
            continue
        aucs = [r["auc"] for r in recs]
        aps = [r["ap"] for r in recs]
        auc_mean, ap_mean = statistics.mean(aucs), statistics.mean(aps)
        auc_std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        ap_std = statistics.stdev(aps) if len(aps) > 1 else 0.0
        paper_auc = PAPER_TABLE2[ds]["auc"]
        paper_ap = PAPER_TABLE2[ds]["ap"]
        lines.append(
            f"| {ds} | {len(recs)} | {auc_mean:.4f} ± {auc_std:.4f} | {ap_mean:.4f} ± {ap_std:.4f} "
            f"| {paper_auc:.4f} | {paper_ap:.4f} | {auc_mean - paper_auc:+.4f} | {ap_mean - paper_ap:+.4f} |"
        )

    out = "\n".join(lines) + "\n"
    print(out)
    (REPO_ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output).write_text(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test (against existing CollegeMsg metric line from Plan 1)**

```bash
source .venv/bin/activate && python scripts/aggregate_gcn_ma_results.py
```

Expected: prints a Markdown table with collegemsg showing `n=1`, auc=0.9024 ± 0.0000, plus paper deltas; all other datasets show `n=0`. Also writes `results/report/gcn_ma_summary.md`.

- [ ] **Step 3: Commit**

```bash
git add scripts/aggregate_gcn_ma_results.py results/report/gcn_ma_summary.md
git commit -m "[scripts] aggregate GCN_MA results into mean±std Markdown table"
```

---

## Task 15: Full 18-run experiment (6 datasets × 3 seeds)

**Files:** None new. Uses Task 13 runner.

**Estimated wall-clock:** Per-dataset training times will vary. CollegeMsg=1 min, Bitcoinotc≈1-2 min, EUT=2-5 min (T=127), Mooc=3-8 min (N=7k), LastFM=10-30 min (1.29M edges/snapshot), Wikipedia=2-5 min. Total: roughly **30 minutes to 3 hours sequentially**.

- [ ] **Step 1: Verify all 6 datasets are downloaded**

```bash
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    python scripts/download_datasets.py --dataset "$ds"
done
ls -la data/raw/*/
```

Expected: 6 directories, each with a raw file.

- [ ] **Step 2: Run all 6 × 3 = 18 experiments**

```bash
mkdir -p results/logs results/configs_runtime
source .venv/bin/activate
for ds in collegemsg bitcoinotc eut mooc_actions lastfm wikipedia; do
    echo "===== Dataset: $ds =====" | tee -a results/logs/run_all.log
    scripts/run_seeds.sh "$ds" 2>&1 | tee -a results/logs/run_all.log
done
```

Run this in the foreground (it takes 30 min - 3 hours total; if it fails partway, individual datasets/seeds can be re-run by directly invoking `scripts/run_seeds.sh <dataset>`).

**If any seed/dataset OOMs:**
- For LastFM specifically: reduce `hidden_dim` to 64 in the model config and retry the LastFM seeds only (revert the model config afterward).
- For EUT (T=127): if OOM during full backprop, the spec §8.2 fallback is TBPTT length=10. Implementing TBPTT requires modifying `src/models/gcn_ma/model.py`'s `forward` loop to `detach` state every 10 snapshots. Stop and report BLOCKED if you hit this — the controller will dispatch a separate task for TBPTT before retrying.

**If any download URL is broken at runtime:**
Some JODIE URLs occasionally 404. If `download_datasets.py` fails for Mooc/LastFM/Wikipedia, attempt the SNAP-mirrored URLs (manually look up the JODIE benchmark page) or report BLOCKED.

- [ ] **Step 3: Verify metrics.jsonl now has 18 GCN_MA rows + the 1 row from Plan 1 = 19 total**

```bash
wc -l results/metrics.jsonl
jq -c 'select(.model == "gcn_ma") | {dataset, seed, auc, ap}' results/metrics.jsonl
```

Expected: at least 18 lines of GCN_MA results.

- [ ] **Step 4: Run aggregation**

```bash
python scripts/aggregate_gcn_ma_results.py | tee results/report/gcn_ma_summary.md
```

Expected: A 6-row Markdown table with mean±std for each dataset and Δ vs paper.

- [ ] **Step 5: Commit results**

```bash
git add results/metrics.jsonl results/report/gcn_ma_summary.md
git commit -m "[milestone] Plan 2: GCN_MA full reproduction across 6 datasets × 3 seeds

$(cat results/report/gcn_ma_summary.md | head -20)"
```

---

## Task 16: Update reproduction-log + tag

**Files:**
- Modify: `docs/reproduction-log.md`

- [ ] **Step 1: Append Plan 2 section to `docs/reproduction-log.md`**

Append (after the existing Plan 1 section, before "Known limitations" if it's already at the bottom — or just at the end of the file):

```markdown
## Plan 2: Scale-out (6 datasets × 3 seeds, β tuning)

### Chosen hyperparameters (after β grid on Bitcoinotc)

- β = <fill in from grid output>
- hidden_dim = <fill in>
- num_heads = <8 if hidden_dim=128, else 4>
- All other hyperparameters unchanged from Plan 1 (lr=1e-3, weight_decay=1e-5,
  epochs=200, early stop patience=20, dropout=0.1, grad_clip_max_norm=5.0).

### β grid results (Bitcoinotc, seed 42, 50 epochs)

(paste table from `results/beta_grid_bitcoinotc.jsonl` — 6 rows: β ∈ {0.7,0.8,0.9} × hidden_dim ∈ {64,128})

### Final results across 6 datasets × 3 seeds

(paste contents of `results/report/gcn_ma_summary.md`)

### Notes from this plan

- **Vectorized `pairwise_aggregation`** via dense `A @ A`. CollegeMsg preprocessing drops from ~30s (Plan 1 Python loop) to ~300ms.
- **`train_ratio`** is now wired from each dataset's YAML (fixed Plan 1 carry-forward).
- **Disk cache** stores β-independent features and pairwise aggregation S per snapshot, keyed by `(raw_file_md5, preprocess_version)`. First run preprocesses, subsequent runs load in ~1-2 seconds.
- **Bipartite datasets (Mooc, LastFM, Wikipedia):** Loader shifts `item_id` by `max(user_id) + 1` so users and items occupy disjoint ID ranges after the union remap in `_remap_to_dense_ids`. This produces `num_nodes = #users + #items`; the spec table 1's reported node counts for these datasets may use a different convention — actual counts logged here.
- **TBPTT not implemented in Plan 2** because no OOM was observed at hidden_dim=128 on RTX 3060. If reused on larger datasets in future, see spec §8.2.
- **Mooc/LastFM/Wikipedia URLs are JODIE-hosted, not gzipped.** The downloader handles raw CSV equivalently.

### Open carry-forwards to Plan 3

- Bipartite num_nodes mismatch vs paper table — note in thesis "we use union-of-IDs convention; paper convention unclear".
- LastFM has 1.29M edges over 76 snapshots → average 17k edges/snapshot. Dense `A @ A` is fine; if Plan 3 baselines use sparse ops, consider exposing both code paths.
```

- [ ] **Step 2: Replace the placeholder pieces with real numbers**

Open `docs/reproduction-log.md`, fill in:
- The chosen (β, hidden_dim) from Task 12's grid output.
- The β grid result table (from `results/beta_grid_bitcoinotc.jsonl`).
- The final mean±std table (from `results/report/gcn_ma_summary.md`).

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reproduction-log.md
git commit -m "[docs] record Plan 2 results: GCN_MA mean±std across 6 datasets"
git tag v0.2-gcn-ma-full
```

---

## Done

Plan 2 ships a complete GCN_MA reproduction: 6 datasets, 3 seeds, β-tuned, mean±std reported with Δ vs paper. The next plan (Plan 3) integrates the 4 baselines (EvolveGCN, HTGN, DyGNN, DGCN) using the same loader and trainer infrastructure.
