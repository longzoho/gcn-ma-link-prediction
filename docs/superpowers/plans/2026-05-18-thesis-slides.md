# Thesis Defense Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render 3 new dataset-topology plots, write a Gamma-friendly Vietnamese outline, and generate a 22-slide thesis defense deck (+ 6 appendix) via Gamma MCP — per `docs/superpowers/specs/2026-05-18-thesis-slides-design.md`.

**Architecture:** New plot logic lives in a dedicated module `scripts/plot_dataset_topology.py` (loads cached snapshot tensors from `data/processed/<name>/<name>_T<T>_<hash>.pt`, computes per-dataset structural metrics, writes 3 PNGs to `results/report/plots/`). A thin wiring in `scripts/make_plots.py` exposes them as `--plots dataset_snapshots,edge_growth,topology_map` so they can be triggered from the existing CLI. Tests at `tests/test_plot_dataset_topology.py` mirror the smoke-test pattern in `tests/test_make_plots.py`. The Gamma outline lives at `docs/slides/thesis_defense_outline.md`, and the deck itself is generated through `mcp__claude_ai_Gamma__generate`.

**Tech Stack:** Python 3.11, PyTorch 2.4 (load `.pt` cache), NetworkX ≥ 3.2 (graph layout + degree), NumPy/SciPy (Gini coefficient), Matplotlib (headless `Agg` backend, already used by `make_plots.py`), pytest (existing test framework), Gamma MCP server.

---

## File Structure

**Create:**
- `scripts/plot_dataset_topology.py` — 3 plot functions + helpers + CLI entrypoint
- `tests/test_plot_dataset_topology.py` — smoke tests for each plot
- `docs/slides/thesis_defense_outline.md` — Gamma-friendly outline (Vietnamese)

**Modify:**
- `scripts/make_plots.py` — add 3 new plot keys (`dataset_snapshots`, `edge_growth`, `topology_map`) into the `main()` dispatcher (lines ~407–435 region)

**Outputs (artifacts):**
- `results/report/plots/dataset_snapshots_grid.png` — Slide 5
- `results/report/plots/edge_growth_density.png` — Slide 6
- `results/report/plots/topology_map_2d.png` — Slide 7
- `results/report/plots/topology_map_2d_with_winners.png` — Slide 18
- Gamma deck URL (saved to `docs/slides/gamma_deck_url.txt`)

---

## Task 1: Scaffold the new plot module

**Files:**
- Create: `scripts/plot_dataset_topology.py`

- [ ] **Step 1: Create the module skeleton**

Create `scripts/plot_dataset_topology.py`:

```python
"""Dataset-topology plots for thesis defense slides.

Produces three figures:

- ``dataset_snapshots_grid.png`` — 3 datasets × 3 timepoints small multiples
  (CollegeMsg, EUT, LastFM at t=0, T/2, T).  Used in Slide 5.
- ``edge_growth_density.png`` — two side-by-side line charts: edges per
  snapshot and density per snapshot, 6 lines each.  Used in Slide 6.
- ``topology_map_2d.png`` / ``topology_map_2d_with_winners.png`` — 2-D
  scatter of (mean density, degree-distribution Gini), marker shape for
  bipartite/unipartite.  Used in Slide 7 (plain) and Slide 18 (with
  winning model overlaid per dataset).

All data is read from the pre-built ``.pt`` snapshot caches in
``data/processed/<name>/`` so this script does not re-run the loaders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Dataset registry — name, T, bipartite, display label, default color.
DATASETS: list[tuple[str, int, bool, str, str]] = [
    ("collegemsg",   47,  False, "CollegeMsg",   "#1f77b4"),
    ("bitcoinotc",   62,  False, "Bitcoinotc",   "#ff7f0e"),
    ("eut",          127, False, "EUT",          "#2ca02c"),
    ("mooc_actions", 72,  True,  "Mooc-actions", "#d62728"),
    ("lastfm",       41,  True,  "LastFM",       "#9467bd"),
    ("wikipedia",    42,  True,  "Wikipedia",    "#8c564b"),
]

# Winners per dataset (from results/report/baselines_summary.md, AUC top-1).
DATASET_WINNERS: dict[str, str] = {
    "collegemsg":   "HTGN",
    "bitcoinotc":   "HTGN",
    "eut":          "DGCN",
    "mooc_actions": "DyGNN",
    "lastfm":       "EvolveGCN-O",
    "wikipedia":    "DyGNN",
}


def load_cached_snapshots(name: str, T: int) -> dict:
    """Return the first matching cache file for (name, T).

    Cache files live at ``data/processed/<name>/<name>_T<T>_<hash>.pt`` and
    are dicts with keys ``features``, ``edge_index`` (list of [2, E] tensors,
    one per snapshot), ``edge_ts``, ``num_nodes``, ``num_time_steps``.
    """
    cache_dir = REPO_ROOT / "data" / "processed" / name
    candidates = sorted(cache_dir.glob(f"{name}_T{T}_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No cached snapshots for {name} (T={T}). "
            f"Run scripts/train.py or scripts/download_datasets.py first."
        )
    return torch.load(candidates[0], weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "report" / "plots",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Comma-separated subset: snapshots,edge_growth,topology. 'all' = everything.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plots = (
        ["snapshots", "edge_growth", "topology"]
        if args.plots == "all"
        else args.plots.split(",")
    )

    if "snapshots" in plots:
        plot_dataset_snapshots_grid(args.out_dir / "dataset_snapshots_grid.png")
        print(f"Wrote {args.out_dir / 'dataset_snapshots_grid.png'}")
    if "edge_growth" in plots:
        plot_edge_growth_density(args.out_dir / "edge_growth_density.png")
        print(f"Wrote {args.out_dir / 'edge_growth_density.png'}")
    if "topology" in plots:
        plot_topology_map(
            args.out_dir / "topology_map_2d.png",
            args.out_dir / "topology_map_2d_with_winners.png",
        )
        print(f"Wrote {args.out_dir / 'topology_map_2d.png'}")
        print(f"Wrote {args.out_dir / 'topology_map_2d_with_winners.png'}")


# Plot functions are defined in later tasks.
def plot_dataset_snapshots_grid(out_path: Path) -> None: ...
def plot_edge_growth_density(out_path: Path) -> None: ...
def plot_topology_map(out_plain: Path, out_with_winners: Path) -> None: ...


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "import scripts.plot_dataset_topology as m; print(m.DATASETS[0])"`
Expected: `('collegemsg', 47, False, 'CollegeMsg', '#1f77b4')`

- [ ] **Step 3: Verify a cache file is readable for at least one dataset**

Run: `python -c "from scripts.plot_dataset_topology import load_cached_snapshots; d = load_cached_snapshots('collegemsg', 47); print(sorted(d.keys()))"`
Expected output contains `edge_index`, `num_nodes`, `num_time_steps`. If the script raises `FileNotFoundError`, the user must first run `python scripts/download_datasets.py --dataset collegemsg && python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml --epochs 1` to populate the cache.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_dataset_topology.py
git commit -m "[plots] scaffold plot_dataset_topology module"
```

---

## Task 2: Plot 1 — dataset_snapshots_grid.png

**Files:**
- Modify: `scripts/plot_dataset_topology.py` (replace `plot_dataset_snapshots_grid` stub)
- Test: `tests/test_plot_dataset_topology.py` (create)

**Goal:** 3×3 grid of node-link diagrams. Rows = 3 representative datasets (CollegeMsg sparse-unipartite, EUT dense-unipartite, LastFM dense-bipartite). Cols = t=0, T/2, T. Each cell shows the *cumulative graph through snapshot t* drawn with `nx.spring_layout` and small node markers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_dataset_topology.py`:

```python
"""Smoke tests for dataset-topology plots."""
from pathlib import Path

import pytest

from scripts.plot_dataset_topology import (
    DATASETS,
    load_cached_snapshots,
    plot_dataset_snapshots_grid,
)


def _has_cache(name: str, T: int) -> bool:
    """Skip-helper: true iff a cache file exists for (name, T)."""
    try:
        load_cached_snapshots(name, T)
        return True
    except FileNotFoundError:
        return False


REPRESENTATIVE = [("collegemsg", 47), ("eut", 127), ("lastfm", 41)]


@pytest.mark.skipif(
    not all(_has_cache(n, T) for n, T in REPRESENTATIVE),
    reason="Snapshot caches required for at least collegemsg/eut/lastfm.",
)
def test_plot_dataset_snapshots_grid_writes_nonempty_png(tmp_path):
    out = tmp_path / "snapshots.png"
    plot_dataset_snapshots_grid(out)
    assert out.exists()
    assert out.stat().st_size > 5000  # at least 5 KB
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_plot_dataset_topology.py::test_plot_dataset_snapshots_grid_writes_nonempty_png -v`
Expected: FAIL (the stub returns `None` without writing the file, so `out.exists()` is `False`).

- [ ] **Step 3: Implement the plot function**

In `scripts/plot_dataset_topology.py`, replace the stub for `plot_dataset_snapshots_grid` with:

```python
def _build_cumulative_graph(snapshots: list[torch.Tensor], up_to: int) -> nx.Graph:
    """Build an undirected NetworkX graph from edge_index tensors up to snapshot ``up_to``."""
    G = nx.Graph()
    for t in range(up_to + 1):
        ei = snapshots[t]
        # edge_index is shape [2, E_t]; convert to list of (u, v) python ints.
        for u, v in ei.t().tolist():
            G.add_edge(int(u), int(v))
    return G


def plot_dataset_snapshots_grid(out_path: Path) -> None:
    """3×3 grid of node-link diagrams (3 datasets × 3 timepoints)."""
    representatives = [
        ("collegemsg",   47,  False, "CollegeMsg (sparse, unipartite)"),
        ("eut",          127, False, "EUT (dense, unipartite)"),
        ("lastfm",       41,  True,  "LastFM (dense, bipartite)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))

    for row, (name, T, _bipartite, title) in enumerate(representatives):
        data = load_cached_snapshots(name, T)
        edge_index_list = data["edge_index"]
        # Three timepoints: 0, T/2, T-1.
        for col, t in enumerate([0, T // 2, T - 1]):
            G = _build_cumulative_graph(edge_index_list, t)
            ax = axes[row, col]
            # Subsample nodes if too dense so layout converges in <5 s and
            # the plot is readable.
            if G.number_of_nodes() > 400:
                # Keep the 400 highest-degree nodes (preserves hubs).
                top = sorted(G.degree, key=lambda x: -x[1])[:400]
                G = G.subgraph([n for n, _d in top]).copy()
            pos = nx.spring_layout(G, seed=42, k=1.0 / max(1, G.number_of_nodes()) ** 0.5)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.4)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6, node_color="#1f77b4")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_frame_on(False)
            if row == 0:
                col_title = ["t = 0", "t = T/2", "t = T"][col]
                ax.set_title(col_title, fontsize=11)
            if col == 0:
                ax.set_ylabel(title, fontsize=10, rotation=90, labelpad=10)

    fig.suptitle("Mỗi dataset là một quỹ đạo cấu trúc khác nhau", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_plot_dataset_topology.py::test_plot_dataset_snapshots_grid_writes_nonempty_png -v`
Expected: PASS. If skipped, the user lacks cache files — run a 1-epoch dummy train to materialise them.

- [ ] **Step 5: Visually inspect the output**

Run: `python scripts/plot_dataset_topology.py --plots snapshots`
Open `results/report/plots/dataset_snapshots_grid.png`. Acceptance criteria:
- 3 rows × 3 columns visible
- Row labels read "CollegeMsg (sparse, unipartite)", "EUT (dense, unipartite)", "LastFM (dense, bipartite)"
- Column titles t=0 / t=T/2 / t=T visible at top
- Each cell shows a node-link diagram (not a blank box or pure black blob)
- File size ≥ 50 KB

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_dataset_topology.py tests/test_plot_dataset_topology.py
git add results/report/plots/dataset_snapshots_grid.png
git commit -m "[plots] dataset_snapshots_grid for Slide 5"
```

---

## Task 3: Plot 2 — edge_growth_density.png

**Files:**
- Modify: `scripts/plot_dataset_topology.py` (replace `plot_edge_growth_density` stub)
- Test: `tests/test_plot_dataset_topology.py` (extend)

**Goal:** Two side-by-side line charts. (a) Edges *per snapshot* (not cumulative — paper-style). (b) Density $\rho^t = 2 E^t / (N (N-1))$. 6 lines (one per dataset).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plot_dataset_topology.py`:

```python
from scripts.plot_dataset_topology import (
    plot_edge_growth_density,
)


def _all_caches_present() -> bool:
    return all(_has_cache(n, T) for n, T, *_ in DATASETS)


@pytest.mark.skipif(not _all_caches_present(), reason="All 6 caches required.")
def test_plot_edge_growth_density_writes_nonempty_png(tmp_path):
    out = tmp_path / "edge_growth.png"
    plot_edge_growth_density(out)
    assert out.exists()
    assert out.stat().st_size > 5000
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_plot_dataset_topology.py::test_plot_edge_growth_density_writes_nonempty_png -v`
Expected: FAIL (stub returns None).

- [ ] **Step 3: Implement the plot function**

In `scripts/plot_dataset_topology.py`, replace the stub for `plot_edge_growth_density`:

```python
def _edges_per_snapshot(data: dict) -> list[int]:
    """Return list of edge counts for each snapshot (length T)."""
    return [int(ei.shape[1]) for ei in data["edge_index"]]


def _density_per_snapshot(data: dict) -> list[float]:
    """Density rho_t = 2 * E_t / (N * (N - 1)).  N is the global node count."""
    N = int(data["num_nodes"])
    if N < 2:
        return [0.0] * len(data["edge_index"])
    edges = _edges_per_snapshot(data)
    return [2.0 * e / (N * (N - 1)) for e in edges]


def plot_edge_growth_density(out_path: Path) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    for name, T, _bipartite, label, color in DATASETS:
        try:
            data = load_cached_snapshots(name, T)
        except FileNotFoundError:
            print(f"Warning: missing cache for {name}; skipping line.")
            continue
        edges = _edges_per_snapshot(data)
        density = _density_per_snapshot(data)
        ts = list(range(1, len(edges) + 1))
        ax_l.plot(ts, edges, label=label, color=color, linewidth=1.6)
        ax_r.plot(ts, density, label=label, color=color, linewidth=1.6)

    ax_l.set_xlabel("Snapshot index $t$")
    ax_l.set_ylabel("Số cạnh trong snapshot $E^t$")
    ax_l.set_yscale("log")  # span is 0..1e5 across datasets
    ax_l.set_title("(a) Tăng trưởng số cạnh theo thời gian")
    ax_l.grid(True, which="both", alpha=0.3)
    ax_l.legend(loc="lower right", fontsize=9)

    ax_r.set_xlabel("Snapshot index $t$")
    ax_r.set_ylabel(r"Mật độ $\rho^t = 2 E^t / N(N-1)$")
    ax_r.set_yscale("log")
    ax_r.set_title("(b) Mật độ tức thời theo thời gian")
    ax_r.grid(True, which="both", alpha=0.3)
    ax_r.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_plot_dataset_topology.py::test_plot_edge_growth_density_writes_nonempty_png -v`
Expected: PASS.

- [ ] **Step 5: Visually inspect**

Run: `python scripts/plot_dataset_topology.py --plots edge_growth`
Open `results/report/plots/edge_growth_density.png`. Acceptance:
- Two side-by-side panels (a) and (b)
- 6 colored lines visible in each (legend in lower-right of each)
- Log-scale Y axes labeled correctly
- LastFM line is highest in panel (a) (1.3M edges total → many per snapshot)

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_dataset_topology.py tests/test_plot_dataset_topology.py
git add results/report/plots/edge_growth_density.png
git commit -m "[plots] edge_growth_density for Slide 6"
```

---

## Task 4: Plot 3 — topology_map_2d.png + topology_map_2d_with_winners.png

**Files:**
- Modify: `scripts/plot_dataset_topology.py` (replace `plot_topology_map` stub)
- Test: `tests/test_plot_dataset_topology.py` (extend)

**Goal:** A 2-D scatter that becomes the "framework" referenced by both Slide 7 and Slide 18.
- **X axis:** mean density across snapshots, $\bar{\rho} = \frac{1}{T} \sum_t \rho^t$, log scale.
- **Y axis:** **degree-distribution Gini coefficient** on the cumulative graph. Cheap to compute (O(N log N)), captures hub/hierarchical structure (Gini → 1 = star-like, Gini → 0 = regular). Decision documented per spec §3 "Tree-likeness Gromov δ would be theoretically purer but is O(N^4); Gini is the cheap proxy that still ranks datasets sensibly along the hierarchy axis." If a reviewer pushes back, fallback = `avg_shortest_path / log(N)` (computable via `scipy.sparse.csgraph.shortest_path` on the cumulative adjacency — see Appendix in this plan).
- **Marker shape:** ◆ bipartite, ● unipartite (matplotlib markers `D` and `o`).
- **Two output files:** the plain version (Slide 7) and a version with `DATASET_WINNERS[name]` annotated as text next to each point (Slide 18).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plot_dataset_topology.py`:

```python
from scripts.plot_dataset_topology import (
    plot_topology_map,
    degree_gini,
)
import numpy as np


def test_degree_gini_uniform_is_zero():
    """A regular graph (everyone same degree) has Gini = 0."""
    degrees = np.array([4, 4, 4, 4, 4])
    assert degree_gini(degrees) == pytest.approx(0.0, abs=1e-9)


def test_degree_gini_star_approaches_one():
    """A star graph has high Gini (one hub, N-1 leaves)."""
    degrees = np.array([10] + [1] * 10)  # one hub, ten leaves
    g = degree_gini(degrees)
    assert g > 0.5  # very skewed


@pytest.mark.skipif(not _all_caches_present(), reason="All 6 caches required.")
def test_plot_topology_map_writes_both_pngs(tmp_path):
    plain = tmp_path / "topology.png"
    annotated = tmp_path / "topology_winners.png"
    plot_topology_map(plain, annotated)
    assert plain.exists() and plain.stat().st_size > 5000
    assert annotated.exists() and annotated.stat().st_size > 5000
```

- [ ] **Step 2: Run tests, verify failures**

Run: `pytest tests/test_plot_dataset_topology.py -v -k "gini or topology_map"`
Expected: 3 tests, all FAIL (`degree_gini` and the real `plot_topology_map` not implemented).

- [ ] **Step 3: Implement helpers + plot function**

In `scripts/plot_dataset_topology.py`, replace the stub for `plot_topology_map` and add helpers:

```python
def degree_gini(degrees: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.

    Formula: G = sum_i sum_j |x_i - x_j| / (2 n^2 mean(x)).
    Vectorised via sorted-cumsum trick: G = (2 sum_i i * x_(i) ) / (n sum x) - (n+1)/n.
    Returns 0 for the empty / all-zero case.
    """
    x = np.asarray(degrees, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # 1-indexed weights
    return float((2.0 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * cum[-1]) - (n + 1) / n)


def _mean_density(data: dict) -> float:
    return float(np.mean(_density_per_snapshot(data)))


def _cumulative_degree_gini(data: dict) -> float:
    """Build the cumulative undirected graph, compute degree-Gini."""
    N = int(data["num_nodes"])
    edges_set: set[tuple[int, int]] = set()
    for ei in data["edge_index"]:
        for u, v in ei.t().tolist():
            a, b = int(u), int(v)
            if a == b:
                continue
            edges_set.add((min(a, b), max(a, b)))
    degrees = np.zeros(N, dtype=np.int64)
    for u, v in edges_set:
        degrees[u] += 1
        degrees[v] += 1
    return degree_gini(degrees)


def _collect_topology_points() -> list[dict]:
    """Compute (mean density, degree Gini, bipartite flag) for every dataset."""
    pts: list[dict] = []
    for name, T, bipartite, label, color in DATASETS:
        try:
            data = load_cached_snapshots(name, T)
        except FileNotFoundError:
            print(f"Warning: missing cache for {name}; skipping in topology map.")
            continue
        pts.append({
            "name": name,
            "label": label,
            "color": color,
            "bipartite": bipartite,
            "density": _mean_density(data),
            "gini": _cumulative_degree_gini(data),
        })
    return pts


def _draw_topology_scatter(ax, points: Iterable[dict], annotate_winner: bool = False) -> None:
    for p in points:
        marker = "D" if p["bipartite"] else "o"
        ax.scatter(
            p["density"], p["gini"],
            marker=marker, s=160, c=p["color"],
            edgecolors="black", linewidth=0.8, zorder=3,
        )
        # Dataset label always (offset up-right).
        ax.annotate(
            p["label"],
            xy=(p["density"], p["gini"]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=10, zorder=4,
        )
        if annotate_winner:
            winner = DATASET_WINNERS.get(p["name"], "?")
            ax.annotate(
                f"→ {winner}",
                xy=(p["density"], p["gini"]),
                xytext=(8, -12), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#b00020", zorder=4,
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"Mật độ trung bình $\bar{\rho}$ (log scale)")
    ax.set_ylabel("Degree-distribution Gini (proxy cho phân cấp)")
    ax.grid(True, alpha=0.3)
    # Legend for marker shape.
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Unipartite"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Bipartite"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=10)


def plot_topology_map(out_plain: Path, out_with_winners: Path) -> None:
    points = _collect_topology_points()

    # --- Plain version (Slide 7) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_topology_scatter(ax, points, annotate_winner=False)
    ax.set_title("Bản đồ 6 mạng theo 2 trục cấu trúc")
    fig.tight_layout()
    fig.savefig(out_plain, dpi=150)
    plt.close(fig)

    # --- Annotated version (Slide 18) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_topology_scatter(ax, points, annotate_winner=True)
    ax.set_title("Diagnosis: cấu trúc tiến hóa ↔ mô hình thắng (AUC top-1)")
    fig.tight_layout()
    fig.savefig(out_with_winners, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_plot_dataset_topology.py -v -k "gini or topology_map"`
Expected: all PASS. (`degree_gini` tests do not need caches; the third test will skip if caches absent.)

- [ ] **Step 5: Visually inspect**

Run: `python scripts/plot_dataset_topology.py --plots topology`
Open both PNGs. Acceptance for `topology_map_2d.png`:
- 6 markers visible, each labeled with its dataset name
- LastFM and Mooc-actions sit in the dense (right) region with diamond markers
- CollegeMsg + Bitcoinotc sit in the sparse (left) region with circle markers
- Marker legend (Unipartite/Bipartite) visible at lower-right

For `topology_map_2d_with_winners.png` additionally:
- Each dataset has a bold red "→ <Model>" annotation below it
- Annotations match `DATASET_WINNERS` (CollegeMsg → HTGN, Mooc-actions → DyGNN, etc.)

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_dataset_topology.py tests/test_plot_dataset_topology.py
git add results/report/plots/topology_map_2d.png results/report/plots/topology_map_2d_with_winners.png
git commit -m "[plots] topology_map_2d + winners for Slides 7 & 18"
```

---

## Task 5: Wire the new plots into make_plots.py

**Files:**
- Modify: `scripts/make_plots.py` (main() dispatch block, around line 407)

**Goal:** Allow `python scripts/make_plots.py --plots all` (or `--plots dataset_snapshots,edge_growth,topology_map`) to also produce the three new plots, so the existing make_plots workflow stays the single entrypoint.

- [ ] **Step 1: Locate the dispatch block**

Run: `grep -n '"auc_bar" in plots' scripts/make_plots.py`
Expected: 1 line matching `if "auc_bar" in plots:` somewhere ~line 412.

- [ ] **Step 2: Add the new dispatch keys**

In `scripts/make_plots.py`, find the existing `plots = ...` default list (around line 407):

```python
    plots = args.plots.split(",") if args.plots != "all" else [
        "auc_bar", "ap_bar", "learning_curves", "ranking_heatmap",
        "beta_sensitivity", "runtime", "dataset_stats",
    ]
```

Replace it with:

```python
    plots = args.plots.split(",") if args.plots != "all" else [
        "auc_bar", "ap_bar", "learning_curves", "ranking_heatmap",
        "beta_sensitivity", "runtime", "dataset_stats",
        "dataset_snapshots", "edge_growth", "topology_map",
    ]
```

Then at the end of `main()`, just before `if __name__ == "__main__":` (or before the final `print`/return), add a new block:

```python
    if any(k in plots for k in ("dataset_snapshots", "edge_growth", "topology_map")):
        from scripts.plot_dataset_topology import (
            plot_dataset_snapshots_grid,
            plot_edge_growth_density,
            plot_topology_map,
        )
        if "dataset_snapshots" in plots:
            plot_dataset_snapshots_grid(args.out_dir / "dataset_snapshots_grid.png")
            print(f"Wrote {args.out_dir / 'dataset_snapshots_grid.png'}")
        if "edge_growth" in plots:
            plot_edge_growth_density(args.out_dir / "edge_growth_density.png")
            print(f"Wrote {args.out_dir / 'edge_growth_density.png'}")
        if "topology_map" in plots:
            plot_topology_map(
                args.out_dir / "topology_map_2d.png",
                args.out_dir / "topology_map_2d_with_winners.png",
            )
            print(f"Wrote {args.out_dir / 'topology_map_2d.png'}")
            print(f"Wrote {args.out_dir / 'topology_map_2d_with_winners.png'}")
```

- [ ] **Step 3: Run the existing test suite to confirm we didn't break anything**

Run: `pytest tests/test_make_plots.py -v`
Expected: All existing tests still PASS.

- [ ] **Step 4: Smoke-test the wiring**

Run: `python scripts/make_plots.py --plots dataset_snapshots,edge_growth,topology_map`
Expected stdout contains:
```
Wrote .../results/report/plots/dataset_snapshots_grid.png
Wrote .../results/report/plots/edge_growth_density.png
Wrote .../results/report/plots/topology_map_2d.png
Wrote .../results/report/plots/topology_map_2d_with_winners.png
```
And `ls results/report/plots/` includes all four files.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_plots.py
git commit -m "[plots] wire dataset_snapshots/edge_growth/topology_map into make_plots CLI"
```

---

## Task 6: Render every plot end-to-end + visual sanity check

**Files:** none modified; this task only runs commands and inspects outputs.

- [ ] **Step 1: Render the full plot suite**

Run: `python scripts/make_plots.py --plots all`
Expected stdout includes `Wrote` lines for: `auc_comparison.png`, `ap_comparison.png`, `ranking_heatmap.png`, `beta_sensitivity.png`, `runtime_comparison.png`, all 6 `learning_curves_*.png`, `dataset_snapshots_grid.png`, `edge_growth_density.png`, `topology_map_2d.png`, `topology_map_2d_with_winners.png`.

- [ ] **Step 2: Verify all artefacts referenced by the spec exist**

Run:
```bash
for f in auc_comparison.png ap_comparison.png ranking_heatmap.png \
         beta_sensitivity.png runtime_comparison.png \
         learning_curves_collegemsg.png learning_curves_bitcoinotc.png \
         learning_curves_eut.png learning_curves_mooc_actions.png \
         learning_curves_lastfm.png learning_curves_wikipedia.png \
         dataset_snapshots_grid.png edge_growth_density.png \
         topology_map_2d.png topology_map_2d_with_winners.png; do
  if [ ! -f "results/report/plots/$f" ]; then echo "MISSING: $f"; fi
done
```
Expected: no `MISSING:` lines printed.

- [ ] **Step 3: Verify dataset_stats.md matches the spec's Slide 4 table**

Run: `cat results/report/dataset_stats.md`
Expected: 6 rows with N, E, T, bipartite columns matching the values in spec §3 Slide 4 (collegemsg 1899/59835/47/False, etc.). If values differ, the spec's Slide 4 table needs an update before generating the Gamma outline.

- [ ] **Step 4: No commit (this task only verifies outputs of Tasks 2–5).**

---

## Task 7: Write the Gamma-friendly outline

**Files:**
- Create: `docs/slides/thesis_defense_outline.md`

**Goal:** A single Markdown file Gamma can consume to generate ~22 slides. Format constraints (from earlier Gamma generation work in this repo): one `## Slide N — Title` per slide (Gamma treats `##` as slide breaks), 3–6 bullet points per slide, no nested bullets deeper than one level, image references as `IMAGE: <relative path>` (Gamma will not auto-pull from the filesystem — these are reminders for the manual editor pass after generation).

- [ ] **Step 1: Create the outline file**

Create `docs/slides/thesis_defense_outline.md` with the full content below. The content is the verbatim translation of spec §3 into Gamma-friendly Markdown.

```markdown
# Slide deck outline — Bảo vệ luận văn

Ngôn ngữ: tiếng Việt. Thời lượng: ~20 phút. Mục tiêu: 22 slide chính + 6 phụ lục.
Trục story: temporal graph evolution → dataset-as-evolving-graph → 5 chiến lược bám thời gian → kết quả → diagnosis topology ↔ winner.

---

## Slide 1 — Tái hiện và phân tích so sánh GCN_MA cho dự đoán liên kết trên đồ thị động

- Đề tài luận văn tốt nghiệp
- Tác giả: [Tên]
- Giảng viên hướng dẫn: [Tên GVHD]
- Ngày bảo vệ: [Ngày]
- IMAGE: results/report/plots/dataset_snapshots_grid.png (3 snapshot làm hình nền)

## Slide 2 — Bài toán: dự đoán liên kết trên đồ thị động

- Cho chuỗi snapshot $G^1, G^2, \dots, G^t$, dự đoán cạnh nào sẽ xuất hiện ở $G^{t+1}$
- Khác static link prediction: phải bám đồng thời cấu trúc và thời gian
- Ứng dụng: gợi ý bạn, dự đoán giao dịch tài chính, phát hiện gian lận, gợi ý nội dung
- Câu hỏi cốt lõi: làm sao biểu diễn được sự tiến hóa của cấu trúc mạng?

## Slide 3 — Ba câu hỏi nghiên cứu

- Câu 1: Tái hiện GCN_MA (Mei & Zhao 2024) có khớp số trong paper không?
- Câu 2: So với baseline hiện đại (HTGN, DyGNN, EvolveGCN, DGCN), GCN_MA đứng ở đâu?
- **Câu 3 (trục chính):** Cấu trúc tiến hóa của mạng quyết định mô hình nào thắng như thế nào?

## Slide 4 — Sáu mạng, sáu câu chuyện tiến hóa

- CollegeMsg: 1,899 node, 59K cạnh, 47 snapshot, unipartite — tin nhắn sinh viên
- Bitcoinotc: 5,881 node, 36K cạnh, 62 snapshot, unipartite — đánh giá tin cậy
- EUT: 986 node, 332K cạnh, 127 snapshot, unipartite — email tổ chức
- Mooc-actions: 7,144 node, 412K cạnh, 72 snapshot, bipartite — học trực tuyến
- LastFM: 1,980 node, 1.29M cạnh, 41 snapshot, bipartite — nghe nhạc
- Wikipedia: 7,474 node, 110K cạnh, 42 snapshot, bipartite — chỉnh sửa wiki

## Slide 5 — Trực quan: mạng tiến hóa qua snapshot

- 3 dataset đại diện × 3 mốc thời gian (t=0, T/2, T)
- CollegeMsg sparse-unipartite: tăng trưởng tuần tự theo học kỳ
- EUT dense-unipartite: bão hòa nhanh, mọi người email lẫn nhau
- LastFM dense-bipartite: tăng trưởng siêu tuyến tính (1.3M cạnh!)
- IMAGE: results/report/plots/dataset_snapshots_grid.png

## Slide 6 — Động học của mạng theo thời gian

- (a) Số cạnh per snapshot — 6 đường, log scale
- (b) Mật độ $\rho^t = 2 E^t / N(N-1)$ — 6 đường, log scale
- LastFM tăng dốc, CollegeMsg bậc thang, EUT bão hòa sớm
- IMAGE: results/report/plots/edge_growth_density.png

## Slide 7 — Bản đồ 6 mạng theo 2 trục cấu trúc

- Trục X: mật độ trung bình $\bar{\rho}$ (log scale)
- Trục Y: degree-distribution Gini (proxy cho phân cấp)
- Marker: ◆ bipartite / ● unipartite
- Đây là khung phân tích → sẽ overlay winner ở Slide 18
- IMAGE: results/report/plots/topology_map_2d.png

## Slide 8 — GCN_MA (Mei & Zhao 2024)

- Pipeline: A^t → NRNAE → spectral GCN(W^t) → H^t → multi-head attention → Z^t → decoder
- NRNAE: $\hat{S} = A + \beta S + I$ với $S(i,j) = |N(i) \cap N(j)| \cdot \deg(i) \cdot CC(i)$
- Cơ chế bám thời gian: $W^t = \mathrm{LSTM}(W^{t-1})$ — tiến hóa qua **trọng số**, không qua trạng thái node
- Hyperparameter cố định: $\beta = 0.8$ (grid search trên Bitcoinotc validation)

## Slide 9 — EvolveGCN-O (Pareja et al. 2020)

- Pipeline: X^t → GCN(W^t) → H^t, với $W^t = \mathrm{GRU}(W^{t-1})$
- Cơ chế bám thời gian: GRU tiến hóa trọng số — phiên bản tối giản của ý tưởng "evolve-the-weights"
- 2 lớp GRCU; dùng `nn.Embedding` thay one-hot identity để tiết kiệm RAM
- Vị trí trong bức tranh: cùng họ GCN_MA, nhưng không có NRNAE và không có attention

## Slide 10 — HTGN (Yang et al. 2021) — đối thủ mạnh nhất

- Pipeline: X^t → HGCN trên Poincaré ball (c=1.0) → log map về Euclidean → Z^t
- Thêm Hyperbolic Temporal Attention qua snapshot
- Cơ chế bám thời gian: temporal attention trên không gian hyperbolic
- Lý do mạnh: đa tạp cong biểu diễn cây phân cấp với độ chính xác cao hơn Euclidean
- Forshadowing: top-1 hoặc top-2 ở 6/6 dataset

## Slide 11 — DyGNN (Ma et al. 2020) — bộ nhớ per-node

- Pipeline: mỗi cạnh đến → GRU update memory cho cả src và dst → propagate cho neighbors
- Cơ chế bám thời gian: per-node memory, event-driven thay vì snapshot-driven
- Vectorized variant (vendored) thay per-edge loop để fit GPU
- Lưu ý: N/A trên LastFM (OOM ở 1.3M cạnh) — sẽ hiển thị "—" trong bảng kết quả

## Slide 12 — DGCN (Manessi et al. 2020) — baseline đơn giản

- Pipeline: GCN per snapshot → ghép embedding qua thời gian → LSTM trên trục thời gian → decoder
- Cơ chế bám thời gian: LSTM trên chuỗi embedding (không phải trọng số)
- WD-GCN variant: chia sẻ tham số GCN giữa các snapshot
- Forshadowing: mạnh bất ngờ ở EUT, đứng top-1 (0.9847 AUC)
- Tóm tắt 5 chiến lược: GCN_MA = LSTM trên W; EvolveGCN = GRU trên W; HTGN = attention + hyperbolic; DyGNN = memory per-node; DGCN = LSTM trên embedding

## Slide 13 — Thiết lập thực nghiệm: minh bạch và tái lập

- Chia tập temporal: ~70% snapshot đầu train, ~15% val, ~15% test
- Negative sampling 1:1 (mỗi cạnh dương ghép 1 cạnh âm random)
- Hyperparameter: lấy paper khi có ($\beta$, $c$); grid-search khi paper không nêu (lr, hidden, heads, epochs)
- 3 seeds {42, 123, 2024}; Adam; early stopping theo val AUC
- Hardware: NVIDIA RTX 3060 12GB, CUDA 12.1, PyTorch 2.4.0
- Metric: AUC + AP, báo cáo mean ± std qua 3 seeds

## Slide 14 — Bảng tổng hợp 5 mô hình × 6 datasets

- GCN_MA tái hiện đúng paper ở 3/6 dataset trong ±1.5 điểm AUC, 4/6 trong ±2.5 điểm
- LastFM là outlier với gap ~7.5 điểm — sẽ thảo luận riêng ở Slide 19
- Top-1 phân bố: HTGN (collegemsg, bitcoinotc), DyGNN (mooc, wikipedia), DGCN (eut), EvolveGCN-O (lastfm)
- DyGNN×LastFM: N/A (OOM)
- (Bảng đầy đủ AUC + AP đặt trên slide này — Gamma sẽ render từ markdown table)

## Slide 15 — So sánh AUC & AP trực quan

- HTGN dẫn đầu hoặc top-2 ở 6/6 dataset — consistent strong baseline
- DyGNN bùng nổ trên mooc-actions + wikipedia (cả 2 bipartite + dense)
- GCN_MA bám sát paper nhưng tụt mạnh ở LastFM
- IMAGE: results/report/plots/auc_comparison.png
- IMAGE: results/report/plots/ap_comparison.png

## Slide 16 — Ranking heatmap: ai thắng ở đâu

- Trục dọc: 5 mô hình; trục ngang: 6 dataset; ô đậm = rank cao
- 3 điểm chú ý: HTGN top trên CollegeMsg; DyGNN top trên Wiki/Mooc; DGCN top trên EUT (hơn HTGN ~0.001)
- DyGNN×LastFM masked grey (em-dash) — OOM, không so sánh được
- IMAGE: results/report/plots/ranking_heatmap.png

## Slide 17 — Learning curves chọn lọc

- CollegeMsg (sparse, unipartite): HTGN hội tụ nhanh, GCN_MA ổn định nhưng plateau thấp hơn
- Mooc-actions (dense, bipartite): DyGNN hội tụ nhanh nhất, GCN_MA bám sát top
- 4 dataset còn lại đặt ở Appendix A2 (sẵn sàng nếu hội đồng hỏi)
- IMAGE: results/report/plots/learning_curves_collegemsg.png
- IMAGE: results/report/plots/learning_curves_mooc_actions.png

## Slide 18 — Diagnosis: cấu trúc tiến hóa ↔ mô hình thắng

- Mệnh đề 1: Cấu trúc phân cấp ẩn → hyperbolic ăn điểm (HTGN top-1 ở CollegeMsg, Bitcoinotc)
- Mệnh đề 2: Sự kiện dồn dập trên bipartite dày → bộ nhớ per-node ăn điểm (DyGNN top-1 ở Mooc-actions, Wikipedia)
- Mệnh đề 3: Mạng dày không cấu trúc đặc thù → baseline đơn giản đủ tốt (DGCN top-1 ở EUT)
- Cấu trúc tiến hóa là tín hiệu thiết kế — không có mô hình một-cỡ-vừa-cho-tất-cả
- IMAGE: results/report/plots/topology_map_2d_with_winners.png

## Slide 19 — Nơi GCN_MA tỏa sáng và tụt hậu

- Tỏa sáng: Mooc-actions gap 0.4, Wikipedia 0.5, CollegeMsg 1.4 — 3/6 trong ±1.5 điểm AUC
- EUT (gap 2.1) và Bitcoinotc (gap 5.6) lệch xa hơn nhưng vẫn hợp lý qua 3 seeds
- Tụt hậu: LastFM gap ~7.5 — giả thuyết NRNAE bão hòa khi mạng siêu dense (1.3M cạnh)
- Thua HTGN/DyGNN ở 4/6 dataset (top-1)
- β = 0.8 là validation choice, không cherry-picked
- IMAGE: results/report/plots/beta_sensitivity.png

## Slide 20 — Runtime & trade-off: bức tranh đầy đủ

- GCN_MA thuộc nhóm nhẹ (LSTM trên W, NRNAE tính 1 lần)
- HTGN chậm 3–5× vì hyperbolic ops (exp_map, log_map)
- DyGNN chậm nhất do per-edge update tuần tự
- Bảo vệ paper gốc: GCN_MA không SOTA accuracy, nhưng là sweet spot accuracy/cost
- IMAGE: results/report/plots/runtime_comparison.png

## Slide 21 — Đóng góp & bài học

- Đóng góp 1: Tái hiện thành công GCN_MA — 3/6 dataset trong ±1.5 điểm AUC, code mở
- Đóng góp 2: Mở rộng so sánh với 4 baseline hiện đại trên cùng pipeline → fair-comparison mà paper gốc thiếu
- Đóng góp 3: Khung diagnosis cấu trúc-tiến-hóa ↔ mô hình để chọn baseline phù hợp
- Bài học: cấu trúc đồ thị là tín hiệu chọn mô hình; GCN_MA là sweet spot; paper gốc thiếu reproducibility
- Hướng mở rộng: NRNAE + LSTM-W kết hợp với hyperbolic hoặc per-node memory

## Slide 22 — Xin cảm ơn — Q&A

- Tên + email + tên GVHD
- QR code → GitHub repo (code + reproduction-log + thesis chapter PDF)
- Sẵn sàng nhận câu hỏi
- IMAGE: results/report/plots/dataset_snapshots_grid.png (hình nền lặp lại để đóng khung visual)

---

## Appendix (6 slide phụ — không trình bày mặc định, dùng cho Q&A)

## Slide A1 — Hyperparameter chi tiết 5 mô hình

- GCN_MA: hidden=128, β=0.8, heads=4, lr=1e-3, epochs=100, early-stop patience=20
- EvolveGCN-O: 2 lớp GRCU, hidden=128, lr=1e-3
- HTGN: c=1.0, hidden=128, lr=1e-3, patience=20
- DyGNN: memory_dim=128, lr=1e-3, vectorized variant
- DGCN: 2 lớp GCN per snapshot, LSTM hidden=128, WD-GCN variant
- Toàn bộ chi tiết trong `results/configs_runtime/`

## Slide A2 — Learning curves còn lại

- IMAGE: results/report/plots/learning_curves_bitcoinotc.png
- IMAGE: results/report/plots/learning_curves_eut.png
- IMAGE: results/report/plots/learning_curves_lastfm.png
- IMAGE: results/report/plots/learning_curves_wikipedia.png

## Slide A3 — Deviation chi tiết so với paper gốc

- collegemsg: gap −1.44 điểm AUC (within ±1.5)
- bitcoinotc: gap −5.60
- eut: gap −2.14
- mooc_actions: gap −0.35 (within ±1.5)
- lastfm: gap −7.53 (outlier, NRNAE saturation hypothesis)
- wikipedia: gap −0.46 (within ±1.5)

## Slide A4 — Chi tiết NRNAE với ví dụ tính toán

- Ví dụ: graph 4 node, cạnh {(1,2), (2,3), (3,1), (3,4)}
- $\deg = [2, 2, 3, 1]$
- $CC(1) = 1.0$ (hai hàng xóm 2,3 nối nhau), $CC(2)=1.0$, $CC(3)=0.33$, $CC(4)=0$
- $AS = [2, 2, 1, 0]$
- $S(1,2) = |\{3\}| \cdot AS(1) = 2$; tương tự cho các cặp khác
- $\hat{S} = A + 0.8 \cdot S + I$

## Slide A5 — Notes triển khai khó

- DyGNN OOM trên LastFM (1.3M cạnh) → bỏ dataset này khỏi run
- HTGN: upstream gọi argparse khi import → workaround reset sys.argv
- HTGN: hidden_initial không phải nn.Parameter → override .to() để rebuild trên target device
- EvolveGCN: PyTorch 2.4 patch nâng GRCU_layers thành nn.ModuleList; restore _parameters = {}
- Adjacency symmetrize để bipartite không bị triệt tiêu về zero embedding

## Slide A6 — Link tài nguyên

- GitHub repo: [URL]
- Reproduction log: docs/reproduction-log.md
- Thesis chapter PDF: docs/thesis_chapter.pdf
- Strengths/weaknesses analysis: docs/gcn-ma-strengths-weaknesses.pdf
```

- [ ] **Step 2: Verify the outline parses as Markdown and has 22 + 6 slide headings**

Run:
```bash
grep -c "^## Slide " docs/slides/thesis_defense_outline.md
```
Expected output: `28` (22 main slides + 6 appendix slides).

- [ ] **Step 3: Commit**

```bash
git add docs/slides/thesis_defense_outline.md
git commit -m "[slides] add Gamma-friendly Vietnamese outline (22 + 6 slides)"
```

---

## Task 8: Generate the Gamma deck via MCP

**Files:**
- Create: `docs/slides/gamma_deck_url.txt` (records the returned share URL/ID)

**Goal:** Convert the outline into an actual Gamma deck. Gamma will not pull images from the filesystem; we will note in the deck description that the user should upload the `results/report/plots/*.png` images manually in the Gamma editor before final export. This is acceptable for the spec (see spec §5 Bước 4 "Chỉnh tay trong Gamma editor").

- [ ] **Step 1: Confirm the Gamma MCP tools are available**

Use `ToolSearch` with query `select:mcp__claude_ai_Gamma__generate,mcp__claude_ai_Gamma__get_generation_status` to load schemas. If not present, ask the user to enable the Gamma MCP server before proceeding.

- [ ] **Step 2: Prepare the Gamma generation payload**

Read `docs/slides/thesis_defense_outline.md` and use it as the `inputText` (or equivalent) parameter. Key tuning:
- Language: Vietnamese (Gamma auto-detects from content)
- Number of cards: ~28 (matches `## Slide` headings)
- Theme: clean / academic (Gamma chooses defaults; the user can re-theme post-generation)
- Format: 16:9 presentation
- Tone: professional / academic

**Important:** Do **not** combine outline edits with the generation call. If the generated deck is wrong, fix the outline first, commit, then re-call.

- [ ] **Step 3: Call `mcp__claude_ai_Gamma__generate`**

Pass the outline content. The exact parameter schema is exposed when you load the tool via ToolSearch — follow that schema. Suggested prompt suffix to append:

> "Hãy giữ nguyên thứ tự slide và tiêu đề. Mỗi `## Slide N — Title` là một slide riêng. Các dòng `IMAGE: <path>` là placeholder cho ảnh sẽ được người dùng upload sau — đặt một khung trống ở vị trí phù hợp. Ngôn ngữ tiếng Việt. Theme: clean academic, ít màu mè."

- [ ] **Step 4: Wait for generation, record URL**

Poll `mcp__claude_ai_Gamma__get_generation_status` if the generation is async (depends on the MCP server behaviour). Once complete, capture the share URL and Gamma file ID.

Create `docs/slides/gamma_deck_url.txt` containing:

```
Gamma deck for thesis defense — generated <ISO timestamp>
Share URL: <URL returned by MCP>
File ID:   <ID returned by MCP>
Outline source: docs/slides/thesis_defense_outline.md
Manual editor steps still required:
  1. Upload all PNGs from results/report/plots/ into the corresponding IMAGE: placeholders
  2. Adjust theme if desired
  3. Export to PDF for backup; export to PPTX for hand-off
```

- [ ] **Step 5: Spot-check the deck count and titles**

Open the share URL in a browser (or have the user do so). Acceptance:
- 22 main slides + 6 appendix = 28 cards in total
- Slide 1 title matches "Tái hiện và phân tích so sánh GCN_MA..."
- Slide 18 title matches "Diagnosis: cấu trúc tiến hóa ↔ mô hình thắng"
- Slide A6 is the final card

If counts mismatch, the outline likely had a heading Gamma didn't recognise — edit the outline, commit, regenerate.

- [ ] **Step 6: Commit**

```bash
git add docs/slides/gamma_deck_url.txt
git commit -m "[slides] generate Gamma deck — share URL recorded"
```

---

## Task 9: Final aggregation — update README + close out

**Files:**
- Modify: `README.md` (add a "Slides" section pointing at the outline + the spec)

- [ ] **Step 1: Append a Slides section to README**

Open `README.md` and after the existing `## Layout` section, add:

```markdown

## Slides

- Outline (Gamma input):  `docs/slides/thesis_defense_outline.md`
- Spec:                    `docs/superpowers/specs/2026-05-18-thesis-slides-design.md`
- Plan:                    `docs/superpowers/plans/2026-05-18-thesis-slides.md`
- Generated deck URL:      `docs/slides/gamma_deck_url.txt`
- Topology plots:          `results/report/plots/topology_map_2d*.png`, `dataset_snapshots_grid.png`, `edge_growth_density.png`
```

- [ ] **Step 2: Final smoke test — run the full plot pipeline once more**

Run: `python scripts/make_plots.py --plots all` and check stdout for no warnings/errors.

- [ ] **Step 3: Commit + done**

```bash
git add README.md
git commit -m "[docs] README: link to thesis defense slides spec/plan/outline"
```

---

## Appendix: Fallback Y-axis metric for topology map

If a reviewer rejects degree-Gini as a hierarchy proxy (legitimate critique: Gini captures hub-ness, not strictly tree-likeness), swap in **average shortest path / log N** computed on the cumulative undirected graph.

Replacement helper for `_cumulative_degree_gini` in `scripts/plot_dataset_topology.py`:

```python
def _cumulative_avg_path_over_logN(data: dict) -> float:
    """avg_shortest_path / log(N) — close to 1 for tree-like, much smaller for cliques."""
    import scipy.sparse
    import scipy.sparse.csgraph
    N = int(data["num_nodes"])
    rows: list[int] = []
    cols: list[int] = []
    for ei in data["edge_index"]:
        for u, v in ei.t().tolist():
            rows.append(int(u)); cols.append(int(v))
            rows.append(int(v)); cols.append(int(u))  # symmetrize
    if not rows:
        return 0.0
    mat = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(N, N),
    )
    # Largest connected component only (otherwise shortest_path returns inf).
    n_cc, labels = scipy.sparse.csgraph.connected_components(mat, directed=False)
    largest_label = int(np.bincount(labels).argmax())
    mask = labels == largest_label
    sub = mat[mask][:, mask]
    # Sample up to 500 sources for speed.
    rng = np.random.default_rng(seed=0)
    n_sub = sub.shape[0]
    sources = rng.choice(n_sub, size=min(500, n_sub), replace=False)
    dists = scipy.sparse.csgraph.shortest_path(sub, indices=sources, unweighted=True)
    dists = dists[np.isfinite(dists)]
    dists = dists[dists > 0]
    if dists.size == 0 or n_sub < 2:
        return 0.0
    avg_d = float(dists.mean())
    return avg_d / np.log(n_sub)
```

To switch metrics: rename the function call in `_collect_topology_points` from `_cumulative_degree_gini(data)` to `_cumulative_avg_path_over_logN(data)`, and update the Y-axis label in `_draw_topology_scatter` from `"Degree-distribution Gini (proxy cho phân cấp)"` to `r"Avg shortest path / $\log N$ (tree-likeness proxy)"`. Re-run `python scripts/make_plots.py --plots topology_map`.
