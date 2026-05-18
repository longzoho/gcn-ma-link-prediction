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
from typing import Iterable  # noqa: F401  # used by helpers added in Tasks 2-4

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
    ("mooc_actions", 72,  True,  "Mooc-Actions", "#d62728"),
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


# Plot functions for later tasks.
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


if __name__ == "__main__":
    main()
