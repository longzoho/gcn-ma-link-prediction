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
def plot_edge_growth_density(out_path: Path) -> None: ...
def plot_topology_map(out_plain: Path, out_with_winners: Path) -> None: ...


if __name__ == "__main__":
    main()
