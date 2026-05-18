"""Generate all plots and stats tables from metrics JSONL + parsed training curves.

Usage:
    python scripts/make_plots.py                 # generate all assets
    python scripts/make_plots.py --plots auc_bar # subset (comma-separated)
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

# Ensure the repo root is on sys.path so `src` is importable when running
# this script directly (e.g. `python scripts/make_plots.py`).
_REPO_ROOT_CANDIDATE = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_ORDER = ["gcn_ma", "evolvegcn_o", "htgn", "dygnn", "dgcn"]
DATASETS_ORDER = ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]
MODEL_LABELS = {
    "gcn_ma": "GCN_MA",
    "evolvegcn_o": "EvolveGCN-O",
    "htgn": "HTGN",
    "dygnn": "DyGNN",
    "dgcn": "DGCN",
}
DATASET_LABELS = {
    "collegemsg": "CollegeMsg",
    "bitcoinotc": "Bitcoinotc",
    "eut": "EUT",
    "mooc_actions": "Mooc-Actions",
    "lastfm": "LastFM",
    "wikipedia": "Wikipedia",
}


# --------------------------------------------------------------------------
# Data loading + aggregation
# --------------------------------------------------------------------------

def load_metrics(path: Path, models: Optional[list[str]] = None) -> pd.DataFrame:
    """Load metrics.jsonl. Optionally filter to specific models."""
    records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(records)
    if models is not None:
        df = df[df["model"].isin(models)].copy()
    return df


def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models 1..N per dataset by descending AUC mean.

    Returns a wide DataFrame: rows = models, cols = datasets, values = rank.
    """
    agg = df.groupby(["model", "dataset"])["auc"].mean().reset_index()
    ranks = (
        agg.assign(rank=agg.groupby("dataset")["auc"].rank(method="min", ascending=False).astype(int))
           .pivot(index="model", columns="dataset", values="rank")
    )
    return ranks


# --------------------------------------------------------------------------
# Plot: AUC/AP grouped bar (5 models × 6 datasets, error bars ±std)
# --------------------------------------------------------------------------

def plot_auc_comparison(df: pd.DataFrame, out_path: Path, metric: str = "auc") -> None:
    """Grouped bar chart with error bars; one group per dataset, one bar per model."""
    agg = df.groupby(["model", "dataset"])[metric].agg(["mean", "std"]).reset_index()
    datasets = [d for d in DATASETS_ORDER if d in agg["dataset"].unique()]
    # Also include any datasets not in DATASETS_ORDER (e.g. test data "x")
    extra_datasets = [d for d in agg["dataset"].unique() if d not in DATASETS_ORDER]
    datasets = datasets + extra_datasets

    models = [m for m in MODELS_ORDER if m in agg["model"].unique()]
    # Also include any models not in MODELS_ORDER
    extra_models = [m for m in agg["model"].unique() if m not in MODELS_ORDER]
    models = models + extra_models

    x = np.arange(len(datasets))
    width = 0.15
    n_models = len(models)
    # Center bars around x positions
    offsets = np.arange(n_models) - (n_models - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        means = []
        stds = []
        for ds in datasets:
            row = agg[(agg["model"] == model) & (agg["dataset"] == ds)]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(row["mean"].iloc[0])
                stds.append(row["std"].iloc[0] if not pd.isna(row["std"].iloc[0]) else 0)
        label = MODEL_LABELS.get(model, model)
        ax.bar(x + offsets[i] * width, means, width, yerr=stds, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=15)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} comparison — {n_models} baselines × {len(datasets)} datasets (mean ± std, 3 seeds)")
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc="lower right", ncol=min(n_models, 5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot: Learning curves per dataset (5-model mean ± std band)
# --------------------------------------------------------------------------

def plot_learning_curves(curves_df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    """One PNG: val_auc vs epoch, 5 model lines (mean across seeds, ±std band)."""
    sub = curves_df[curves_df["dataset"] == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    # Use canonical order first; fall back to actual unique models in data
    canonical = [m for m in MODELS_ORDER if m in sub["model"].unique()]
    other = [m for m in sub["model"].unique() if m not in MODELS_ORDER]
    models_to_plot = canonical + other
    for model in models_to_plot:
        msub = sub[sub["model"] == model]
        if msub.empty:
            continue
        agg = msub.groupby("epoch")["val_auc"].agg(["mean", "std"]).reset_index()
        label = MODEL_LABELS.get(model, model)
        ax.plot(agg["epoch"], agg["mean"], label=label, linewidth=1.5)
        ax.fill_between(
            agg["epoch"],
            agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            alpha=0.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_auc")
    title_label = DATASET_LABELS.get(dataset, dataset)
    ax.set_title(f"Learning curves — {title_label} (mean ± std, 3 seeds)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot: Ranking heatmap (5 × 6, rank colors)
# --------------------------------------------------------------------------

def plot_ranking_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of per-dataset ranks. Cell = rank 1-5; color from green (1) to red (5)."""
    import seaborn as sns
    ranks = compute_ranking(df)
    # Reorder rows/columns to canonical order if available; otherwise use actual unique values
    canonical_rows = [m for m in MODELS_ORDER if m in ranks.index]
    canonical_cols = [d for d in DATASETS_ORDER if d in ranks.columns]
    rows = canonical_rows if canonical_rows else list(ranks.index)
    cols = canonical_cols if canonical_cols else list(ranks.columns)
    ranks = ranks.loc[rows, cols]
    # Build annotation grid: integer rank, "—" for NaN (missing model×dataset)
    annot = ranks.copy().astype(object)
    for r in rows:
        for c in cols:
            v = ranks.loc[r, c]
            annot.loc[r, c] = "—" if pd.isna(v) else f"{int(v)}"
    # Mask NaN cells so they render grey (no fill color)
    mask = ranks.isna()
    labels_x = [DATASET_LABELS.get(c, c) for c in cols]
    labels_y = [MODEL_LABELS.get(r, r) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        ranks,
        annot=annot.values,
        fmt="",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Rank"},
        xticklabels=labels_x,
        yticklabels=labels_y,
        ax=ax,
        linewidths=0.5,
        mask=mask,
    )
    ax.set_title("Ranking per dataset (1 = best AUC mean; — = not run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot: β sensitivity (from Plan 2 grid data)
# --------------------------------------------------------------------------

def plot_beta_sensitivity(beta_jsonl: Path, out_path: Path) -> None:
    """Line plot: β vs val_auc, separate line per hidden_dim."""
    records = [json.loads(l) for l in Path(beta_jsonl).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 5))
    for hd in sorted(df["hidden_dim"].unique()):
        sub = df[df["hidden_dim"] == hd].sort_values("beta")
        ax.plot(sub["beta"], sub["val_auc"], marker="o", linewidth=2, label=f"hidden_dim={hd}")
    ax.set_xlabel("β (NRNAE weight)")
    ax.set_ylabel("val_auc")
    ax.set_title("β sensitivity — Bitcoinotc, seed 42, 50 epochs (Plan 2 grid)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot: Runtime comparison
# --------------------------------------------------------------------------

def plot_runtime_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar of total wall-clock seconds per (model, dataset), log y."""
    agg = df.groupby(["model", "dataset"])["runtime_s"].sum().reset_index()
    datasets = [d for d in DATASETS_ORDER if d in agg["dataset"].unique()]
    models = [m for m in MODELS_ORDER if m in agg["model"].unique()]
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        ys = []
        for ds in datasets:
            row = agg[(agg["model"] == model) & (agg["dataset"] == ds)]
            ys.append(row["runtime_s"].iloc[0] if len(row) else 0)
        ax.bar(x + (i - 2) * width, ys, width, label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=15)
    ax.set_ylabel("Total wall-clock (s, log scale, 3 seeds)")
    ax.set_yscale("log")
    ax.set_title("Total runtime per (model, dataset) — log scale, sum of 3 seeds")
    ax.legend(loc="upper left", ncol=5)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Dataset stats table
# --------------------------------------------------------------------------

def write_dataset_stats(stats: list[dict], out_path: Path) -> None:
    """Write a markdown table of dataset statistics."""
    if not stats:
        out_path.write_text("(no datasets)\n")
        return
    keys = list(stats[0].keys())

    # Format keys: keep N/E/T uppercase, capitalize others
    def label(k: str) -> str:
        return k if k in ("N", "E", "T") else k.capitalize()

    header = "| " + " | ".join(label(k) for k in keys) + " |"
    sep = "|" + "|".join(["---"] * len(keys)) + "|"
    rows = []
    for row in stats:
        rows.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([header, sep] + rows) + "\n")


# --------------------------------------------------------------------------
# Dataset stats computation
# --------------------------------------------------------------------------

def _stats_from_cache(name: str, T: int, bipartite: bool) -> dict | None:
    """Fast path: read N, E directly from the pre-built .pt cache file.

    Cache files are dicts with keys: features, edge_index, edge_ts,
    num_nodes, num_time_steps.  beta only affects features, not topology,
    so any cache file for this dataset+T is fine.
    """
    import torch
    cache_dir = REPO_ROOT / "data" / "processed" / name
    pattern = f"{name}_T{T}_*.pt"
    candidates = sorted(cache_dir.glob(pattern))
    if not candidates:
        return None
    data = torch.load(candidates[0], weights_only=False)
    total_edges = sum(ei.shape[1] for ei in data["edge_index"])
    return {
        "dataset": name,
        "N": int(data["num_nodes"]),
        "E": int(total_edges),
        "T": T,
        "bipartite": bipartite,
    }


def compute_dataset_stats() -> list[dict]:
    """Compute N, E, T, bipartite for each dataset.

    Fast path: read directly from the pre-built .pt cache files (avoids the
    slow loader pipeline).  Falls back to running the loader if no cache exists.
    """
    datasets = [
        ("collegemsg",   47,  False),
        ("bitcoinotc",   62,  False),
        ("eut",          127, False),
        ("mooc_actions", 72,  True),
        ("lastfm",       41,  True),
        ("wikipedia",    42,  True),
    ]
    raw_paths = {
        "collegemsg":   "data/raw/collegemsg/CollegeMsg.txt.gz",
        "bitcoinotc":   "data/raw/bitcoinotc/soc-sign-bitcoinotc.csv.gz",
        "eut":          "data/raw/eut/email-Eu-core-temporal.txt.gz",
        "mooc_actions": "data/raw/mooc_actions/mooc.csv",
        "lastfm":       "data/raw/lastfm/lastfm.csv",
        "wikipedia":    "data/raw/wikipedia/wikipedia.csv",
    }

    out: list[dict] = []
    for name, T, bipartite in datasets:
        # --- fast path: read from cache ---
        try:
            row = _stats_from_cache(name, T, bipartite)
            if row is not None:
                out.append(row)
                continue
        except Exception as e:
            print(f"Warning: cache read failed for {name}: {e}")

        # --- slow path: run the full loader ---
        try:
            from importlib import import_module
            loader_map = {
                "collegemsg":   "src.data.loaders.collegemsg.CollegeMsgLoader",
                "bitcoinotc":   "src.data.loaders.bitcoinotc.BitcoinotcLoader",
                "eut":          "src.data.loaders.eut.EUTLoader",
                "mooc_actions": "src.data.loaders.mooc_actions.MoocActionsLoader",
                "lastfm":       "src.data.loaders.lastfm.LastFMLoader",
                "wikipedia":    "src.data.loaders.wikipedia.WikipediaLoader",
            }
            mod_path, cls_name = loader_map[name].rsplit(".", 1)
            loader = getattr(import_module(mod_path), cls_name)()
            g = loader.build(
                raw_path=REPO_ROOT / raw_paths[name],
                cache_dir=REPO_ROOT / f"data/processed/{name}",
                num_time_steps=T,
                beta=0.8,
            )
            total_edges = sum(s.edge_index.shape[1] for s in g.snapshots)
            out.append({
                "dataset": name,
                "N": g.num_nodes,
                "E": int(total_edges),
                "T": T,
                "bipartite": bipartite,
            })
        except Exception as e:
            out.append({"dataset": name, "N": "N/A", "E": "N/A", "T": T, "bipartite": bipartite})
            print(f"Warning: failed to load {name}: {e}")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "results" / "metrics.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "report" / "plots",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Comma-separated subset; or 'all' for everything.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.metrics, models=MODELS_ORDER)
    plots = args.plots.split(",") if args.plots != "all" else [
        "auc_bar", "ap_bar", "learning_curves", "ranking_heatmap",
        "beta_sensitivity", "runtime", "dataset_stats",
        "dataset_snapshots", "edge_growth", "topology_map",
    ]

    if "auc_bar" in plots:
        plot_auc_comparison(df, args.out_dir / "auc_comparison.png", metric="auc")
        print(f"Wrote {args.out_dir / 'auc_comparison.png'}")
    if "ap_bar" in plots:
        plot_auc_comparison(df, args.out_dir / "ap_comparison.png", metric="ap")
        print(f"Wrote {args.out_dir / 'ap_comparison.png'}")
    if "learning_curves" in plots:
        curves_path = REPO_ROOT / "results" / "report" / "training_curves.jsonl"
        if not curves_path.exists():
            print(f"Skipping learning_curves — run scripts/parse_training_logs.py first to produce {curves_path}")
        else:
            curves_records = [json.loads(l) for l in curves_path.read_text().splitlines() if l.strip()]
            curves_df = pd.DataFrame(curves_records)
            for ds in DATASETS_ORDER:
                if (curves_df["dataset"] == ds).any():
                    out_p = args.out_dir / f"learning_curves_{ds}.png"
                    plot_learning_curves(curves_df, ds, out_p)
                    print(f"Wrote {out_p}")
    if "ranking_heatmap" in plots:
        plot_ranking_heatmap(df, args.out_dir / "ranking_heatmap.png")
        print(f"Wrote {args.out_dir / 'ranking_heatmap.png'}")
    if "beta_sensitivity" in plots:
        beta_p = REPO_ROOT / "results" / "beta_grid_bitcoinotc.jsonl"
        plot_beta_sensitivity(beta_p, args.out_dir / "beta_sensitivity.png")
        print(f"Wrote {args.out_dir / 'beta_sensitivity.png'}")
    if "runtime" in plots:
        plot_runtime_comparison(df, args.out_dir / "runtime_comparison.png")
        print(f"Wrote {args.out_dir / 'runtime_comparison.png'}")
    if "dataset_stats" in plots:
        stats = compute_dataset_stats()
        out = args.out_dir.parent / "dataset_stats.md"
        write_dataset_stats(stats, out)
        print(f"Wrote {out}")

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


if __name__ == "__main__":
    main()
