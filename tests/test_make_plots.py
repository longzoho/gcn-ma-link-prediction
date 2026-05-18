"""Smoke tests for the plot generator."""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_plots import (
    load_metrics,
    compute_ranking,
    plot_auc_comparison,
    write_dataset_stats,
)


# --------------------------------------------------------------------------
# load_metrics
# --------------------------------------------------------------------------

def test_load_metrics_filters_models(tmp_path):
    """load_metrics returns a DataFrame restricted to requested models."""
    src = tmp_path / "metrics.jsonl"
    src.write_text(
        json.dumps({"model": "gcn_ma", "dataset": "x", "seed": 42, "auc": 0.9, "ap": 0.9, "runtime_s": 100}) + "\n"
        + json.dumps({"model": "evolvegcn_o", "dataset": "x", "seed": 42, "auc": 0.85, "ap": 0.85, "runtime_s": 200}) + "\n"
        + json.dumps({"model": "htgn", "dataset": "x", "seed": 42, "auc": 0.92, "ap": 0.92, "runtime_s": 300}) + "\n"
    )
    df = load_metrics(src, models=["gcn_ma", "htgn"])
    assert set(df["model"].unique()) == {"gcn_ma", "htgn"}
    assert len(df) == 2


# --------------------------------------------------------------------------
# compute_ranking
# --------------------------------------------------------------------------

def test_compute_ranking_returns_per_dataset_ranks():
    """For each dataset, rank models 1..N by descending AUC mean."""
    df = pd.DataFrame([
        {"model": "a", "dataset": "x", "auc": 0.9},
        {"model": "b", "dataset": "x", "auc": 0.95},
        {"model": "c", "dataset": "x", "auc": 0.85},
    ])
    ranks = compute_ranking(df)
    # ranks is a wide DataFrame: rows=models, cols=datasets, values=rank
    assert ranks.loc["b", "x"] == 1   # highest auc -> rank 1
    assert ranks.loc["a", "x"] == 2
    assert ranks.loc["c", "x"] == 3


# --------------------------------------------------------------------------
# plot_auc_comparison
# --------------------------------------------------------------------------

def test_plot_auc_comparison_writes_nonempty_png(tmp_path):
    """The plot function writes a non-empty PNG."""
    df = pd.DataFrame([
        {"model": "gcn_ma", "dataset": "x", "seed": s, "auc": 0.9 + 0.001 * s, "ap": 0.9}
        for s in [42, 123, 2024]
    ] + [
        {"model": "htgn", "dataset": "x", "seed": s, "auc": 0.95 + 0.001 * s, "ap": 0.95}
        for s in [42, 123, 2024]
    ])
    out = tmp_path / "auc.png"
    plot_auc_comparison(df, out, metric="auc")
    assert out.exists()
    assert out.stat().st_size > 1000  # at least 1 KB of PNG


# --------------------------------------------------------------------------
# write_dataset_stats
# --------------------------------------------------------------------------

def test_write_dataset_stats_creates_markdown_table(tmp_path):
    """write_dataset_stats produces a markdown table with the expected columns."""
    stats = [
        {"dataset": "collegemsg", "N": 1899, "E": 59835, "T": 47, "bipartite": False},
        {"dataset": "mooc_actions", "N": 7144, "E": 411749, "T": 72, "bipartite": True},
    ]
    out = tmp_path / "dataset_stats.md"
    write_dataset_stats(stats, out)
    content = out.read_text()
    assert "| Dataset |" in content or "| dataset |" in content.lower()
    assert "collegemsg" in content
    assert "mooc_actions" in content
    assert "1899" in content
    assert "411749" in content
