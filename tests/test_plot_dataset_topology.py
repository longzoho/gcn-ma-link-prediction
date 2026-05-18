"""Smoke tests for dataset-topology plots."""
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_dataset_topology import (
    DATASETS,
    degree_gini,
    load_cached_snapshots,
    plot_dataset_snapshots_grid,
    plot_edge_growth_density,
    plot_topology_map,
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


def _all_caches_present() -> bool:
    return all(_has_cache(n, T) for n, T, *_ in DATASETS)


@pytest.mark.skipif(not _all_caches_present(), reason="All 6 caches required.")
def test_plot_edge_growth_density_writes_nonempty_png(tmp_path):
    out = tmp_path / "edge_growth.png"
    plot_edge_growth_density(out)
    assert out.exists()
    assert out.stat().st_size > 5000


def test_degree_gini_uniform_is_zero():
    """A regular graph (everyone same degree) has Gini = 0."""
    degrees = np.array([4, 4, 4, 4, 4])
    assert degree_gini(degrees) == pytest.approx(0.0, abs=1e-9)


def test_degree_gini_star_approaches_one():
    """A hub-and-spoke degree sequence has high Gini (very skewed distribution)."""
    degrees = np.array([50] + [1] * 10)  # one dominant hub, ten leaves
    g = degree_gini(degrees)
    assert g > 0.5  # very skewed


@pytest.mark.skipif(not _all_caches_present(), reason="All 6 caches required.")
def test_plot_topology_map_writes_both_pngs(tmp_path):
    plain = tmp_path / "topology.png"
    annotated = tmp_path / "topology_winners.png"
    plot_topology_map(plain, annotated)
    assert plain.exists() and plain.stat().st_size > 5000
    assert annotated.exists() and annotated.stat().st_size > 5000
