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
