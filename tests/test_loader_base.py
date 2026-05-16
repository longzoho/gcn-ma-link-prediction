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


class _QuantileDummyLoader(SNAPTemporalLoader):
    dataset_name = "quantile_dummy"
    preprocess_version = "v1"
    binning_strategy = "quantile"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])


def test_quantile_binning_produces_no_empty_bins(tmp_path: Path):
    """A bursty timeline that would have empty equal-time bins must have
    non-empty bins under quantile strategy.

    The data has 30 events clustered in ts=[0..29] and 30 events clustered
    in ts=[970..999]. Equal-time bins with T=6 over [0..999] give bins of
    width ~167, so the middle 4 bins are empty. Quantile binning puts ~10
    events per bin so every bin is non-empty.
    """
    rows = []
    # 30 events in the first ~3% of the time range
    for i in range(30):
        rows.append(f"{i % 5} {(i + 1) % 5} {i}")
    # 30 events in the last ~3% of the time range
    for i in range(30):
        rows.append(f"{i % 5} {(i + 1) % 5} {970 + i}")
    raw = "\n".join(rows)
    raw_path = tmp_path / "raw.txt.gz"
    with gzip.open(raw_path, "wt") as f:
        f.write(raw)

    g = _QuantileDummyLoader().build(
        raw_path=raw_path,
        cache_dir=tmp_path / "cache",
        num_time_steps=6,
        beta=0.8,
    )
    # With quantile binning, every snapshot must be non-empty.
    empty = [t for t, s in enumerate(g.snapshots) if s.edge_index.shape[1] == 0]
    assert empty == [], f"quantile binning still produced empty snapshots at {empty}"
