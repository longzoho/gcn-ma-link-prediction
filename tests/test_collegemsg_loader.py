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
