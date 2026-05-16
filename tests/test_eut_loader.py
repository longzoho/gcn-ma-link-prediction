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
