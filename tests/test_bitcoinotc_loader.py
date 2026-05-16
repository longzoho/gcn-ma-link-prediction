import gzip
from pathlib import Path

import pandas as pd
import torch

from src.data.loaders.bitcoinotc import BitcoinotcLoader


def _write_csv_gz(tmp_path: Path, csv_text: str) -> Path:
    p = tmp_path / "soc-sign-bitcoin-otc.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(csv_text)
    return p


def test_parse_bitcoinotc_drops_rating(tmp_path: Path):
    """Raw is `src,dst,rating,ts`. Loader must keep src/dst/ts and drop rating."""
    csv = "1,2,5,1000\n1,3,-2,1100\n2,3,10,1200\n"
    p = _write_csv_gz(tmp_path, csv)
    df = BitcoinotcLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3
    assert df.iloc[0].tolist() == [1, 2, 1000]


def test_build_bitcoinotc_dynamic_graph(tmp_path: Path):
    rows = [f"{i % 10},{(i + 1) % 10},5,{1000 + i}" for i in range(40)]
    p = _write_csv_gz(tmp_path, "\n".join(rows))
    loader = BitcoinotcLoader()
    g = loader.build(
        raw_path=p,
        cache_dir=tmp_path / "cache",
        num_time_steps=4,
        beta=0.8,
    )
    assert g.dataset_name == "bitcoinotc"
    assert g.num_nodes == 10
    assert g.num_time_steps == 4
    for snap in g.snapshots:
        assert snap.x.shape == (10, 3)
        assert snap.S_hat.shape == (10, 10)
