"""Loader for SNAP CollegeMsg temporal network."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class CollegeMsgLoader(SNAPTemporalLoader):
    dataset_name = "collegemsg"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        """Parse SNAP CollegeMsg `.txt.gz` (whitespace `src dst ts`) → DataFrame."""
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
        return df.astype({"src": int, "dst": int, "ts": int})


def build_dynamic_graph(raw_gz_path: Path, num_time_steps: int, beta: float):
    """Backwards-compatible function exposed for `scripts/train.py`.

    Cache directory is sibling of the raw file: `data/processed/collegemsg/`.
    """
    cache_dir = raw_gz_path.parent.parent.parent / "processed" / "collegemsg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return CollegeMsgLoader().build(
        raw_path=raw_gz_path,
        cache_dir=cache_dir,
        num_time_steps=num_time_steps,
        beta=beta,
    )


def parse_collegemsg_file(path: Path) -> pd.DataFrame:
    """Backwards-compatible function for direct parse access."""
    return CollegeMsgLoader().parse(path)
