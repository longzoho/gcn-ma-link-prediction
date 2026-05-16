"""Loader for SNAP Email-EU-core-temporal."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class EUTLoader(SNAPTemporalLoader):
    dataset_name = "eut"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
        return df.astype({"src": int, "dst": int, "ts": int})
