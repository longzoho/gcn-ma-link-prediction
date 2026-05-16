"""Loader for SNAP Bitcoin OTC trust network."""
import gzip
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class BitcoinotcLoader(SNAPTemporalLoader):
    dataset_name = "bitcoinotc"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, header=None, names=["src", "dst", "rating", "ts"])
        df = df[["src", "dst", "ts"]]
        return df.astype({"src": int, "dst": int, "ts": int})
