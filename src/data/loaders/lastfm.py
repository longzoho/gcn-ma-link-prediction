"""Loader for JODIE LastFM dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class LastFMLoader(SNAPTemporalLoader):
    dataset_name = "lastfm"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        # JODIE lastfm.csv header declares 5 columns but data has more.
        # Read only the first 3 positional columns: user_id, item_id, timestamp.
        df = pd.read_csv(
            path, header=0, usecols=[0, 1, 2],
            names=["src", "dst", "ts"],
        )
        # Bipartite: shift item IDs to avoid overlap with user IDs.
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
