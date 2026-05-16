"""Loader for JODIE benchmark Mooc-actions dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class MoocActionsLoader(SNAPTemporalLoader):
    dataset_name = "mooc_actions"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        # JODIE mooc.csv has a misaligned header (5 column names but 8 data columns).
        # Skip the header row entirely and read only the first 3 positional columns
        # which are always user_id, item_id, timestamp.
        df = pd.read_csv(
            path, header=0, usecols=[0, 1, 2],
            names=["src", "dst", "ts"],
        )
        # Bipartite: shift item IDs to avoid overlap with user IDs after remap.
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
