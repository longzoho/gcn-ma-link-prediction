"""Loader for JODIE benchmark Mooc-actions dataset."""
from pathlib import Path

import pandas as pd

from src.data.loaders._base import SNAPTemporalLoader


class MoocActionsLoader(SNAPTemporalLoader):
    dataset_name = "mooc_actions"
    preprocess_version = "v1"

    def parse(self, path: Path) -> pd.DataFrame:
        # JODIE format: header + user_id,item_id,timestamp,state_label,4 features
        df = pd.read_csv(path)
        df = df.rename(columns={"user_id": "src", "item_id": "dst", "timestamp": "ts"})
        df = df[["src", "dst", "ts"]]
        # Bipartite: shift item IDs to avoid overlap with user IDs.
        max_user = df["src"].max()
        df["dst"] = df["dst"] + (max_user + 1)
        return df.astype({"src": int, "dst": int, "ts": int})
