"""Loader for SNAP CollegeMsg temporal network."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.data.preprocess import compute_snapshot_features


def parse_collegemsg_file(path: Path) -> pd.DataFrame:
    """Parse SNAP CollegeMsg `.txt.gz` (whitespace `src dst ts`) → DataFrame."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=["src", "dst", "ts"])
    return df.astype({"src": int, "dst": int, "ts": int})


def _remap_to_dense_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remap arbitrary node IDs to 0..N-1. Returns (remapped_df, N)."""
    unique_ids = sorted(set(df.src.unique()) | set(df.dst.unique()))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    df = df.copy()
    df["src"] = df["src"].map(id_map)
    df["dst"] = df["dst"].map(id_map)
    return df, len(unique_ids)


def _snapshot_bin_edges(ts: pd.Series, num_time_steps: int) -> np.ndarray:
    """Equal-time-window bin edges from t_min to t_max inclusive."""
    t_min, t_max = float(ts.min()), float(ts.max())
    return np.linspace(t_min, t_max + 1e-6, num_time_steps + 1)


def build_dynamic_graph(
    raw_gz_path: Path,
    num_time_steps: int,
    beta: float,
) -> DynamicGraph:
    """Parse CollegeMsg gzip file and produce a fully-preprocessed DynamicGraph.

    Each snapshot's `.x` contains [degree, CC, AS] per node and `.S_hat`
    contains the enhanced adjacency from NRNAE.
    """
    df = parse_collegemsg_file(raw_gz_path)
    df, num_nodes = _remap_to_dense_ids(df)

    bins = _snapshot_bin_edges(df.ts, num_time_steps)
    snapshots: list[Data] = []
    for t in range(num_time_steps):
        mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
        sub = df.loc[mask, ["src", "dst"]].values
        edges_list = [(int(u), int(v)) for u, v in sub]

        features, S_hat = compute_snapshot_features(edges_list, num_nodes, beta)
        if len(edges_list) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data.x = features
        data.S_hat = S_hat
        snapshots.append(data)

    # node_features as fallback: one-hot identity (used by baselines, not GCN_MA)
    return DynamicGraph(
        snapshots=snapshots,
        node_features=torch.eye(num_nodes),
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        dataset_name="collegemsg",
    )
