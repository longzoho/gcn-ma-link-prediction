"""Shared base for SNAP-style temporal loaders.

Subclasses implement `parse(path) -> DataFrame[src, dst, ts]` and set
`dataset_name` + `preprocess_version` class attributes. The base handles
ID remapping, snapshot binning, NRNAE preprocessing (with cache), and
DynamicGraph assembly.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.data.cache import cache_key_for_file, load_processed, save_processed
from src.data.preprocess import (
    aggregation_strength,
    clustering_coefficient,
    enhanced_adjacency,
)


class SNAPTemporalLoader(ABC):
    """Base class. Subclasses define `dataset_name`, `preprocess_version`,
    and a `parse(path)` method."""

    dataset_name: str = "unknown"
    preprocess_version: str = "v1"
    binning_strategy: str = "equal_time"  # or "quantile"
    # Bump this whenever the cache payload format changes (independent of
    # dataset-specific preprocess_version) to invalidate all existing caches.
    _CACHE_FORMAT_VERSION: str = "fmt3"  # fmt3 adds edge_ts per snapshot for DyGNN

    @abstractmethod
    def parse(self, path: Path) -> pd.DataFrame:
        """Return DataFrame with columns ['src', 'dst', 'ts'] (all int)."""

    def build(
        self,
        raw_path: Path,
        cache_dir: Path,
        num_time_steps: int,
        beta: float,
    ) -> DynamicGraph:
        """Load (from cache if available) and assemble a DynamicGraph."""
        # If raw file is gone, fall back to any matching cache entry for this
        # dataset + T combination (raw file was already hashed on first run).
        cached = None
        cache_path: Path | None = None
        if not raw_path.exists():
            pattern = f"{self.dataset_name}_T{num_time_steps}_*.pt"
            candidates = list(cache_dir.glob(pattern))
            if candidates:
                cache_path = candidates[0]
                cached = load_processed(cache_path)

        if cached is None:
            version = f"{self.preprocess_version}-{self._CACHE_FORMAT_VERSION}"
            cache_key = cache_key_for_file(raw_path, version=version)
            cache_path = cache_dir / f"{self.dataset_name}_T{num_time_steps}_{cache_key}.pt"
            cached = load_processed(cache_path)

        if cached is None:
            cached = self._preprocess(raw_path, num_time_steps)
            save_processed(cached, cache_path)

        # Recompute S from cached edge_index + features (AS = features[:, 2]),
        # then compose S_hat = A + β·S + I (β-dependent, not cached).
        snapshots: list[Data] = []
        N = cached["num_nodes"]
        for t in range(cached["num_time_steps"]):
            edge_index = cached["edge_index"][t]
            edge_ts = cached["edge_ts"][t]  # NEW: fmt3
            features = cached["features"][t]
            A = torch.zeros(N, N)
            if edge_index.numel() > 0:
                A[edge_index[0], edge_index[1]] = 1.0
                A[edge_index[1], edge_index[0]] = 1.0
            common = A @ A
            common.fill_diagonal_(0.0)
            as_col = features[:, 2]  # AS is column 2 of features
            S = common * as_col.unsqueeze(1)
            S_hat = enhanced_adjacency(A, S, beta=beta)

            data = Data(edge_index=edge_index, num_nodes=N)
            data.x = features
            data.S_hat = S_hat
            data.edge_ts = edge_ts  # NEW: fmt3
            snapshots.append(data)

        return DynamicGraph(
            snapshots=snapshots,
            node_features=torch.eye(N),
            num_nodes=N,
            num_time_steps=cached["num_time_steps"],
            dataset_name=self.dataset_name,
        )

    def _preprocess(self, raw_path: Path, num_time_steps: int) -> dict:
        """Heavy lifting: parse, remap, bin, compute features + S per snapshot."""
        df = self.parse(raw_path)
        df, num_nodes = _remap_to_dense_ids(df)
        if self.binning_strategy == "quantile":
            bins = _quantile_bin_edges(df.ts, num_time_steps)
        elif self.binning_strategy == "equal_time":
            bins = _snapshot_bin_edges(df.ts, num_time_steps)
        else:
            raise ValueError(
                f"Unknown binning_strategy={self.binning_strategy!r} "
                f"for {self.dataset_name}. Use 'equal_time' or 'quantile'."
            )

        features_list: list[torch.Tensor] = []
        edge_index_list: list[torch.Tensor] = []
        edge_ts_list: list[torch.Tensor] = []  # NEW: fmt3 adds edge_ts per snapshot

        for t in range(num_time_steps):
            mask = (df.ts >= bins[t]) & (df.ts < bins[t + 1])
            sub = df.loc[mask, ["src", "dst", "ts"]].values  # add "ts" column
            edges_with_ts = [(int(u), int(v), float(ts)) for u, v, ts in sub if u != v]
            edges_list = [(u, v) for u, v, ts in edges_with_ts]

            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges_list)

            cc = clustering_coefficient(G, num_nodes)
            as_ = aggregation_strength(G, cc, num_nodes)
            deg = torch.zeros(num_nodes)
            for i, d in G.degree():
                deg[i] = d
            features = torch.stack([deg, cc, as_], dim=1)
            # S is NOT cached — recomputed cheaply from edge_index + features
            # in build() via A@A. Saves 100-1000x cache size for large datasets.

            if edges_with_ts:
                ts_array = torch.tensor([ts for u, v, ts in edges_with_ts], dtype=torch.float64)
            else:
                ts_array = torch.empty(0, dtype=torch.float64)
            edge_ts_list.append(ts_array)

            if len(edges_list) == 0:
                edge_index = torch.empty(2, 0, dtype=torch.long)
            else:
                edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

            features_list.append(features)
            edge_index_list.append(edge_index)

        return {
            "features": features_list,
            "edge_index": edge_index_list,
            "edge_ts": edge_ts_list,  # NEW: fmt3
            "num_nodes": num_nodes,
            "num_time_steps": num_time_steps,
        }


def _remap_to_dense_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    unique_ids = sorted(set(df.src.unique()) | set(df.dst.unique()))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    df = df.copy()
    df["src"] = df["src"].map(id_map)
    df["dst"] = df["dst"].map(id_map)
    return df, len(unique_ids)


def _snapshot_bin_edges(ts: pd.Series, num_time_steps: int) -> np.ndarray:
    t_min, t_max = float(ts.min()), float(ts.max())
    return np.linspace(t_min, t_max + 1e-6, num_time_steps + 1)


def _quantile_bin_edges(ts: pd.Series, num_time_steps: int) -> np.ndarray:
    """Bin edges by event-count quantiles.

    Each bin contains approximately equal number of events (rows in `ts`).
    Useful for bursty datasets like EUT where equal-time bins produce
    many empty snapshots.

    Returns an array of length num_time_steps + 1 with strictly increasing
    edges. Edge cases:
        - Duplicate quantiles (when many events share a timestamp) are
          deduplicated and adjusted with small epsilons so we still get
          num_time_steps non-empty bins.
        - The final edge is bumped by 1e-6 to make the last bin inclusive
          of t_max (same convention as _snapshot_bin_edges).
    """
    quantiles = np.linspace(0, 1, num_time_steps + 1)
    bins = ts.quantile(quantiles).values.astype(float)
    # Bump duplicates monotonically so all bins are non-empty
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6
    bins[-1] = max(bins[-1], float(ts.max()) + 1e-6)
    return bins
