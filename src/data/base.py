"""Standard dynamic-graph container and temporal split helper."""
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class DynamicGraph:
    """A sequence of graph snapshots over time.

    Attributes:
        snapshots: list of PyG Data objects, one per time step.
        node_features: shared node feature matrix [N, F] (may be per-snapshot
            in a separate field — see GCN_MA usage; this is the canonical
            fallback used by baselines).
        num_nodes: N.
        num_time_steps: T = len(snapshots).
        dataset_name: identifier for logging.
    """

    snapshots: list[Data]
    node_features: torch.Tensor
    num_nodes: int
    num_time_steps: int
    dataset_name: str


def temporal_split(num_time_steps: int, train_ratio: float = 0.8) -> tuple[int, int, int]:
    """Compute (train_end, val_step, test_start) per spec §6.4.

    Semantics of returned indices:
        - Training iterates `t in [0, val_step)` and predicts snapshot t+1.
        - Validation uses `t = val_step` (model sees snapshots [0..val_step],
          predicts snapshot val_step+1 == train_end).
        - Test iterates `t in [test_start, num_time_steps - 1)` and predicts t+1.

    Args:
        num_time_steps: T.
        train_ratio: fraction of snapshots reserved for training+val.

    Returns:
        train_end: snapshot index at the train/test boundary (val target).
        val_step: t input for validation (= train_end - 1).
        test_start: first t for test iteration (= train_end).
    """
    train_end = int(num_time_steps * train_ratio)
    val_step = train_end - 1
    test_start = train_end
    return train_end, val_step, test_start
