import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph, temporal_split


def _make_snapshot(num_edges: int, num_nodes: int = 5) -> Data:
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def test_dynamic_graph_fields():
    snaps = [_make_snapshot(3) for _ in range(10)]
    g = DynamicGraph(
        snapshots=snaps,
        node_features=torch.zeros(5, 3),
        num_nodes=5,
        num_time_steps=10,
        dataset_name="dummy",
    )
    assert g.num_time_steps == 10
    assert len(g.snapshots) == 10
    assert g.node_features.shape == (5, 3)


def test_temporal_split_8_1_rest():
    """Spec §6.4: Val target is snapshot ⌊0.8T⌋, test targets are snapshots
    [⌊0.8T⌋+1, T-1]. Returned tuple is (train_end, val_step, test_start) where
    `val_step` is the t value (model sees [0..val_step], predicts val_step+1)
    and `test_start` is the first t for test iteration."""
    train_end, val_step, test_start = temporal_split(num_time_steps=10, train_ratio=0.8)
    # floor(0.8 * 10) = 8
    # val target = snapshot 8 → val_step (t input) = 7
    # test targets = snapshots [9, 9] (i.e. just snapshot 9) → test t = 8
    assert train_end == 8
    assert val_step == 7
    assert test_start == 8


def test_temporal_split_non_round():
    # 47 time steps (CollegeMsg)
    train_end, val_step, test_start = temporal_split(num_time_steps=47, train_ratio=0.8)
    # floor(0.8 * 47) = 37
    # val target = snapshot 37, val_step = 36, test_start = 37
    assert train_end == 37
    assert val_step == 36
    assert test_start == 37
