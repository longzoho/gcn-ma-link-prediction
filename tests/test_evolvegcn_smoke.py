"""Smoke tests for EvolveGCN-O upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "EvolveGCN"))


def test_can_import_upstream_egcn_o():
    """Upstream EGCN class must import without PyTorch 2.4 errors."""
    from egcn_o import EGCN  # noqa: F401
    assert EGCN is not None


import torch

from src.models.evolvegcn import EvolveGCN_O


def test_can_construct_evolvegcn_o():
    """Adapter constructor must succeed with realistic shapes."""
    m = EvolveGCN_O(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    )
    assert m is not None
    # Node embedding should be initialized
    assert m.node_emb.weight.shape == (50, 64)


from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int) -> list[Data]:
    snaps = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, N * 2))
        d = Data(edge_index=ei, num_nodes=N)
        snaps.append(d)
    return snaps


def test_evolvegcn_forward_shape():
    N, T, D = 50, 5, 64
    m = EvolveGCN_O(num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D), f"expected ({N}, {D}), got {Z.shape}"


def test_evolvegcn_gradient_flows():
    N, T, D = 50, 4, 64
    m = EvolveGCN_O(num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    # At least one parameter must have finite gradient
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no finite non-zero gradient on any parameter"
