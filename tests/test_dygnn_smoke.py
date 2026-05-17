"""Smoke tests for DyGNN upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "DyGNN"))


def test_can_import_upstream_dygnn():
    """Upstream DyGNN class must import under PyTorch 2.4."""
    from model_recurrent import DyGNN  # noqa: F401
    assert DyGNN is not None


import torch

from src.models.dygnn import DyGNN


def test_can_construct_dygnn():
    m = DyGNN(
        num_nodes=50,
        hidden_dim=64,
        edge_dim=16,
        dropout=0.1,
        decay_method="log",
        decay_rate=1.0,
        device=torch.device("cpu"),
    )
    assert m is not None
    # Adapter exposes a memory_init tensor that maps to upstream's node_representations
    assert m.memory_init.shape == (50, 64)
    # Memory should start at zero per spec §9.1
    assert torch.allclose(m.memory_init, torch.zeros(50, 64))
    # Upstream attached
    assert hasattr(m, "_upstream"), "adapter must hold the upstream DyGNN instance"
    # Decoder attached
    assert hasattr(m, "decoder")


from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int, max_edges: int = 20) -> list[Data]:
    snaps = []
    torch.manual_seed(0)
    for t in range(T):
        e = max(1, max_edges // 2)
        ei = torch.randint(0, N, (2, e))
        ts = torch.rand(e, dtype=torch.float64) + float(t)
        d = Data(edge_index=ei, num_nodes=N)
        d.edge_ts = ts
        snaps.append(d)
    return snaps


def test_dygnn_forward_shape():
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D)
    assert torch.isfinite(Z).all()


def test_dygnn_gradient_flows():
    N, T, D = 50, 4, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    Z.sum().backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no trainable parameter received gradient"


def test_dygnn_cache_reuses_within_epoch():
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z1 = m(snaps, time_step=3)
    Z2 = m(snaps, time_step=3)
    torch.testing.assert_close(Z1, Z2)


def test_dygnn_cache_rebuilds_on_t_regression():
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    snaps = _make_dummy_snapshots(N, T)
    Z4_first = m(snaps, time_step=4)
    Z2 = m(snaps, time_step=2)  # t regression → rebuild
    Z4_second = m(snaps, time_step=4)
    # Same params + deterministic rebuild → Z4 reproducible
    torch.testing.assert_close(Z4_first, Z4_second, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(Z2, Z4_first, atol=1e-3), "Z(t=2) should differ from Z(t=4)"
