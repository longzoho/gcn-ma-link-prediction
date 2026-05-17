"""Smoke tests for DyGNN (path B reimpl, vectorized)."""
import torch
from torch_geometric.data import Data

from src.models.dygnn import DyGNN
from src.models.dygnn.edge_update import CoupledGRUUpdate
from src.models.dygnn.node_memory import NodeMemory


# --------------------------------------------------------------------------
# NodeMemory (path B B1)
# --------------------------------------------------------------------------

def test_node_memory_init_zero():
    nm = NodeMemory(num_nodes=50, dim=64)
    assert nm.state.shape == (50, 64)
    assert torch.allclose(nm.state, torch.zeros(50, 64))


def test_node_memory_fresh_clone():
    nm = NodeMemory(num_nodes=10, dim=8)
    cloned = nm.fresh_clone()
    assert cloned.shape == (10, 8)
    assert cloned.requires_grad
    assert torch.allclose(cloned, nm.state)


# --------------------------------------------------------------------------
# CoupledGRUUpdate (path B B2 — VECTORIZED)
# --------------------------------------------------------------------------

def test_coupled_gru_update_shape():
    """Vectorized: takes full snapshot (edge_index, edge_ts) at once."""
    N, D, E = 20, 16, 5
    mem = torch.randn(N, D, requires_grad=True)
    edge_index = torch.randint(0, N, (2, E))
    edge_ts = torch.rand(E, dtype=torch.float64)
    layer = CoupledGRUUpdate(node_dim=D, decay_method="log", decay_rate=1.0)
    new_mem = layer(mem, edge_index, edge_ts)
    assert new_mem.shape == (N, D)
    assert torch.isfinite(new_mem).all()


def test_coupled_gru_update_gradient_flows():
    N, D, E = 20, 16, 5
    mem = torch.randn(N, D, requires_grad=True)
    edge_index = torch.randint(0, N, (2, E))
    edge_ts = torch.rand(E, dtype=torch.float64)
    layer = CoupledGRUUpdate(node_dim=D, decay_method="log", decay_rate=1.0)
    new_mem = layer(mem, edge_index, edge_ts)
    new_mem.sum().backward()
    assert mem.grad is not None and torch.isfinite(mem.grad).all()
    has_param_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in layer.parameters()
    )
    assert has_param_grad


def test_coupled_gru_update_no_edges_returns_memory():
    N, D = 20, 16
    mem = torch.randn(N, D, requires_grad=True)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_ts = torch.empty(0, dtype=torch.float64)
    layer = CoupledGRUUpdate(node_dim=D, decay_method="log", decay_rate=1.0)
    new_mem = layer(mem, edge_index, edge_ts)
    torch.testing.assert_close(new_mem, mem)


# --------------------------------------------------------------------------
# DyGNN composition (path B B4)
# --------------------------------------------------------------------------

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


def test_dygnn_construct():
    m = DyGNN(
        num_nodes=50, hidden_dim=64, node_memory_dim=64, edge_dim=16,
        dropout=0.1, decay_method="log", decay_rate=1.0,
    )
    assert m is not None
    assert m.memory_init.shape == (50, 64)
    assert torch.allclose(m.memory_init.detach(), torch.zeros(50, 64))
    assert hasattr(m, "decoder")


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
    assert has_grad


def test_dygnn_cache_reuses_within_epoch_eval():
    """Cache reuse only safe under no_grad (eval). In training, every backward
    frees the shared autograd graph, so the cache rebuilds per forward."""
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    m.eval()
    snaps = _make_dummy_snapshots(N, T)
    with torch.no_grad():
        Z1 = m(snaps, time_step=3)
        Z2 = m(snaps, time_step=3)
    torch.testing.assert_close(Z1, Z2)


def test_dygnn_cache_rebuilds_on_t_regression_eval():
    N, T, D = 50, 5, 64
    m = DyGNN(num_nodes=N, hidden_dim=D, node_memory_dim=D, edge_dim=16, dropout=0.0)
    m.eval()
    snaps = _make_dummy_snapshots(N, T)
    with torch.no_grad():
        Z4_first = m(snaps, time_step=4)
        Z2 = m(snaps, time_step=2)
        Z4_second = m(snaps, time_step=4)
    torch.testing.assert_close(Z4_first, Z4_second, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(Z2, Z4_first, atol=1e-3), "Z(t=2) should differ from Z(t=4)"
