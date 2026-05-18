"""Smoke tests for DGCN (path B reimpl from Manessi 2020, WD-GCN variant)."""
import torch

from src.models.dgcn import SpectralGCNLayer


# --------------------------------------------------------------------------
# SpectralGCNLayer
# --------------------------------------------------------------------------

def test_spectral_gcn_layer_shape():
    """Single forward maps [N, in_dim] -> [N, out_dim] with finite output."""
    N, D_in, D_out = 20, 16, 32
    torch.manual_seed(0)
    x = torch.randn(N, D_in)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    layer = SpectralGCNLayer(D_in, D_out, dropout=0.0)
    out = layer(x, ei, N)
    assert out.shape == (N, D_out)
    assert torch.isfinite(out).all()


def test_spectral_gcn_layer_self_loops_keep_isolated_node_active():
    """An isolated node still receives a self-loop signal (degree >= 1)."""
    N, D = 5, 8
    x = torch.randn(N, D, requires_grad=True)
    # Node 4 isolated (no edges)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    layer = SpectralGCNLayer(D, D, dropout=0.0)
    out = layer(x, ei, N)
    assert torch.isfinite(out[4]).all()
    # Output for isolated node should be a function of its own input (self-loop)
    assert not torch.allclose(out[4], torch.zeros(D))


def test_spectral_gcn_layer_empty_edge_index():
    """When edge_index has zero edges, self-loops alone keep the layer well-defined."""
    N, D = 5, 8
    x = torch.randn(N, D)
    ei = torch.empty(2, 0, dtype=torch.long)
    layer = SpectralGCNLayer(D, D, dropout=0.0)
    out = layer(x, ei, N)
    assert out.shape == (N, D)
    assert torch.isfinite(out).all()
