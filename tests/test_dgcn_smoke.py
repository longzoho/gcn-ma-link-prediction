"""Smoke tests for DGCN (path B reimpl from Manessi 2020, WD-GCN variant)."""
import torch

from src.models.dgcn import DGCN, SpectralGCNLayer


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


# --------------------------------------------------------------------------
# DGCN composition
# --------------------------------------------------------------------------

def test_dgcn_construct():
    """Default construction wires up embedding, 2 GCN layers, 1-layer LSTM, decoder."""
    m = DGCN(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_gcn_layers=2,
        num_lstm_layers=1,
        dropout=0.1,
    )
    assert m is not None
    assert m.num_nodes == 50
    assert m.hidden_dim == 64
    assert m.node_emb.weight.shape == (50, 64)
    assert len(m.gcn_layers) == 2
    assert m.gcn_layers[0].in_dim == 64   # feat_dim
    assert m.gcn_layers[0].out_dim == 64  # hidden_dim
    assert m.gcn_layers[1].in_dim == 64   # hidden_dim
    assert m.gcn_layers[1].out_dim == 64  # hidden_dim
    assert isinstance(m.lstm, torch.nn.LSTM)
    assert m.lstm.input_size == 64
    assert m.lstm.hidden_size == 64
    assert m.lstm.num_layers == 1
    assert hasattr(m, "decoder")
