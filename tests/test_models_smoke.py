import torch

from src.models.gcn_ma.gcn_layer import EnhancedGCNLayer
from src.models.gcn_ma.lstm_weight import LSTMWeightUpdater


def test_gcn_layer_output_shape():
    N, F_in, D = 10, 3, 8
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
    W = torch.randn(F_in, D, requires_grad=True)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    assert H.shape == (N, D)


def test_gcn_layer_gradient_flows():
    N, F_in, D = 10, 3, 8
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
    W = torch.randn(F_in, D, requires_grad=True)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    loss = H.sum()
    loss.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_gcn_layer_handles_isolated_nodes():
    """Zero-degree node should not produce NaN (degree normalization edge case)."""
    N, F_in, D = 5, 3, 4
    X = torch.randn(N, F_in)
    S_hat = torch.eye(N)  # only identity → all "degree 1" via self-loop
    W = torch.randn(F_in, D)
    layer = EnhancedGCNLayer()
    H = layer(X, S_hat, W)
    assert torch.isfinite(H).all()


def test_lstm_weight_updater_preserves_shape():
    F_in, D = 3, 8
    updater = LSTMWeightUpdater(in_features=F_in, out_features=D)
    W_t = torch.randn(F_in, D)
    h_t, c_t = updater.init_state(W_t.device)
    W_next, h_next, c_next = updater(W_t, h_t, c_t)
    assert W_next.shape == (F_in, D)
    assert h_next.shape == h_t.shape
    assert c_next.shape == c_t.shape


def test_lstm_weight_updater_gradients_flow():
    F_in, D = 3, 4
    updater = LSTMWeightUpdater(in_features=F_in, out_features=D)
    W_t = torch.randn(F_in, D, requires_grad=True)
    h_t, c_t = updater.init_state(W_t.device)
    W_next, _, _ = updater(W_t, h_t, c_t)
    loss = W_next.sum()
    loss.backward()
    assert W_t.grad is not None
    assert torch.isfinite(W_t.grad).all()
    # The LSTM cell should also accumulate grads on its own params
    for p in updater.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


from src.models.gcn_ma.attention import MultiHeadSelfAttention
import pytest


def test_attention_preserves_shape():
    N, D = 10, 16
    attn = MultiHeadSelfAttention(embed_dim=D, num_heads=4, dropout=0.0)
    H = torch.randn(N, D)
    Z = attn(H)
    assert Z.shape == (N, D)


def test_attention_embed_dim_divisible_by_heads_validated():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=10, num_heads=3, dropout=0.0)


from src.models.gcn_ma.link_decoder import LinkDecoderMLP


def test_link_decoder_output_shape():
    N, D, E = 10, 16, 7
    Z = torch.randn(N, D)
    edges = torch.randint(0, N, (2, E))
    decoder = LinkDecoderMLP(embed_dim=D, hidden_dim=D, dropout=0.0)
    logits = decoder(Z, edges)
    assert logits.shape == (E,)


def test_link_decoder_logit_not_probability():
    """Decoder returns raw logits, NOT sigmoid (we use BCEWithLogitsLoss)."""
    Z = torch.randn(5, 4)
    edges = torch.tensor([[0, 1], [2, 3]])
    decoder = LinkDecoderMLP(embed_dim=4, hidden_dim=4, dropout=0.0)
    logits = decoder(Z, edges)
    # If output were already sigmoid'd we'd expect all to be in [0,1].
    # A 2-layer MLP with random init can produce values outside this range.
    # Statistically over 100 calls at least one should be outside [0,1].
    found_outside = False
    for _ in range(100):
        decoder = LinkDecoderMLP(embed_dim=4, hidden_dim=4, dropout=0.0)
        out = decoder(Z, edges)
        if (out < 0).any() or (out > 1).any():
            found_outside = True
            break
    assert found_outside, "decoder appears to apply sigmoid internally"


from torch_geometric.data import Data

from src.models.gcn_ma.model import GCN_MA


def _make_dummy_snapshots(N: int, T: int, F_in: int = 3) -> list[Data]:
    snaps = []
    for _ in range(T):
        d = Data(edge_index=torch.randint(0, N, (2, N * 2)), num_nodes=N)
        d.x = torch.randn(N, F_in)
        d.S_hat = torch.eye(N) + 0.1 * torch.rand(N, N)
        snaps.append(d)
    return snaps


def test_gcn_ma_forward_shape():
    N, T, D = 8, 5, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    assert Z.shape == (N, D)


def test_gcn_ma_gradient_flows():
    N, T, D = 8, 4, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_gcn_ma_predict_link():
    N, T, D = 8, 3, 16
    model = GCN_MA(feat_dim=3, hidden_dim=D, num_heads=4, dropout=0.0)
    snapshots = _make_dummy_snapshots(N, T)
    Z = model(snapshots, time_step=T - 1)
    edges = torch.tensor([[0, 1, 2], [3, 4, 5]])
    logits = model.predict_link(Z, edges)
    assert logits.shape == (3,)
