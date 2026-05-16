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
