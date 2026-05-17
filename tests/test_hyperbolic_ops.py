import torch

from src.models.hyperbolic_ops import log_map_origin


def test_log_map_origin_at_zero_returns_zero():
    """log_map at the ball's origin should be zero."""
    x = torch.zeros(5, 4)
    out = log_map_origin(x, c=1.0)
    torch.testing.assert_close(out, torch.zeros(5, 4))


def test_log_map_origin_finite_for_typical_norm():
    """log_map output should be finite (no NaN/Inf) for typical embeddings."""
    torch.manual_seed(0)
    x = torch.randn(10, 8) * 0.1  # small enough to be inside ball at c=1
    out = log_map_origin(x, c=1.0)
    assert torch.isfinite(out).all(), "log_map output has NaN/Inf"


def test_log_map_origin_clamps_at_boundary():
    """log_map should not produce Inf when ||x|| is at or beyond 1/sqrt(c)."""
    x_at_boundary = torch.eye(4)  # unit vectors, each with ||x||=1
    out = log_map_origin(x_at_boundary, c=1.0)
    assert torch.isfinite(out).all(), "log_map crashed at boundary"

    x_beyond = torch.eye(4) * 1.5
    out = log_map_origin(x_beyond, c=1.0)
    assert torch.isfinite(out).all(), "log_map crashed beyond boundary"


def test_log_map_origin_preserves_direction():
    """log_map_0 should preserve angular direction (just scale magnitude)."""
    x = torch.tensor([[3.0, 4.0]]) * 0.01  # small enough to be in ball
    out = log_map_origin(x, c=1.0)
    x_normed = x / x.norm()
    out_normed = out / out.norm()
    torch.testing.assert_close(x_normed, out_normed, atol=1e-5, rtol=1e-5)
