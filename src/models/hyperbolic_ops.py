"""Hyperbolic operations on the Poincaré ball.

Used by HTGN baseline (adapter) to project hyperbolic node embeddings
to the Euclidean tangent space at the origin, where the shared MLP
decoder consumes them.
"""
import torch


def log_map_origin(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """Poincaré ball → tangent space at origin (Euclidean approximation).

    log_0(x) = (1 / sqrt(c)) * arctanh(sqrt(c) * ||x||) * (x / ||x||)

    Args:
        x: hyperbolic embeddings, shape [..., D]. Each row's norm must be
           strictly less than 1/sqrt(c); rows on or beyond the boundary
           are clamped before atanh to avoid NaN.
        c: curvature parameter (>0). Default 1.0 (unit ball).
        eps: small constant for numerical stability.

    Returns:
        Euclidean embeddings of the same shape as `x`.
    """
    sqrt_c = c ** 0.5
    # Clamp norm to safely stay within the Poincaré ball
    max_norm = 1.0 / sqrt_c - eps
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps, max=max_norm)
    scaled_norm = sqrt_c * norm
    # Ensure atanh argument is safely bounded away from 1 to avoid infinities
    scaled_norm = scaled_norm.clamp(max=1.0 - 1e-6)
    factor = torch.atanh(scaled_norm) / scaled_norm
    return factor * x
