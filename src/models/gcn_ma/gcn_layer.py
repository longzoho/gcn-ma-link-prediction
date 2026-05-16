"""Spectral GCN layer with NRNAE-enhanced adjacency.

H^t = ReLU(D̂^(-1/2) · Ŝ^t · D̂^(-1/2) · X^t · W^t)

Where Ŝ^t comes from preprocessing (Ŝ = A + β·S + I) and D̂ is the row-sum
degree matrix of Ŝ.
"""
import torch
from torch import nn


class EnhancedGCNLayer(nn.Module):
    """Stateless spectral GCN layer. Weights W^t are provided externally
    (the LSTM weight updater owns them).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, X: torch.Tensor, S_hat: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        """X: [N, F], S_hat: [N, N], W: [F, D] → H: [N, D]."""
        deg = S_hat.sum(dim=1).clamp(min=self.eps)
        d_inv_sqrt = deg.pow(-0.5)
        # Symmetric normalization
        S_norm = d_inv_sqrt.unsqueeze(1) * S_hat * d_inv_sqrt.unsqueeze(0)
        H = S_norm @ X @ W
        return torch.relu(H)
