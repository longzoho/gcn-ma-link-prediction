"""MLP link decoder.

logit(u, v) = MLP_2-layer([Z[u] ⊕ Z[v]])
"""
import torch
from torch import nn


class LinkDecoderMLP(nn.Module):
    """2-layer MLP returning raw logits over edge pairs."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Z: [N, D], edges: [2, E] → logits: [E]."""
        src, dst = edges[0], edges[1]
        pair = torch.cat([Z[src], Z[dst]], dim=-1)
        return self.net(pair).squeeze(-1)
