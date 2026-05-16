"""Multi-head self-attention with residual + LayerNorm."""
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention applied to a single snapshot's embeddings.

    Z = LayerNorm(H + MultiHead(H, H, H))
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: [N, D] → Z: [N, D]."""
        h_batched = H.unsqueeze(0)  # [1, N, D]
        attn_out, _ = self.mha(h_batched, h_batched, h_batched)
        z = self.norm(h_batched + attn_out)
        return z.squeeze(0)
