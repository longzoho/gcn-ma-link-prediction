"""LSTM-based weight evolver for GCN_MA.

W^t = LSTMCell(flatten(W^{t-1}), state^{t-1})

Treats the weight matrix W ∈ R^{F×D} as a flattened vector of size F*D.
"""
import torch
from torch import nn


class LSTMWeightUpdater(nn.Module):
    """LSTM cell that evolves a [F, D] weight matrix across time steps."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = in_features * out_features
        self.cell = nn.LSTMCell(input_size=self.dim, hidden_size=self.dim)

    def init_state(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, self.dim, device=device)
        c = torch.zeros(1, self.dim, device=device)
        return h, c

    def forward(
        self, W: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = W.reshape(1, self.dim)
        h_next, c_next = self.cell(flat, (h, c))
        W_next = h_next.reshape(self.in_features, self.out_features)
        return W_next, h_next, c_next
