"""GCN_MA: composition of NRNAE-enhanced GCN, LSTM weight evolver,
multi-head self-attention, and an MLP link decoder.

Per paper §3:
    H^t = GCNLayer(X^t, Ŝ^t, W^t)
    W^t = LSTMCell(W^{t-1})
    Z^t = MultiHeadSelfAttn(H^t)
    P^t = σ(MLP([Z^t[u] ⊕ Z^t[v]]))
"""
import math

import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.attention import MultiHeadSelfAttention
from src.models.gcn_ma.gcn_layer import EnhancedGCNLayer
from src.models.gcn_ma.link_decoder import LinkDecoderMLP
from src.models.gcn_ma.lstm_weight import LSTMWeightUpdater


class GCN_MA(DynamicLinkPredictor):
    def __init__(
        self,
        feat_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.gcn = EnhancedGCNLayer()
        self.lstm_w = LSTMWeightUpdater(in_features=feat_dim, out_features=hidden_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Initial W^0 via Xavier
        bound = 1.0 / math.sqrt(feat_dim)
        self.W0 = nn.Parameter(torch.empty(feat_dim, hidden_dim).uniform_(-bound, bound))

    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        """Run the model up to and including snapshot `time_step`.

        Returns Z^{time_step} ∈ R^{N×D}.
        """
        device = self.W0.device
        W = self.W0
        h, c = self.lstm_w.init_state(device)

        H = None
        for tau in range(time_step + 1):
            snap = snapshots[tau]
            X = snap.x.to(device)
            S_hat = snap.S_hat.to(device)
            H = self.gcn(X, S_hat, W)
            if tau < time_step:
                W, h, c = self.lstm_w(W, h, c)

        # apply attention only on the final snapshot's embeddings
        assert H is not None
        return self.attn(H)

    def predict_link(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z, edges)
