"""DGCN baseline (Manessi 2020, WD-GCN variant) — from-scratch reimplementation.

Architecture: snapshot-based GCN stack + LSTM over time per node.
Per spec §4-5. No upstream repo exists; this is the project's own implementation.
"""
import torch
from torch import nn
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class SpectralGCNLayer(nn.Module):
    """One D^(-1/2) Â D^(-1/2) X W step with self-loops.

    Builds a sparse normalized adjacency on-the-fly from edge_index — no dense
    N×N matrix is ever materialized (fits Wikipedia/LastFM in 12GB easily).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        device = x.device
        # Add self-loops: each node i gets edge (i, i)
        self_loops = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)  # [2, N]
        if edge_index.numel() > 0:
            ei = torch.cat([edge_index.to(device), self_loops], dim=1)  # [2, E+N]
        else:
            ei = self_loops

        # Degree (symmetric): each edge contributes 1 to both endpoints' degrees
        ones = torch.ones(ei.shape[1], device=device)
        deg = torch.zeros(N, device=device).index_add(0, ei[0], ones)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

        # Edge weights = D^(-1/2)[src] * D^(-1/2)[dst]
        edge_weight = deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]  # [E+N]

        # Sparse adjacency Â = sum_e edge_weight[e] · e_{src,dst}
        A_hat = torch.sparse_coo_tensor(ei, edge_weight, (N, N)).coalesce()

        # Â · X · W
        agg = torch.sparse.mm(A_hat, x)  # [N, in_dim]
        out = self.linear(agg)            # [N, out_dim]
        out = torch.relu(out)
        out = self.dropout(out)
        return out


# DGCN composition (Task 2)
