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


class DGCN(DynamicLinkPredictor):
    """DGCN baseline (Manessi 2020, WD-GCN variant).

    WD-GCN = "Waterfall" Dynamic GCN: stack of GCN layers per snapshot, then
    a single LSTM over the time dimension per node, then a shared MLP decoder.
    No NRNAE — fair-baseline policy. See spec §4.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError(f"num_gcn_layers must be >= 1, got {num_gcn_layers}")
        if num_lstm_layers < 1:
            raise ValueError(f"num_lstm_layers must be >= 1, got {num_lstm_layers}")
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Learnable node features (replaces paper's one-hot I_N — RAM constraint,
        # documented in reproduction-log.md as a Plan 2 deviation).
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # GCN stack: layer 0 takes feat_dim, subsequent layers take hidden_dim
        layers = []
        for i in range(num_gcn_layers):
            in_d = feat_dim if i == 0 else hidden_dim
            layers.append(SpectralGCNLayer(in_d, hidden_dim, dropout=dropout))
        self.gcn_layers = nn.ModuleList(layers)

        # Per-node temporal LSTM over the GCN stack output sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Shared decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step: int) -> torch.Tensor:
        """Return per-node embedding Z^t [N, hidden_dim] at end of snapshot `time_step`.

        Pipeline:
            1. For each snapshot t' ∈ [0, time_step]:
                 X = self.node_emb.weight              # [N, feat_dim], shared learnable
                 ei_sym = symmetrize(snapshots[t'].edge_index)
                 H = X
                 for layer in self.gcn_layers:
                     H = layer(H, ei_sym, N)
                 gcn_outputs.append(H)                  # [N, hidden_dim]
            2. Stack: [time_step+1, N, hidden_dim] -> permute to [N, time_step+1, hidden_dim]
            3. LSTM (batch=N, seq_len=time_step+1) -> last hidden state per node
            4. Return [N, hidden_dim]
        """
        N = self.num_nodes
        device = self.node_emb.weight.device
        gcn_outputs = []
        for t in range(time_step + 1):
            snap = snapshots[t]
            ei = snap.edge_index.to(device) if snap.edge_index.numel() > 0 else snap.edge_index
            # Symmetrize for bipartite datasets (mooc, wikipedia, lastfm).
            # Harmless on already-undirected datasets — adds duplicate edges
            # whose weights renormalize together.
            if ei.numel() > 0:
                ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                ei_sym = ei
            h = self.node_emb.weight  # [N, feat_dim]
            for layer in self.gcn_layers:
                h = layer(h, ei_sym, N)
            gcn_outputs.append(h)

        # [T, N, D] -> [N, T, D] for batch_first LSTM
        stacked = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)
        out, _ = self.lstm(stacked)  # [N, T, D]
        return out[:, -1, :]  # last time step per node

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
