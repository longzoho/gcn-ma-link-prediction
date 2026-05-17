"""Vectorized coupled GRU edge update for DyGNN (path B).

Paper Eq. 4-7 specifies per-edge sequential updates with separate
source-side and target-side GRU cells. The upstream alge24/DyGNN implements
this as a Python loop over edges (~10ms/edge → intractable on our datasets).

This module batches the within-snapshot update:
  1. For each edge (u, v, t), build messages u→v and v→u weighted by w(Δt).
  2. Aggregate messages per destination node via scatter (sum, then count-normalized).
  3. Apply a single GRUCell to all nodes in parallel using the aggregated message
     as input and current memory as hidden state.

Trade-off: strict per-edge chronological order within a snapshot is lost;
cross-snapshot temporal order is preserved (snapshots applied sequentially).
This is the same approximation TGN and related models make. Documented in
reproduction-log.
"""
import torch
from torch import nn


def _time_decay(dt: torch.Tensor, method: str, rate: float) -> torch.Tensor:
    dt_safe = dt.clamp(min=0.0)
    if method == "log":
        return 1.0 / torch.log(dt_safe + torch.e)
    if method == "polynomial":
        return 1.0 / ((dt_safe + 1.0) ** rate)
    if method == "exponential":
        return torch.exp(-rate * dt_safe)
    raise ValueError(f"Unknown decay_method: {method}")


class CoupledGRUUpdate(nn.Module):
    """Vectorized batched memory update for one snapshot's worth of edges."""

    def __init__(self, node_dim: int, decay_method: str = "log", decay_rate: float = 1.0):
        super().__init__()
        self.node_dim = node_dim
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        self.gru_source = nn.GRUCell(input_size=node_dim, hidden_size=node_dim)
        self.gru_target = nn.GRUCell(input_size=node_dim, hidden_size=node_dim)

    def forward(
        self,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        edge_ts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one batched update for all edges in a snapshot.

        Args:
            memory: [N, D] current node state.
            edge_index: [2, E] (src, dst) pairs.
            edge_ts: [E] timestamps (float).

        Returns:
            new_memory: [N, D] updated state.
        """
        N, D = memory.shape
        E = edge_index.shape[1]
        if E == 0:
            return memory

        ts = edge_ts.to(memory.device).to(torch.float64)
        ts_norm = (ts - ts.min()).to(memory.dtype)
        decay = _time_decay(ts_norm, self.decay_method, self.decay_rate).unsqueeze(1)

        src = edge_index[0].to(memory.device).long()
        dst = edge_index[1].to(memory.device).long()

        msg_to_src = memory[dst] * decay
        msg_to_dst = memory[src] * decay

        agg_src = memory.new_zeros(N, D).index_add(0, src, msg_to_src)
        agg_dst = memory.new_zeros(N, D).index_add(0, dst, msg_to_dst)

        ones = memory.new_ones(E)
        cnt_src_raw = memory.new_zeros(N).index_add(0, src, ones)
        cnt_dst_raw = memory.new_zeros(N).index_add(0, dst, ones)

        active_src = (cnt_src_raw > 0).unsqueeze(1)
        active_dst = (cnt_dst_raw > 0).unsqueeze(1)

        agg_src = agg_src / cnt_src_raw.clamp(min=1.0).unsqueeze(1)
        agg_dst = agg_dst / cnt_dst_raw.clamp(min=1.0).unsqueeze(1)

        new_from_src = self.gru_source(agg_src, memory)
        new_from_tgt = self.gru_target(agg_dst, memory)

        new_memory = torch.where(active_src, new_from_src, memory)
        new_memory = torch.where(active_dst, new_from_tgt, new_memory)
        return new_memory
