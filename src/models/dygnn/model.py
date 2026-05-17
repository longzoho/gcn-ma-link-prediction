"""DyGNN composition (path B, vectorized).

NodeMemory + CoupledGRUUpdate (vectorized over snapshot edges) +
InteractionUnit (identity) + shared MLP decoder. Per-epoch cache as in
spec §5: rebuild on new epoch or time-step regression; gradient flows
through the chain of snapshot updates back to the initial memory state.
"""
import torch
from torch_geometric.data import Data

from src.models.base import DynamicLinkPredictor
from src.models.dygnn.edge_update import CoupledGRUUpdate
from src.models.dygnn.interaction import InteractionUnit
from src.models.dygnn.node_memory import NodeMemory
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class DyGNN(DynamicLinkPredictor):
    """DyGNN reimplemented from paper Eq. 4-7 (path B, vectorized)."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        node_memory_dim: int | None = None,
        edge_dim: int = 16,
        dropout: float = 0.1,
        decay_method: str = "log",
        decay_rate: float = 1.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if node_memory_dim is None:
            node_memory_dim = hidden_dim
        if node_memory_dim != hidden_dim:
            raise ValueError(
                f"node_memory_dim ({node_memory_dim}) must equal hidden_dim ({hidden_dim})"
            )
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.memory = NodeMemory(num_nodes=num_nodes, dim=node_memory_dim)
        self.update_module = CoupledGRUUpdate(
            node_dim=node_memory_dim,
            decay_method=decay_method,
            decay_rate=decay_rate,
        )
        self.interaction = InteractionUnit(node_dim=node_memory_dim)
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        self._cached_memory_per_t: list[torch.Tensor] | None = None
        self._prev_max_t: int = -1

    @property
    def memory_init(self) -> torch.Tensor:
        return self.memory.state

    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        # In training, every backward() frees the autograd graph that the cache
        # references. Reusing cached entries across training steps yields an
        # inplace-modification error. Cache reuse is safe only under no-grad
        # (eval / inference). See path B notes in reproduction-log.
        needs_rebuild = (
            self._cached_memory_per_t is None
            or time_step < self._prev_max_t
            or torch.is_grad_enabled()
        )
        if needs_rebuild:
            self._build_cache(snapshots)
        self._prev_max_t = max(self._prev_max_t, time_step)
        return self._cached_memory_per_t[time_step]

    def _build_cache(self, snapshots: list[Data]) -> None:
        device = self.memory.state.device
        mem = self.memory.fresh_clone()
        cache: list[torch.Tensor] = []

        for snap in snapshots:
            ei = snap.edge_index.to(device) if snap.edge_index.numel() > 0 else None
            if ei is None or ei.numel() == 0 or not hasattr(snap, "edge_ts") or snap.edge_ts.numel() == 0:
                cache.append(mem)
                continue
            ts = snap.edge_ts.to(device)
            mem = self.update_module(mem, ei, ts)
            mem = self.interaction(mem)
            cache.append(mem)

        self._cached_memory_per_t = cache
        self._prev_max_t = -1

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
