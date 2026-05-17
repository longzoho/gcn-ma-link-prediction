"""Adapter for alge24/DyGNN baseline (path A: vendored upstream).

Wraps upstream `DyGNN` class from third_party/DyGNN/model_recurrent.py.
Differences from paper:
    - Per-epoch memory cache + gradient approximation (Task 5).
    - Symmetric edge processing (Task 5).
    - Shared MLP decoder (not paper's scoring head).
    - LastFM skipped due to compute budget.
"""
import sys
from pathlib import Path

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM_DIR = _REPO_ROOT / "third_party" / "DyGNN"
if str(_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_DIR))

from model_recurrent import DyGNN as _UpstreamDyGNN  # noqa: E402

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class DyGNN(DynamicLinkPredictor):
    """DyGNN encoder wrapped as a DynamicLinkPredictor.

    The upstream class IS the entire DyGNN model (edge_updater, node_updater,
    combiner, decayer, attention, propagation, and node_representations memory).
    We wrap it as our snapshot-based adapter pattern. forward() driving the
    per-epoch memory cache is implemented in Task 5.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        edge_dim: int = 16,
        dropout: float = 0.1,
        decay_method: str = "log",
        decay_rate: float = 1.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.device_ = torch.device(device) if not isinstance(device, torch.device) else device

        self._upstream = _UpstreamDyGNN(
            num_embeddings=num_nodes,
            embedding_dims=hidden_dim,
            edge_output_size=edge_dim,
            device=self.device_,
            w=decay_rate,
            decay_method=decay_method,
            drop_p=dropout,
            if_propagation=1,
            if_no_time=0,
            if_updated=0,
            second_order=False,
            is_att=False,
        )

        # Spec §9.1: zero-init the memory. Upstream's nn.Embedding defaults to N(0,1).
        # Overwrite both the live buffer and the _copy snapshot (used by reset_reps).
        with torch.no_grad():
            self._upstream.node_representations.weight.zero_()
            self._upstream.node_representations_copy.weight.zero_()
            self._upstream.cell_head.weight.zero_()
            self._upstream.cell_head_copy.weight.zero_()
            self._upstream.cell_tail.weight.zero_()
            self._upstream.cell_tail_copy.weight.zero_()
            self._upstream.hidden_head.weight.zero_()
            self._upstream.hidden_head_copy.weight.zero_()
            self._upstream.hidden_tail.weight.zero_()
            self._upstream.hidden_tail_copy.weight.zero_()

        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Per-epoch cache state (populated in Task 5)
        self._cached_memory_per_t: list[torch.Tensor] | None = None
        self._prev_max_t: int = -1

    @property
    def memory_init(self) -> torch.Tensor:
        """Read-only view of the upstream node_representations zero-init buffer.

        Exposed for tests and inspection. Live state lives in
        `self._upstream.node_representations`.
        """
        return self._upstream.node_representations_copy.weight.detach()

    def forward(self, snapshots, time_step):
        """Per-epoch cache forward — implemented in Task 5."""
        raise NotImplementedError("DyGNN.forward — implemented in Task 5")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
