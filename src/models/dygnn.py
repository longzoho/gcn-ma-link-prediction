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
        node_memory_dim: int | None = None,  # alias for hidden_dim (plan conformance)
        edge_dim: int = 16,
        dropout: float = 0.1,
        decay_method: str = "log",
        decay_rate: float = 1.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if node_memory_dim is not None and node_memory_dim != hidden_dim:
            raise ValueError(
                f"node_memory_dim ({node_memory_dim}) must equal hidden_dim ({hidden_dim})"
            )
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

    def forward(self, snapshots, time_step: int) -> torch.Tensor:
        """Return per-node embedding `Z [N, D]` at end of snapshot `time_step`.

        Per-epoch cache: rebuild when `time_step < self._prev_max_t` (new epoch
        or t regression), else return cached state. Gradient flows through the
        current snapshot's edge outputs scattered onto a detached base memory
        (per-epoch gradient approximation, spec §5).
        """
        if self._cached_memory_per_t is None or time_step < self._prev_max_t:
            self._build_cache(snapshots)
        self._prev_max_t = max(self._prev_max_t, time_step)
        return self._cached_memory_per_t[time_step]

    def _reset_upstream_memory(self) -> None:
        """Restore upstream embedding buffers + recent_timestamp + interaction_timestamp.

        Upstream `reset_reps()` only resets the 5 embedding buffers. We additionally
        zero `recent_timestamp` (controls per-node decay) and rebuild the
        `interaction_timestamp` sparse matrix so propagation neighbors are reset.
        """
        from scipy.sparse import lil_matrix
        import numpy as np

        self._upstream.reset_reps()
        with torch.no_grad():
            self._upstream.recent_timestamp.zero_()
        self._upstream.interaction_timestamp = lil_matrix(
            (self._upstream.num_embeddings, self._upstream.num_embeddings),
            dtype=np.float32,
        )

    def _build_cache(self, snapshots) -> None:
        """Process all snapshots chronologically, building per-t per-node Z cache.

        For each snapshot t:
          1. Z_base = upstream.node_representations.weight.detach().clone() (gradient cut)
          2. Build interactions tensor (sorted by edge_ts, symmetric u<->v with epsilon offset)
          3. Call self._upstream(interactions) -> (out_head [E', D], out_tail [E', D], _, _)
          4. Build Z[t]: clone Z_base, scatter out_head/out_tail via index_copy
          5. Upstream mutates internal buffers under no_grad — persistent memory for t+1

        Reset upstream memory before processing snapshot 0 (new-epoch boundary).
        Upstream forward uses numpy.random.choice for negative sampling; wrap each call in
        torch.random.fork_rng with a fixed seed for deterministic cache rebuilds.
        """
        device = next(self._upstream.parameters()).device
        T = len(snapshots)

        self._reset_upstream_memory()
        cache: list[torch.Tensor] = []

        for t in range(T):
            snap = snapshots[t]
            ei = snap.edge_index.to(device) if snap.edge_index.numel() > 0 else None

            # Snapshot base memory (detached — gradient approximation, spec §5)
            Z_base = self._upstream.node_representations.weight.detach().clone()

            if ei is None or ei.numel() == 0 or not hasattr(snap, "edge_ts") or snap.edge_ts.numel() == 0:
                cache.append(Z_base)
                continue

            ts = snap.edge_ts.to(device).to(torch.float64)
            sorted_idx = torch.argsort(ts)
            ei_sorted = ei[:, sorted_idx]
            ts_sorted = ts[sorted_idx]
            heads = ei_sorted[0].long()
            tails = ei_sorted[1].long()

            # Symmetric edge processing: each (u,v,t) followed by (v,u,t+eps)
            eps = 1e-6
            heads_sym = torch.cat([heads, tails], dim=0)
            tails_sym = torch.cat([tails, heads], dim=0)
            ts_sym = torch.cat([ts_sorted, ts_sorted + eps], dim=0)

            # Re-sort symmetric stream so timestamps are monotone non-decreasing
            sym_sort = torch.argsort(ts_sym)
            heads_sym = heads_sym[sym_sort]
            tails_sym = tails_sym[sym_sort]
            ts_sym = ts_sym[sym_sort]

            interactions = torch.stack(
                [heads_sym.float(), tails_sym.float(), ts_sym.float()], dim=1
            )  # [E_sym, 3]

            # Upstream forward uses numpy.random.choice for negative sampling;
            # fix seed before each call to ensure deterministic cache rebuilds.
            with torch.random.fork_rng():
                torch.manual_seed(0)
                out_head, out_tail, _, _ = self._upstream(interactions)

            # Scatter to per-node Z (last-write-wins = latest update per node)
            Z = Z_base.clone()
            Z = Z.index_copy(0, heads_sym, out_head)
            Z = Z.index_copy(0, tails_sym, out_tail)
            cache.append(Z)

        self._cached_memory_per_t = cache
        self._prev_max_t = -1

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
