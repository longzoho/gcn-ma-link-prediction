"""Abstract base class for dynamic link-prediction models."""
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.data import Data


class DynamicLinkPredictor(nn.Module, ABC):
    """Common interface for GCN_MA and baselines.

    Subclasses implement `forward(snapshots, t)` returning node embeddings Z^t.
    `predict_link` is shared and may be overridden.
    """

    @abstractmethod
    def forward(self, snapshots: list[Data], time_step: int) -> torch.Tensor:
        """Return Z^t ∈ R^{N×D} given snapshots [0..time_step]."""

    def predict_link(self, Z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Default decoder: dot product of source and target embeddings.

        Subclasses with a learned MLP decoder should override and call their
        internal decoder. This default is for sanity testing only.

        Args:
            Z: [N, D] embeddings.
            edges: [2, E] edge index.
        Returns:
            logits: [E] tensor.
        """
        src, dst = edges[0], edges[1]
        return (Z[src] * Z[dst]).sum(dim=-1)
