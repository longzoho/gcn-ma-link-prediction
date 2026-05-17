"""Interaction unit (identity placeholder) for DyGNN path B.

Paper Eq. 8-9 propagates updated u/v signals to their k-hop neighbors.
Our path B reimplementation skips this — the vectorized CoupledGRUUpdate
already captures the dominant signal and adding propagation requires
storing per-snapshot adjacency. Documented deviation in reproduction-log.
"""
import torch
from torch import nn


class InteractionUnit(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, memory: torch.Tensor, **kwargs) -> torch.Tensor:
        return memory
