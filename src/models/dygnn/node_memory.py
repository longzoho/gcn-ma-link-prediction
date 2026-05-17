"""Per-node memory matrix for DyGNN (path B)."""
import torch
from torch import nn


class NodeMemory(nn.Module):
    """Container for per-node memory state, zero-initialized (spec §9.1)."""

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.state = nn.Parameter(torch.zeros(num_nodes, dim))

    def fresh_clone(self) -> torch.Tensor:
        return self.state.clone()
