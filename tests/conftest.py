"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_all():
    """Make every test deterministic."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def tiny_graph_edges():
    """5-node graph with known clustering structure.

    Edges: (0,1), (0,2), (1,2), (2,3), (3,4)
    Triangles: {0,1,2}
    """
    return [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
