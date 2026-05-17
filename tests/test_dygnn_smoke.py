"""Smoke tests for DyGNN upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "DyGNN"))


def test_can_import_upstream_dygnn():
    """Upstream DyGNN class must import under PyTorch 2.4."""
    from model_recurrent import DyGNN  # noqa: F401
    assert DyGNN is not None


import torch

from src.models.dygnn import DyGNN


def test_can_construct_dygnn():
    m = DyGNN(
        num_nodes=50,
        hidden_dim=64,
        edge_dim=16,
        dropout=0.1,
        decay_method="log",
        decay_rate=1.0,
        device=torch.device("cpu"),
    )
    assert m is not None
    # Adapter exposes a memory_init tensor that maps to upstream's node_representations
    assert m.memory_init.shape == (50, 64)
    # Memory should start at zero per spec §9.1
    assert torch.allclose(m.memory_init, torch.zeros(50, 64))
    # Upstream attached
    assert hasattr(m, "_upstream"), "adapter must hold the upstream DyGNN instance"
    # Decoder attached
    assert hasattr(m, "decoder")
