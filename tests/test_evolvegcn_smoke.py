"""Smoke tests for EvolveGCN-O upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "EvolveGCN"))


def test_can_import_upstream_egcn_o():
    """Upstream EGCN class must import without PyTorch 2.4 errors."""
    from egcn_o import EGCN  # noqa: F401
    assert EGCN is not None


import torch

from src.models.evolvegcn import EvolveGCN_O


def test_can_construct_evolvegcn_o():
    """Adapter constructor must succeed with realistic shapes."""
    m = EvolveGCN_O(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    )
    assert m is not None
    # Node embedding should be initialized
    assert m.node_emb.weight.shape == (50, 64)
