"""Smoke tests for HTGN upstream integration.

Module:  script.models.HTGN  (third_party/HTGN as sys.path root)
Class:   HTGN
__init__ signature: (self, args)
forward  signature: (self, edge_index, x=None, weight=None)

PyTorch compat: 0 shims required
geoopt: not required by this module
Additional deps: torch_geometric, torch_scatter (already in venv)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HTGN"))


def test_can_import_upstream_htgn():
    """Upstream HTGN class must import under PyTorch 2.4.

    Shim (1 line): script/config.py calls parser.parse_args() at module level,
    which grabs pytest's argv and raises SystemExit(2).  We reset sys.argv to
    an empty program name before the import so argparse sees no arguments.
    """
    # Shim 1: neutralise upstream's module-level argparse.parse_args() call.
    _saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        from script.models.HTGN import HTGN  # noqa: F401
    finally:
        sys.argv = _saved_argv
    assert HTGN is not None


import torch

from src.models.htgn import HTGN


def test_can_construct_htgn():
    """Adapter constructor succeeds with realistic shapes."""
    m = HTGN(
        num_nodes=50,
        feat_dim=64,
        hidden_dim=64,
        num_layers=2,
        curvature=1.0,
        trainable_curvature=False,
        dropout=0.1,
    )
    assert m is not None
    assert m.node_emb.weight.shape == (50, 64)


from torch_geometric.data import Data


def _make_dummy_snapshots(N: int, T: int) -> list[Data]:
    snaps = []
    for _ in range(T):
        ei = torch.randint(0, N, (2, N * 2))
        d = Data(edge_index=ei, num_nodes=N)
        snaps.append(d)
    return snaps


def test_htgn_forward_shape():
    N, T, D = 50, 5, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    assert Z.shape == (N, D), f"expected ({N}, {D}), got {Z.shape}"
    assert torch.isfinite(Z).all(), "HTGN forward produced NaN/Inf"


def test_htgn_gradient_flows():
    N, T, D = 50, 4, 64
    m = HTGN(
        num_nodes=N, feat_dim=D, hidden_dim=D, num_layers=2,
        curvature=1.0, trainable_curvature=False, dropout=0.0,
    )
    snaps = _make_dummy_snapshots(N, T)
    Z = m(snaps, time_step=T - 1)
    loss = Z.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no finite non-zero gradient on any parameter"
