"""Smoke tests for EvolveGCN-O upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "EvolveGCN"))


def test_can_import_upstream_egcn_o():
    """Upstream EGCN class must import without PyTorch 2.4 errors."""
    from egcn_o import EGCN  # noqa: F401
    assert EGCN is not None
