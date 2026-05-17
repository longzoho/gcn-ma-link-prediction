"""Smoke tests for DyGNN upstream integration."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "DyGNN"))


def test_can_import_upstream_dygnn():
    """Upstream DyGNN class must import under PyTorch 2.4."""
    from model_recurrent import DyGNN  # noqa: F401
    assert DyGNN is not None
