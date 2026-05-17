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
