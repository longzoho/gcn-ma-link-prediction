"""DyGNN baseline (path B reimpl, vectorized).

Plan 3c path B: paper-faithful reimplementation of DyGNN's core
(NodeMemory + CoupledGRUUpdate + Interaction) with a vectorized batched
edge-message aggregation instead of the upstream per-edge Python loop.

See `docs/superpowers/specs/2026-05-17-dygnn-integration-design.md` and
`docs/reproduction-log.md` for the rationale (per-edge loop in upstream
made full training intractable: ~10ms/edge × ~120k sym edges/epoch).
"""
from src.models.dygnn.model import DyGNN

__all__ = ["DyGNN"]
