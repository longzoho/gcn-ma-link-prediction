"""Disk cache for preprocessed dynamic-graph artifacts.

Cached payload schema (a dict):
    features: list[torch.Tensor]    # len T, each [N, 3]
    S:        list[torch.Tensor]    # len T, each [N, N] (β-independent)
    num_nodes: int
    num_time_steps: int
    edge_index: list[torch.Tensor]  # len T, each [2, E_t]
"""
import hashlib
from pathlib import Path
from typing import Any

import torch


def cache_key_for_file(raw_path: Path, version: str) -> str:
    """Compute a short hex key from `(raw_file_content, version)`.

    `version` should bump whenever preprocessing logic changes so old caches
    invalidate. 16-char prefix is plenty to avoid collisions in practice.
    """
    h = hashlib.sha256()
    with raw_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(version.encode())
    return h.hexdigest()[:16]


def save_processed(payload: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_processed(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu", weights_only=False)
