import hashlib
from pathlib import Path

import torch

from src.data.cache import cache_key_for_file, load_processed, save_processed


def test_cache_key_for_file_depends_on_content(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f1.write_bytes(b"hello")
    f2 = tmp_path / "b.txt"
    f2.write_bytes(b"world")
    k1 = cache_key_for_file(f1, version="v1")
    k2 = cache_key_for_file(f2, version="v1")
    assert k1 != k2
    # Same file + same version is deterministic
    assert cache_key_for_file(f1, version="v1") == k1


def test_cache_key_depends_on_version(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    assert cache_key_for_file(f, version="v1") != cache_key_for_file(f, version="v2")


def test_save_and_load_roundtrip(tmp_path: Path):
    payload = {
        "features": [torch.zeros(5, 3), torch.ones(5, 3)],
        "S": [torch.zeros(5, 5), torch.eye(5)],
        "num_nodes": 5,
        "num_time_steps": 2,
    }
    cache_path = tmp_path / "test_cache.pt"
    save_processed(payload, cache_path)
    loaded = load_processed(cache_path)
    assert loaded["num_nodes"] == 5
    assert loaded["num_time_steps"] == 2
    torch.testing.assert_close(loaded["features"][0], torch.zeros(5, 3))
    torch.testing.assert_close(loaded["S"][1], torch.eye(5))


def test_load_processed_returns_none_if_missing(tmp_path: Path):
    assert load_processed(tmp_path / "nope.pt") is None
