"""Append-only JSONL metrics writer."""
import json
from pathlib import Path
from typing import Any


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
