"""Download dataset raw files from SNAP / other sources with retry.

Usage:
    python scripts/download_datasets.py --dataset collegemsg
"""
import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_config(name: str) -> dict:
    path = REPO_ROOT / "configs" / "datasets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def download_with_retry(url: str, dest: Path, max_attempts: int = 3) -> None:
    """Stream-download URL → dest with up to max_attempts retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[{attempt}/{max_attempts}] Downloading {url} → {dest}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            print(f"OK ({dest.stat().st_size} bytes)")
            return
        except Exception as e:
            print(f"  failed: {e}")
            if attempt < max_attempts:
                time.sleep(2**attempt)
    print(f"ERROR: failed after {max_attempts} attempts", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset name (e.g. collegemsg)")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)
    raw_dir = REPO_ROOT / "data" / "raw" / args.dataset
    dest = raw_dir / cfg["raw_filename"]
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    download_with_retry(cfg["raw_url"], dest)


if __name__ == "__main__":
    main()
