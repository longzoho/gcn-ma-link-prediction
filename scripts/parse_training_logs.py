"""Parse training log files produced by `scripts/train.py` into a JSONL of per-epoch records.

Each line in the output JSONL is one epoch from one run:
    {"model": "dgcn", "dataset": "collegemsg", "seed": 42,
     "epoch": 0, "loss": 0.50, "val_auc": 0.95, "val_ap": 0.94}

Log filename pattern: <model>_<dataset>_seed<N>_<timestamp>.log
Log content: tqdm progress bars (separated by `\\r`) mixed with epoch-summary
lines like `Epoch   N: loss=X val_auc=Y val_ap=Z`.
"""
import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known dataset name tokens (must be checked before splitting on `_seed`)
# `mooc_actions` is the only multi-word dataset in this project; others are single tokens.
_DATASETS = ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]
_MODELS = ["gcn_ma", "evolvegcn_o", "htgn", "dygnn", "dgcn"]

# Regex for an epoch summary line. Anchored with start-of-line and matches the exact format
# emitted by `Trainer.train_dynamic`. Tolerant of variable whitespace around `Epoch`.
_EPOCH_RE = re.compile(
    r"^\s*Epoch\s+(\d+):\s+loss=([\d.]+)\s+val_auc=([\d.]+)\s+val_ap=([\d.]+)\s*$"
)


def parse_filename(name: str) -> dict:
    """Extract model, dataset, seed from a log filename.

    Filename convention (from scripts/run_seeds.sh):
        <model>_<dataset>_seed<seed>_<timestamp>.log
    where <model> ∈ _MODELS and <dataset> ∈ _DATASETS.
    """
    stem = Path(name).stem  # drop .log
    # Find which model prefix this is
    matched_model = None
    for m in _MODELS:
        if stem.startswith(m + "_"):
            matched_model = m
            break
    if matched_model is None:
        raise ValueError(f"No known model prefix in filename {name!r}")
    rest = stem[len(matched_model) + 1:]  # strip "<model>_"
    # Find dataset (longest-prefix match — mooc_actions before any single-word match)
    matched_dataset = None
    for d in sorted(_DATASETS, key=len, reverse=True):
        if rest.startswith(d + "_seed"):
            matched_dataset = d
            break
    if matched_dataset is None:
        raise ValueError(f"No known dataset in filename {name!r}")
    after_ds = rest[len(matched_dataset) + len("_seed"):]  # strip "<dataset>_seed"
    # Parse seed up to the next "_"
    seed_str = after_ds.split("_", 1)[0]
    seed = int(seed_str)
    return {"model": matched_model, "dataset": matched_dataset, "seed": seed}


def parse_epoch_lines(text: str) -> list[dict]:
    """Extract epoch-summary records from raw log text.

    Splits on BOTH `\\n` and `\\r` to handle tqdm progress lines that overwrite
    with carriage returns. Only lines matching the strict epoch-summary format
    are kept; tqdm progress (`Epoch N:  XX%|...`) is filtered out by the regex.
    """
    records: list[dict] = []
    # Normalize both \r and \n to \n, then split
    normalized = text.replace("\r", "\n")
    for line in normalized.split("\n"):
        m = _EPOCH_RE.match(line)
        if m:
            records.append({
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "val_auc": float(m.group(3)),
                "val_ap": float(m.group(4)),
            })
    return records


def parse_log_file(path: Path) -> list[dict]:
    """Parse one log file. Returns a list of per-epoch records with model/dataset/seed annotated."""
    meta = parse_filename(path.name)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    epoch_records = parse_epoch_lines(text)
    return [{**meta, **r} for r in epoch_records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "results" / "logs",
        help="Directory containing <model>_<dataset>_seed<N>_<timestamp>.log files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "report" / "training_curves.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    log_files = sorted(args.logs_dir.glob("*.log"))
    skipped: list[str] = []
    written = 0
    with args.out.open("w") as f:
        for log in log_files:
            try:
                records = parse_log_file(log)
            except ValueError as e:
                skipped.append(f"{log.name}: {e}")
                continue
            for r in records:
                f.write(json.dumps(r) + "\n")
                written += 1
    print(f"Wrote {written} epoch records from {len(log_files) - len(skipped)} logs to {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for s in skipped[:10]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
