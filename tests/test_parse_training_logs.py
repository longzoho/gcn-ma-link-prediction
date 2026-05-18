"""Smoke tests for the training-log parser."""
import json
import re
from pathlib import Path

from scripts.parse_training_logs import (
    parse_epoch_lines,
    parse_log_file,
    parse_filename,
)


# --------------------------------------------------------------------------
# Filename parser
# --------------------------------------------------------------------------

def test_parse_filename_extracts_model_dataset_seed():
    """The filename pattern is <model>_<dataset>_seed<N>_<timestamp>.log."""
    name = "dgcn_collegemsg_seed42_20260518-014258.log"
    out = parse_filename(name)
    assert out == {"model": "dgcn", "dataset": "collegemsg", "seed": 42}


def test_parse_filename_handles_multi_word_dataset():
    """mooc_actions has an underscore in the dataset name."""
    name = "htgn_mooc_actions_seed123_20260517-110000.log"
    out = parse_filename(name)
    assert out == {"model": "htgn", "dataset": "mooc_actions", "seed": 123}


# --------------------------------------------------------------------------
# Epoch line parser
# --------------------------------------------------------------------------

def test_parse_epoch_lines_extracts_summary():
    """Per-epoch summary lines look like `Epoch   N: loss=X val_auc=Y val_ap=Z`."""
    text = (
        "some preamble\n"
        "Epoch 0:  10%|##| 1/10 [00:00<00:00,  5.0it/s]\r"
        "Epoch 0: 100%|##| 10/10 [00:02<00:00,  4.5it/s]\n"
        "                                          \n"
        "Epoch   0: loss=0.6921 val_auc=0.9100 val_ap=0.9050\n"
        "Epoch   1: loss=0.5500 val_auc=0.9300 val_ap=0.9200\n"
        "Epoch   2: loss=0.4100 val_auc=0.9400 val_ap=0.9300\n"
    )
    records = parse_epoch_lines(text)
    assert len(records) == 3
    assert records[0] == {"epoch": 0, "loss": 0.6921, "val_auc": 0.9100, "val_ap": 0.9050}
    assert records[1] == {"epoch": 1, "loss": 0.5500, "val_auc": 0.9300, "val_ap": 0.9200}
    assert records[2] == {"epoch": 2, "loss": 0.4100, "val_auc": 0.9400, "val_ap": 0.9300}


def test_parse_epoch_lines_ignores_tqdm_progress():
    """tqdm intermediate progress lines (with `\\r` clobber) must not be parsed as epoch summaries."""
    text = (
        "Epoch 0:  50%|####| 5/10 [00:01<00:01,  5.0it/s]\r"
        "Epoch 0:  60%|####| 6/10 [00:01<00:01,  5.0it/s]\r"
    )
    records = parse_epoch_lines(text)
    assert records == []


def test_parse_epoch_lines_handles_empty_input():
    """Empty file yields empty list (not None, not error)."""
    assert parse_epoch_lines("") == []


# --------------------------------------------------------------------------
# Full file parser (integration)
# --------------------------------------------------------------------------

def test_parse_log_file_integration(tmp_path):
    """End-to-end: a log file becomes a list of dicts with model/dataset/seed annotated."""
    log = tmp_path / "dgcn_collegemsg_seed42_20260518-000000.log"
    log.write_text(
        "Epoch 0: 100%|##| 10/10\r\n"
        "Epoch   0: loss=0.50 val_auc=0.95 val_ap=0.94\n"
        "Epoch   1: loss=0.30 val_auc=0.97 val_ap=0.96\n"
    )
    records = parse_log_file(log)
    assert len(records) == 2
    assert records[0]["model"] == "dgcn"
    assert records[0]["dataset"] == "collegemsg"
    assert records[0]["seed"] == 42
    assert records[0]["epoch"] == 0
    assert records[0]["loss"] == 0.50
    assert records[0]["val_auc"] == 0.95
    assert records[0]["val_ap"] == 0.94
