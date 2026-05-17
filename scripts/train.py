"""CLI: train one model on one dataset with one seed.

Usage:
    python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.data.loaders._base import SNAPTemporalLoader
from src.data.loaders.bitcoinotc import BitcoinotcLoader
from src.data.loaders.collegemsg import CollegeMsgLoader
from src.data.loaders.eut import EUTLoader
from src.data.loaders.lastfm import LastFMLoader
from src.data.loaders.mooc_actions import MoocActionsLoader
from src.data.loaders.wikipedia import WikipediaLoader
from src.eval.evaluator import evaluate_dynamic
from src.training.negative_sampling import sample_negative_edges
from src.training.trainer import TrainConfig, train_dynamic
from src.utils.logging import append_metrics
from src.utils.seed import set_seed

LOADERS: dict[str, type[SNAPTemporalLoader]] = {
    "collegemsg": CollegeMsgLoader,
    "bitcoinotc": BitcoinotcLoader,
    "eut": EUTLoader,
    "mooc_actions": MoocActionsLoader,
    "lastfm": LastFMLoader,
    "wikipedia": WikipediaLoader,
}


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _config_hash(*configs: dict) -> str:
    blob = json.dumps(configs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_test_pairs(graph, test_start: int, seed: int):
    """Build pos/neg test pairs for each test time step.

    Iterates t in [test_start, T-1). At each t, model sees snapshots [0..t]
    and predicts snapshot t+1. So this yields test_pairs keyed by t+1, i.e.
    test target snapshots [test_start+1, ..., T-1].
    """
    test_pairs = {}
    for t in range(test_start, graph.num_time_steps - 1):
        pos = graph.snapshots[t + 1].edge_index
        if pos.shape[1] == 0:
            continue
        neg = sample_negative_edges(
            pos, num_nodes=graph.num_nodes,
            num_samples=pos.shape[1], seed=seed + t,
        )
        test_pairs[t + 1] = {"pos": pos, "neg": neg}
    return test_pairs


def _build_model(model_cfg: dict, graph):
    """Construct a DynamicLinkPredictor by model name."""
    name = model_cfg["name"]
    if name == "gcn_ma":
        from src.models.gcn_ma.model import GCN_MA
        return GCN_MA(
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["num_heads"],
            dropout=model_cfg["dropout"],
        )
    elif name == "evolvegcn_o":
        from src.models.evolvegcn import EvolveGCN_O
        return EvolveGCN_O(
            num_nodes=graph.num_nodes,
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
    elif name == "htgn":
        from src.models.htgn import HTGN
        return HTGN(
            num_nodes=graph.num_nodes,
            feat_dim=model_cfg["feat_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            curvature=model_cfg["curvature"],
            trainable_curvature=model_cfg["trainable_curvature"],
            dropout=model_cfg["dropout"],
        )
    elif name == "dygnn":
        from src.models.dygnn import DyGNN
        return DyGNN(
            num_nodes=graph.num_nodes,
            hidden_dim=model_cfg["hidden_dim"],
            node_memory_dim=model_cfg["node_memory_dim"],
            edge_dim=model_cfg["edge_dim"],
            dropout=model_cfg["dropout"],
            decay_method=model_cfg["decay_method"],
            decay_rate=model_cfg["decay_rate"],
        )
    raise ValueError(f"Unknown model name: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    exp = _load_yaml(args.config)
    ds_cfg = _load_yaml(REPO_ROOT / exp["dataset_config"])
    model_cfg = _load_yaml(REPO_ROOT / exp["model_config"])
    config_hash = _config_hash(exp, ds_cfg, model_cfg)

    set_seed(exp["seed"])
    device = torch.device(exp["device"] if torch.cuda.is_available() else "cpu")

    # Load dataset
    raw_path = REPO_ROOT / "data" / "raw" / ds_cfg["name"] / ds_cfg["raw_filename"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Run: python scripts/download_datasets.py --dataset {ds_cfg['name']}"
        )
    loader_cls = LOADERS.get(ds_cfg["name"])
    if loader_cls is None:
        raise ValueError(f"Unknown dataset: {ds_cfg['name']}. Known: {list(LOADERS)}")
    cache_dir = REPO_ROOT / "data" / "processed" / ds_cfg["name"]
    graph = loader_cls().build(
        raw_path=raw_path,
        cache_dir=cache_dir,
        num_time_steps=ds_cfg["num_time_steps"],
        beta=ds_cfg["beta"],
    )

    # Model
    model = _build_model(model_cfg, graph)

    # Train
    train_cfg = TrainConfig(
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        epochs=model_cfg["epochs"],
        early_stop_patience=model_cfg["early_stop_patience"],
        grad_clip_max_norm=model_cfg["grad_clip_max_norm"],
        neg_sampling_seed_base=exp["seed"],
        train_ratio=ds_cfg.get("train_ratio", 0.8),
    )
    ckpt = REPO_ROOT / exp["checkpoint_dir"] / f"{exp['experiment_name']}_seed{exp['seed']}_best.pt"

    t0 = time.time()
    train_result = train_dynamic(model, graph, train_cfg, device, checkpoint_path=ckpt)

    # Load best checkpoint and evaluate on test
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    from src.data.base import temporal_split
    _, _, test_start = temporal_split(graph.num_time_steps, train_ratio=ds_cfg.get("train_ratio", 0.8))
    test_pairs = _build_test_pairs(graph, test_start, seed=999)
    test_time_steps = [t - 1 for t in sorted(test_pairs.keys())]
    test_metrics = evaluate_dynamic(model, graph, test_time_steps, test_pairs)
    runtime_s = time.time() - t0

    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "experiment_name": exp["experiment_name"],
        "model": model_cfg["name"],
        "dataset": ds_cfg["name"],
        "seed": exp["seed"],
        "auc": test_metrics["auc"],
        "ap": test_metrics["ap"],
        "val_auc": train_result["best_val_auc"],
        "val_ap": train_result["best_val_ap"],
        "best_epoch": train_result["best_epoch"],
        "runtime_s": runtime_s,
        "config_hash": config_hash,
        "git_sha": _git_sha(),
    }
    print(json.dumps(record, indent=2))
    append_metrics(REPO_ROOT / exp["metrics_path"], record)


if __name__ == "__main__":
    main()
