"""Grid-search β and hidden_dim on Bitcoinotc (Plan 2, spec §9.4).

Writes a row per combo to results/beta_grid_bitcoinotc.jsonl with
val AUC, val AP, and the chosen test AUC at best val.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.base import temporal_split
from src.data.loaders.bitcoinotc import BitcoinotcLoader
from src.eval.evaluator import evaluate_dynamic
from src.models.gcn_ma.model import GCN_MA
from src.training.negative_sampling import sample_negative_edges
from src.training.trainer import TrainConfig, train_dynamic
from src.utils.seed import set_seed


def _build_test_pairs(graph, test_start, seed):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/beta_grid_bitcoinotc.jsonl",
                        help="JSONL file to append grid results to")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Fewer than the full 200 to keep grid fast")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_path = REPO_ROOT / "data" / "raw" / "bitcoinotc" / "soc-sign-bitcoinotc.csv.gz"
    if not raw_path.exists():
        raise FileNotFoundError("Run: python scripts/download_datasets.py --dataset bitcoinotc")

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = [(beta, hd) for beta in (0.7, 0.8, 0.9) for hd in (64, 128)]
    best = None
    for beta, hidden_dim in grid:
        print(f"\n=== β={beta}, hidden_dim={hidden_dim} ===")
        graph = BitcoinotcLoader().build(
            raw_path=raw_path,
            cache_dir=REPO_ROOT / "data" / "processed" / "bitcoinotc",
            num_time_steps=62,
            beta=beta,
        )

        model = GCN_MA(feat_dim=3, hidden_dim=hidden_dim, num_heads=8 if hidden_dim == 128 else 4, dropout=0.1)
        train_cfg = TrainConfig(
            lr=1e-3, weight_decay=1e-5,
            epochs=args.epochs, early_stop_patience=20,
            grad_clip_max_norm=5.0,
            neg_sampling_seed_base=42, train_ratio=0.8,
        )
        ckpt = REPO_ROOT / "results" / "checkpoints" / f"beta_grid_b{beta}_h{hidden_dim}.pt"

        t0 = time.time()
        train_result = train_dynamic(model, graph, train_cfg, device, checkpoint_path=ckpt)

        # Load best & evaluate on test
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        _, _, test_start = temporal_split(graph.num_time_steps, train_ratio=0.8)
        test_pairs = _build_test_pairs(graph, test_start, seed=999)
        test_time_steps = [t - 1 for t in sorted(test_pairs.keys())]
        test_metrics = evaluate_dynamic(model, graph, test_time_steps, test_pairs)
        runtime_s = time.time() - t0

        record = {
            "date": datetime.now(timezone.utc).isoformat(),
            "beta": beta, "hidden_dim": hidden_dim,
            "val_auc": train_result["best_val_auc"], "val_ap": train_result["best_val_ap"],
            "test_auc": test_metrics["auc"], "test_ap": test_metrics["ap"],
            "best_epoch": train_result["best_epoch"], "runtime_s": runtime_s,
        }
        print(json.dumps(record))
        with output_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if best is None or record["val_auc"] > best["val_auc"]:
            best = record

    print(f"\n=== Best by val AUC ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
