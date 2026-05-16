"""Training loop for dynamic link prediction."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.base import DynamicGraph, temporal_split
from src.eval.evaluator import evaluate_dynamic
from src.models.base import DynamicLinkPredictor
from src.training.losses import link_prediction_loss
from src.training.negative_sampling import sample_negative_edges


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stop_patience: int = 20
    grad_clip_max_norm: float = 5.0
    neg_sampling_seed_base: int = 0


def train_dynamic(
    model: DynamicLinkPredictor,
    graph: DynamicGraph,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict:
    """Train `model` on `graph` and return the best validation result.

    Returns:
        {
            "best_val_auc": float,
            "best_val_ap": float,
            "best_epoch": int,
            "history": list[dict] (per-epoch loss + val_auc),
        }
    """
    train_end, val_step, _ = temporal_split(graph.num_time_steps, train_ratio=0.8)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)

    # Validation: predict snapshot val_step+1 (== train_end) given snapshots [0..val_step].
    val_pos = graph.snapshots[val_step + 1].edge_index
    val_neg = sample_negative_edges(
        val_pos, num_nodes=graph.num_nodes,
        num_samples=val_pos.shape[1], seed=config.neg_sampling_seed_base + 99,
    )
    val_pairs = {val_step + 1: {"pos": val_pos, "neg": val_neg}}

    history: list[dict] = []
    best_val_auc = -1.0
    best_val_ap = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        # Training predicts snapshots [1..val_step] from inputs [0..val_step-1].
        # Never targets snapshots train_end..T-1 (those belong to val/test).
        for t in tqdm(range(val_step), desc=f"Epoch {epoch}", leave=False):
            if t + 1 >= graph.num_time_steps:
                break
            Z_t = model(graph.snapshots, time_step=t)
            pos = graph.snapshots[t + 1].edge_index.to(device)
            if pos.shape[1] == 0:
                continue
            neg = sample_negative_edges(
                pos, num_nodes=graph.num_nodes,
                num_samples=pos.shape[1],
                seed=config.neg_sampling_seed_base + epoch * 1000 + t,
            ).to(device)

            pos_logits = model.predict_link(Z_t, pos)
            neg_logits = model.predict_link(Z_t, neg)
            loss = link_prediction_loss(pos_logits, neg_logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip_max_norm
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        val_metrics = evaluate_dynamic(
            model, graph, time_steps=[val_step], test_pairs=val_pairs
        )
        history.append(
            {"epoch": epoch, "loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        )
        print(
            f"Epoch {epoch:3d}: loss={avg_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_ap={val_metrics['ap']:.4f}"
        )

        scheduler.step(val_metrics["auc"])
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_ap = val_metrics["ap"]
            best_epoch = epoch
            patience = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_auc": best_val_auc},
                    checkpoint_path,
                )
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience}).")
                break

    return {
        "best_val_auc": best_val_auc,
        "best_val_ap": best_val_ap,
        "best_epoch": best_epoch,
        "history": history,
    }
