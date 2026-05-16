"""Pooled link-prediction evaluator over a list of test time steps."""
import numpy as np
import torch

from src.data.base import DynamicGraph
from src.eval.metrics import compute_link_prediction_metrics
from src.models.base import DynamicLinkPredictor

TestPairs = dict[int, dict[str, torch.Tensor]]
# test_pairs[t+1] = {"pos": [2, P], "neg": [2, Q]}


@torch.no_grad()
def evaluate_dynamic(
    model: DynamicLinkPredictor,
    graph: DynamicGraph,
    time_steps: list[int],
    test_pairs: TestPairs,
) -> dict[str, float]:
    """Evaluate the model on each step t ∈ time_steps; pool scores across all
    steps before computing AUC and AP.

    At step t, the model sees snapshots [0..t] and predicts edges of
    snapshot t+1 stored in test_pairs[t + 1].
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for t in time_steps:
        pairs = test_pairs.get(t + 1)
        if pairs is None or pairs["pos"].shape[1] == 0:
            continue
        Z = model(graph.snapshots, time_step=t)
        pos_logits = model.predict_link(Z, pairs["pos"])
        neg_logits = model.predict_link(Z, pairs["neg"])
        scores = torch.sigmoid(torch.cat([pos_logits, neg_logits])).cpu().numpy()
        labels = np.concatenate(
            [
                np.ones(pairs["pos"].shape[1], dtype=np.int64),
                np.zeros(pairs["neg"].shape[1], dtype=np.int64),
            ]
        )
        all_scores.append(scores)
        all_labels.append(labels)

    if not all_scores:
        raise RuntimeError("evaluate_dynamic: no non-empty pairs to evaluate")

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    return compute_link_prediction_metrics(y_true, y_score)
