import numpy as np
import pytest

from src.eval.metrics import compute_ap, compute_auc, compute_link_prediction_metrics


def test_auc_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc(y_true, y_score) == pytest.approx(1.0)


def test_auc_random_balanced():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.random(size=1000)
    auc = compute_auc(y_true, y_score)
    assert 0.4 < auc < 0.6  # near 0.5


def test_ap_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_ap(y_true, y_score) == pytest.approx(1.0)


def test_compute_link_prediction_metrics_returns_dict():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.6])
    metrics = compute_link_prediction_metrics(y_true, y_score)
    assert set(metrics.keys()) == {"auc", "ap"}
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0
