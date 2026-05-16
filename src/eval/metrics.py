"""Link-prediction metrics (AUC, AP) via sklearn."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def compute_link_prediction_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    return {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_ap(y_true, y_score),
    }
