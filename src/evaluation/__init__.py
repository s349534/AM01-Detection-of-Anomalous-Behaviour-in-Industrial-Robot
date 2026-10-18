"""Evaluation module for AM01 anomaly detection.

Public API:
    - evaluate_model: Evaluate a trained model on given data loaders
    - calibrate_threshold: Find optimal threshold on validation data
    - compute_anomaly_scores: Apply threshold to get binary predictions
    - calculate_metrics: Precision, recall, F1, accuracy
    - calculate_auc: ROC-AUC, PR-AUC
    - run_evaluation: Full evaluation pipeline (CLI-compatible)
"""

from .evaluate import evaluate_model, evaluate_on_splits, run_evaluation
from .threshold import calibrate_threshold, find_optimal_threshold, percentile_threshold
from .scoring import (
    compute_reconstruction_errors,
    compute_errors_on_loader,
    compute_errors_from_arrays,
)
from .metrics import (
    calculate_metrics,
    calculate_auc,
    compute_anomaly_scores,
    evaluate_threshold,
)

__all__ = [
    "evaluate_model",
    "evaluate_on_splits",
    "run_evaluation",
    "calibrate_threshold",
    "find_optimal_threshold",
    "percentile_threshold",
    "compute_reconstruction_errors",
    "compute_errors_on_loader",
    "compute_errors_from_arrays",
    "calculate_metrics",
    "calculate_auc",
    "compute_anomaly_scores",
    "evaluate_threshold",
]