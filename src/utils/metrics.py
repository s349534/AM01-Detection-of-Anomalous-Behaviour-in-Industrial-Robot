"""Evaluation metrics for anomaly detection.

This module provides:
- calculate_metrics: Precision, recall, f1, accuracy
- calculate_auc: ROC-AUC and PR-AUC scores
- find_optimal_threshold: F1-maximizing threshold
- percentile_threshold: Percentile-based threshold (for validation set)
- compute_anomaly_scores: Binarize continuous scores using a threshold
- evaluate_threshold: Full evaluation pipeline with threshold selection
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate classification metrics from binary predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 = normal, 1 = anomaly).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision, recall, f1.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
    }


def calculate_auc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Calculate ROC-AUC and PR-AUC scores from continuous anomaly scores.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 = normal, 1 = anomaly).
    scores : array-like of shape (n_samples,)
        Continuous anomaly scores (higher = more anomalous).

    Returns
    -------
    dict
        Dictionary with keys: roc_auc, pr_auc.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    # Handle degenerate cases
    unique_labels = np.unique(y_true)
    if len(unique_labels) <= 1:
        return {"roc_auc": 0.5, "pr_auc": 0.5}

    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    """Compute a threshold as a percentile of reconstruction errors.

    Used on the **validation set (normal only)** to set a decision boundary.
    A high percentile (e.g. 99) means only the top ~1% of "most abnormal"
    normal samples would be flagged — controlling the false positive rate.

    Parameters
    ----------
    scores : array-like
        Reconstruction errors from the validation set (normal data).
    percentile : float
        Percentile in range [0, 100] (e.g. 99 for 99th percentile).

    Returns
    -------
    float
        The score at the given percentile.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.percentile(scores, percentile))


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find the threshold that maximizes F1-score on the given data.

    Sweeps all unique score values as candidate thresholds and selects the
    one that gives the best F1.  Useful for calibrating on the validation set.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels.
    scores : array-like of shape (n_samples,)
        Continuous anomaly scores.

    Returns
    -------
    threshold : float
        The score threshold that maximizes F1.
    best_metrics : dict
        Metrics at the optimal threshold (accuracy, precision, recall, f1).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    # Candidate thresholds: all unique score values + midpoints
    candidate_thresholds = np.unique(scores)
    # Also try midpoints between consecutive thresholds for finer granularity
    midpoints = (candidate_thresholds[:-1] + candidate_thresholds[1:]) / 2
    all_candidates = np.concatenate([candidate_thresholds, midpoints]) if len(candidate_thresholds) > 1 else candidate_thresholds

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}

    for threshold in all_candidates:
        y_pred = (scores >= threshold).astype(np.int32)
        metrics = calculate_metrics(y_true, y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    if not best_metrics:
        best_metrics = {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return best_threshold, best_metrics


def compute_anomaly_scores(
    errors: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binarize continuous reconstruction errors using a threshold.

    Parameters
    ----------
    errors : array-like of shape (n_samples,)
        Continuous reconstruction errors (higher = more anomalous).
    threshold : float
        Decision boundary. Errors >= threshold are flagged as anomalies.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Binary predictions (0 = normal, 1 = anomaly).
    """
    errors = np.asarray(errors, dtype=np.float64)
    return (errors >= threshold).astype(np.int32)


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Full evaluation at a fixed threshold.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    scores : array-like
        Continuous anomaly scores.
    threshold : float
        Fixed decision threshold.

    Returns
    -------
    dict with keys: threshold, pr_auc, roc_auc, accuracy, precision, recall, f1,
                    confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    auc_scores = calculate_auc(y_true, scores)
    y_pred = compute_anomaly_scores(scores, threshold)
    metrics = calculate_metrics(y_true, y_pred)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return {
        "threshold": float(threshold),
        "roc_auc": auc_scores["roc_auc"],
        "pr_auc": auc_scores["pr_auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def reconstruction_error_to_scores(
    errors: np.ndarray,
    val_normal_errors: np.ndarray | None = None,
    percentile: float = 99.0,
) -> np.ndarray:
    """Convert raw reconstruction errors to standardized anomaly scores.

    If val_normal_errors is provided, computes a z-score-like normalization
    using the validation (normal) statistics. Otherwise returns the errors
    as-is.

    Parameters
    ----------
    errors : array-like
        Raw reconstruction errors for the samples to score.
    val_normal_errors : array-like or None
        Reconstruction errors from the validation set (normal data only),
        used to compute mean/std for standardization.
    percentile : float
        Reserved for future percentile-based scaling. Currently unused
        (kept for API compatibility).

    Returns
    -------
    np.ndarray
        Standardized anomaly scores (higher = more anomalous).
    """
    errors = np.asarray(errors, dtype=np.float64)
    if val_normal_errors is not None:
        val_normal_errors = np.asarray(val_normal_errors, dtype=np.float64)
        mean = val_normal_errors.mean()
        std = val_normal_errors.std()
        if std > 0:
            return (errors - mean) / std
    return errors
