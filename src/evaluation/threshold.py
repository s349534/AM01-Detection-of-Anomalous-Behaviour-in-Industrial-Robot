"""Threshold calibration for anomaly detection.

Provides functions to calibrate decision thresholds on validation data,
separate from training. This enables re-calibration on new data without retraining.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import find_optimal_threshold, percentile_threshold, calculate_auc
from src.models.inference import compute_errors_on_loader

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from src.models.autoencoder import SequenceAutoencoder


def calibrate_threshold(
    model: SequenceAutoencoder,
    val_normal_loader: DataLoader,
    val_anomaly_loader: DataLoader,
    device: torch.device,
    metric: str = "mae",
    method: str = "f1_optimal",  # "f1_optimal" | "percentile_99" | "percentile_95"
    percentile: float = 99.0,
) -> tuple[float, dict[str, Any]]:
    """Calibrate decision threshold on validation data.

    This is the KEY function that was embedded in train_ae.py.
    Now reusable: can re-calibrate any trained model on any validation data.

    Parameters
    ----------
    model : SequenceAutoencoder
        Trained model in eval mode.
    val_normal_loader : DataLoader
        Validation normal data (label=0).
    val_anomaly_loader : DataLoader
        Validation anomaly data (label=1).
    device : torch.device
        Device for inference.
    metric : str
        Reconstruction metric ("mae" or "mse").
    method : str
        Calibration method:
        - "f1_optimal": Maximize F1 on val_normal + val_anomaly (default)
        - "percentile_99": 99th percentile of val_normal errors
        - "percentile_95": 95th percentile of val_normal errors
    percentile : float
        Percentile value for percentile-based methods.

    Returns
    -------
    threshold : float
        Calibrated threshold.
    metrics : dict
        Validation metrics at this threshold (f1, precision, recall, roc_auc, pr_auc).
    """
    # Compute errors on validation sets
    val_normal_errors = compute_errors_on_loader(model, val_normal_loader, device, metric=metric)
    val_anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, device, metric=metric)

    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors), dtype=np.int32),
        np.ones(len(val_anomaly_errors), dtype=np.int32),
    ])

    if method == "f1_optimal":
        threshold, val_metrics = find_optimal_threshold(val_labels, val_errors)
    elif method.startswith("percentile"):
        # Percentile on normal only (controls FPR)
        threshold = percentile_threshold(val_normal_errors, percentile)
        from src.evaluation.metrics import compute_anomaly_scores, calculate_metrics
        val_pred = compute_anomaly_scores(val_errors, threshold)
        val_metrics = calculate_metrics(val_labels, val_pred)
        val_metrics.update(calculate_auc(val_labels, val_errors))
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(threshold), val_metrics