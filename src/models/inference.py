"""Inference utilities for the Sequence-Aware 1D-CNN Autoencoder.

Extracted from ``src/models/autoencoder.py`` to separate inference logic
from model architecture. Provides reusable function for computing
reconstruction errors from a DataLoader.

Public API
----------
compute_errors_on_loader — compute per-sample reconstruction errors
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_errors_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric: str = "mae",
    *,
    collect_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction errors from a DataLoader.

    A shared helper used by both the training scripts (``train_ae.py``) and
    the validation scripts (``run_experiment_ae.py``) to eliminate the
    previously duplicated ``_compute_errors`` / ``_compute_errors_for_loader``
    logic.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode (or will be set to eval).
    loader : DataLoader
        Yields batches of ``(window, label)`` or just ``window``.
    device : torch.device
        Device for inference.
    metric : {"mae", "mse"}, default="mae"
        Reconstruction error metric.
    collect_labels : bool, default=False
        If ``True``, also return per-sample labels alongside errors.
        If ``False`` (default), return only the error array.

    Returns
    -------
    np.ndarray
        Per-sample mean reconstruction errors of shape ``(n_samples,)``.
    tuple[np.ndarray, np.ndarray]
        ``(errors, labels)`` when ``collect_labels=True``.
    """
    model.eval()
    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            y = batch[1] if isinstance(batch, (list, tuple)) else None

            errors = model.compute_reconstruction_error(
                x,
                reduction="sample",
                metric=metric,
            )
            all_errors.append(errors.cpu().numpy())
            if collect_labels and y is not None:
                all_labels.append(y.cpu().numpy().flatten())

    if not all_errors:
        ds = loader.dataset
        ds_len = len(ds) if isinstance(ds, torch.utils.data.Sized) else "?"
        raise ValueError(
            f"No batches yielded from DataLoader — dataset may be empty. "
            f"Check: window_size exceeds samples, "
            f"or DataLoader num_workers > 0 causing multiprocessing issues. "
            f"Dataset length: {ds_len}"
        )

    errors = np.concatenate(all_errors)
    if collect_labels:
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int32)
        return errors, labels
    return errors


def compute_val_auc(
    model: torch.nn.Module,
    val_normal_loader: DataLoader,
    val_anomaly_loader: DataLoader,
    device: torch.device,
    metric: str = "mae",
) -> dict[str, float]:
    """Compute ROC-AUC and PR-AUC on validation normal + anomaly errors.

    Encapsulates the duplicated pattern found in run_experiment_ae.py and
    train_ae.py: load errors, concatenate, build labels, call calculate_auc.

    Args:
        model: Trained model in eval mode.
        val_normal_loader: DataLoader for normal validation data.
        val_anomaly_loader: DataLoader for anomaly validation data.
        device: Device for inference.
        metric: Reconstruction error metric ("mae" or "mse").

    Returns:
        Dict with keys "roc_auc" and "pr_auc".
    """
    from src.evaluation.metrics import calculate_auc

    normal_errors = compute_errors_on_loader(model, val_normal_loader, device, metric=metric)
    anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, device, metric=metric)
    errors = np.concatenate([normal_errors, anomaly_errors])
    labels = np.concatenate([
        np.zeros(len(normal_errors), dtype=np.int32),
        np.ones(len(anomaly_errors), dtype=np.int32),
    ])
    return calculate_auc(labels, errors)