"""Anomaly scoring functions.

Extracts the scoring logic from SequenceAutoencoder.compute_reconstruction_error
and compute_errors_on_loader to make them reusable without a model instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.autoencoder import SequenceAutoencoder, compute_errors_on_loader as _compute_errors_on_loader
from src.data.dataset import build_loader


def compute_errors_on_loader(
    model: SequenceAutoencoder,
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
    model : SequenceAutoencoder
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
    return _compute_errors_on_loader(model, loader, device, metric=metric, collect_labels=collect_labels)


def compute_reconstruction_errors(
    model: SequenceAutoencoder,
    loader: DataLoader,
    device: torch.device,
    metric: str = "mae",
    reduction: str = "sample",
) -> np.ndarray:
    """Compute per-sample reconstruction errors.

    Wrapper around model.compute_reconstruction_error for DataLoader iteration.

    Parameters
    ----------
    model : SequenceAutoencoder
        Trained model in eval mode.
    loader : DataLoader
        DataLoader yielding windows.
    device : torch.device
        Device for inference.
    metric : {"mae", "mse"}, default="mae"
        Reconstruction error metric.
    reduction : {"sample", "mean", "none"}, default="sample"
        Reduction method for errors.

    Returns
    -------
    np.ndarray
        Per-sample mean reconstruction errors of shape ``(n_samples,)``.
    """
    return compute_errors_on_loader(model, loader, device, metric=metric)


def compute_errors_from_arrays(
    model: SequenceAutoencoder,
    data: np.ndarray,
    window_size: int,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    metric: str = "mae",
    stride: int = 1,
    label: int = 0,
) -> np.ndarray:
    """Convenience: compute errors directly from numpy array (no DataLoader needed).

    Parameters
    ----------
    model : SequenceAutoencoder
        Trained model in eval mode.
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    window_size : int
        Number of timesteps per window.
    batch_size : int, default=256
        Batch size for inference.
    device : str | torch.device, default="cpu"
        Device for inference.
    metric : {"mae", "mse"}, default="mae"
        Reconstruction error metric.
    stride : int, default=1
        Stride between consecutive windows.
    label : int, default=0
        Label to assign to windows (0 = normal, 1 = anomaly).

    Returns
    -------
    np.ndarray
        Per-sample mean reconstruction errors of shape ``(n_samples,)``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    loader = build_loader(data, window_size=window_size, batch_size=batch_size, label=label, stride=stride)
    return compute_errors_on_loader(model, loader, device, metric=metric)