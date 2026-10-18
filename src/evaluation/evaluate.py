"""High-level evaluation pipeline for trained models.

This is the main entry point for evaluating a trained model on test/validation data.
Can be used as a library function or via CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.autoencoder import SequenceAutoencoder
from src.data.dataset import build_loader
from src.evaluation.threshold import calibrate_threshold
from src.evaluation.metrics import (
    calculate_auc, calculate_metrics, compute_anomaly_scores, evaluate_threshold
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: SequenceAutoencoder,
    test_normal_loader: DataLoader,
    test_anomaly_loader: DataLoader,
    threshold: float,
    device: torch.device,
    metric: str = "mae",
) -> dict[str, Any]:
    """Evaluate a model on test data with a given threshold.

    Parameters
    ----------
    model : SequenceAutoencoder
        Trained model (should be in eval mode).
    test_normal_loader : DataLoader
        Test normal data (label=0).
    test_anomaly_loader : DataLoader
        Test anomaly data (label=1).
    threshold : float
        Decision threshold (calibrated on validation data).
    device : torch.device
        Device for inference.
    metric : str
        Reconstruction metric for scoring.

    Returns
    -------
    dict with keys: test_pr_auc, test_roc_auc, test_f1, test_accuracy,
                    test_precision, test_recall, threshold, confusion_matrix
    """
    from src.evaluation.scoring import compute_errors_on_loader

    test_normal_errors = compute_errors_on_loader(model, test_normal_loader, device, metric=metric)
    test_anomaly_errors = compute_errors_on_loader(model, test_anomaly_loader, device, metric=metric)

    test_errors = np.concatenate([test_normal_errors, test_anomaly_errors])
    test_labels = np.concatenate([
        np.zeros(len(test_normal_errors), dtype=np.int32),
        np.ones(len(test_anomaly_errors), dtype=np.int32),
    ])

    return evaluate_threshold(test_labels, test_errors, threshold)


def evaluate_on_splits(
    model: SequenceAutoencoder,
    processed_dir: str | Path,
    window_size: int,
    batch_size: int,
    threshold: float | None,
    device: torch.device,
    metric: str = "mae",
    stride: int = 1,
    num_workers: int = 0,
    calibration_method: str = "f1_optimal",
) -> dict[str, Any]:
    """Evaluate model on all splits (val + test) from processed data directory.

    Loads data from processed_dir, builds loaders, calibrates threshold on validation,
    then evaluates on test. Returns comprehensive metrics.

    This is the function that train_ae.py should call instead of embedding evaluation logic.
    """
    processed_dir = Path(processed_dir)

    # Load data
    val_normal_data = np.load(processed_dir / "val_normal.npy")
    val_anomaly_data = np.load(processed_dir / "val_anomaly.npy")
    test_normal_data = np.load(processed_dir / "test_normal.npy")
    test_anomaly_data = np.load(processed_dir / "test_anomaly.npy")

    # Build loaders
    val_normal_loader = build_loader(val_normal_data, window_size=window_size, batch_size=batch_size, label=0, num_workers=num_workers)
    val_anomaly_loader = build_loader(val_anomaly_data, window_size=window_size, batch_size=batch_size, label=1, num_workers=num_workers)
    test_normal_loader = build_loader(test_normal_data, window_size=window_size, batch_size=batch_size, label=0, num_workers=num_workers)
    test_anomaly_loader = build_loader(test_anomaly_data, window_size=window_size, batch_size=batch_size, label=1, num_workers=num_workers)

    # Calibrate threshold on validation (if not provided)
    if threshold is None:
        threshold, val_metrics = calibrate_threshold(
            model, val_normal_loader, val_anomaly_loader, device, metric=metric, method=calibration_method
        )
        logger.info(f"Calibrated threshold: {threshold:.6f} (val F1={val_metrics['f1']:.4f})")
    else:
        val_metrics = None

    # Evaluate on test
    test_metrics = evaluate_model(model, test_normal_loader, test_anomaly_loader, threshold, device, metric)

    return {
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_evaluation(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    config: dict | None = None,
    device: str | torch.device | None = None,
    threshold: float | None = None,
    metric: str = "mae",
    calibration_method: str = "f1_optimal",
) -> dict[str, Any]:
    """Full evaluation pipeline from checkpoint.

    Loads model from checkpoint, builds data loaders from processed data,
    calibrates threshold on validation (or uses provided), evaluates on test.

    This is the CLI-compatible entry point.
    """
    from src.utils.config import load_config, get_param

    if config is None:
        config = load_config()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model = SequenceAutoencoder.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()

    window_size = int(get_param(config, "model.window_size", 16))
    batch_size = int(get_param(config, "training.batch_size", 256))

    return evaluate_on_splits(
        model=model,
        processed_dir=processed_dir,
        window_size=window_size,
        batch_size=batch_size,
        threshold=threshold,
        device=dev,
        metric=metric,
        calibration_method=calibration_method,
    )