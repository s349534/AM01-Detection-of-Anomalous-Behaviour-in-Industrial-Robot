"""Single-run experiment: train + evaluate a vanilla AE (project_plan.md §3.1).

This module provides :func:`train_and_evaluate_ae`, the atomic unit invoked by
``run_search_ae.py`` for each sampled hyperparameter configuration.

Workflow per run:
    1. Set seed for reproducibility
    2. Build DataLoaders from processed .npy files (5 separate loaders)
    3. Instantiate ``SequenceAutoencoder`` from config
    4. Train with early stopping on validation normal reconstruction loss
    5. Compute reconstruction errors on val_normal + val_anomaly (HP selection set)
    6. Calibrate threshold (99th percentile on val_normal) and compute PR-AUC/ROC-AUC/F1 on validation
    7. Evaluate on test set (normal + anomaly) for final report only
    8. Return dict matching CSV format §4.8.8 with best_val_pr_auc from validation set

Validation set composition (HP selection):
    - val_normal: 15% of KukaNormal (early stopping + threshold calibration)
    - val_anomaly: 50% of KukaSlow (anomalies for PR-AUC calculation)

Test set composition (final report only):
    - test_normal: 15% of KukaNormal
    - test_anomaly: 50% of KukaSlow
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KukaDataset
from src.models.autoencoder import SequenceAutoencoder, fit_autoencoder
from src.utils.config import get_param
from src.utils.metrics import (
    calculate_auc,
    calculate_metrics,
    compute_anomaly_scores,
    percentile_threshold,
)

logger = logging.getLogger(__name__)

# Default epochs for validation search runs (full training happens in Fase 3.2)
DEFAULT_VAL_EPOCHS = 50


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a shuffled DataLoader for the training set (normal only)."""
    train_data = np.load(processed_dir / "train.npy")
    train_ds = KukaDataset(train_data, window_size=window_size, label=0)
    return DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )


def _build_val_normal_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the validation normal set (early stopping + threshold)."""
    val_data = np.load(processed_dir / "val_normal.npy")
    val_ds = KukaDataset(val_data, window_size=window_size, label=0)
    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )


def _build_val_anomaly_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the validation anomaly set (HP selection)."""
    val_data = np.load(processed_dir / "val_anomaly.npy")
    val_ds = KukaDataset(val_data, window_size=window_size, label=1)
    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )


def _build_test_normal_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the test normal set (final report only)."""
    test_data = np.load(processed_dir / "test_normal.npy")
    test_ds = KukaDataset(test_data, window_size=window_size, label=0)
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )


def _build_test_anomaly_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the test anomaly set (final report only)."""
    test_data = np.load(processed_dir / "test_anomaly.npy")
    test_ds = KukaDataset(test_data, window_size=window_size, label=1)
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )


def _compute_reconstruction_errors(
    model: SequenceAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "mae",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction errors and labels from a DataLoader.

    Parameters
    ----------
    model : SequenceAutoencoder
        Trained (or eval) model.
    dataloader : DataLoader
        Provides batches of ``(window, label)``.
    device : torch.device
        Device to run inference on.
    metric : {"mse", "mae"}
        Reconstruction error metric.

    Returns
    -------
    errors : np.ndarray of shape (n_samples,)
        Per-sample mean reconstruction error.
    labels : np.ndarray of shape (n_samples,)
        Binary labels (0 = normal, 1 = anomaly).
    """
    model.eval()
    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            y = batch[1] if isinstance(batch, (list, tuple)) else None

            errors = model.compute_reconstruction_error(
                x, reduction="sample", metric=metric
            )
            all_errors.append(errors.cpu().numpy())
            if y is not None:
                all_labels.append(y.cpu().numpy().flatten())

    if not all_errors:
        raise ValueError(
            f"No batches yielded from DataLoader — dataset may be empty. "
            f"Check: window_size exceeds validation samples, "
            f"or DataLoader num_workers > 0 causing multiprocessing issues. "
            f"Dataset length: {len(dataloader.dataset)}"
        )

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return errors, labels


def _compute_errors_for_loader(
    model: SequenceAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "mae",
) -> np.ndarray:
    """Compute per-sample reconstruction errors from a DataLoader (no labels returned).

    Uses metric from config (training.reconstruction_metric, default "mae").
    Intended for loaders containing a single class (all normal or all anomaly).
    """
    model.eval()
    all_errors: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            errors = model.compute_reconstruction_error(
                x, reduction="sample", metric=metric
            )
            all_errors.append(errors.cpu().numpy())

    if not all_errors:
        raise ValueError(
            f"No batches yielded from DataLoader — dataset may be empty. "
            f"Check: window_size exceeds samples, "
            f"or DataLoader num_workers > 0 causing multiprocessing issues. "
            f"Dataset length: {len(dataloader.dataset)}"
        )

    return np.concatenate(all_errors)


def train_and_evaluate_ae(
    config: dict[str, Any],
    run_id: int = 0,
    seed: int = 42,
    max_val_epochs: int | None = None,
    threshold_percentile: float = 99.0,
    device: str | torch.device | None = None,
    patience: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train a single AE configuration and evaluate on validation + test.

    Parameters
    ----------
    config : dict
        Full config dict.  Must contain ``model.window_size``,
        ``training.batch_size``, ``paths.data_processed``, plus the
        ``model.*`` keys that may have been overwritten by a
        ParameterSampler sample.
    run_id : int
        Identifier for this search run (written to CSV).
    seed : int
        Random seed for reproducibility.
    max_val_epochs : int or None
        Max epochs for this validation run.  Defaults to
        ``DEFAULT_VAL_EPOCHS``.
    threshold_percentile : float
        Percentile of validation (normal) errors used as decision threshold.
        Default 99 → ~1% FPR on normal data.
    device : str | torch.device | None
        Device to train/evaluate on. ``None`` → auto-detect.
    patience : int or None
        Early Stopping patience for validation runs. If None, reads from
        ``config["training"]["early_stopping"]["patience"]`` (default 10).
    num_workers : int
        Number of DataLoader workers for parallel data loading. Default 0.

    Returns
    -------
    dict
        Result row matching CSV format §4.8.8::

            {
                "run_id": int,
                "W": int,
                "latent_dim": int,
                "encoder_channels": str,   # e.g. "[128, 64]"
                "best_val_pr_auc": float,  # PR-AUC on validation set (HP selection)
                "best_val_f1": float,
                "best_epoch": int,
                "train_time_sec": float,
                "seed": int,
            }

        Plus a ``_test_metrics`` dict with extended metrics (not written to CSV).
    """
    _set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Extract hyperparameters from config ---
    window_size = int(get_param(config, "model.window_size", 16))
    latent_dim = int(get_param(config, "model.latent_dim", 16))
    enc_channels = tuple(get_param(config, "model.encoder.conv_channels", [128, 64]))
    batch_size = int(get_param(config, "training.batch_size", 256))
    epochs = max_val_epochs or int(get_param(config, "training.epochs", DEFAULT_VAL_EPOCHS))
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 0.0))
    patience = patience if patience is not None else int(
        get_param(config, "training.early_stopping.patience", 10)
    )
    min_delta = float(get_param(config, "training.early_stopping.min_delta", 1e-4))
    num_workers = int(get_param(config, "training.num_workers", num_workers))
    processed_dir = Path(get_param(config, "paths.data_processed", "data/processed/"))
    loss_name = get_param(config, "training.loss", "mae")
    metric = get_param(config, "training.reconstruction_metric", "mae")

    logger.info(
        "Run %d (seed=%d): W=%d, latent_dim=%d, channels=%s, epochs=%d",
        run_id, seed, window_size, latent_dim, list(enc_channels), epochs,
    )

    # --- Build dataloaders (5 separate loaders) ---
    train_loader = _build_train_loader(processed_dir, window_size, batch_size, num_workers)
    val_normal_loader = _build_val_normal_loader(processed_dir, window_size, batch_size, num_workers)
    val_anomaly_loader = _build_val_anomaly_loader(processed_dir, window_size, batch_size, num_workers)
    test_normal_loader = _build_test_normal_loader(processed_dir, window_size, batch_size, num_workers)
    test_anomaly_loader = _build_test_anomaly_loader(processed_dir, window_size, batch_size, num_workers)

    # --- Instantiate model ---
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    # --- Train + Early Stopping (uses val_normal only) ---
    start_time = time.time()

    # Use a temporary checkpoint path during validation (cleaned up afterwards)
    ckpt_path = Path(f"/tmp/_ae_val_run_{run_id}_{seed}.pth")
    history = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_normal_loader,        # early stopping on val_normal
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=str(ckpt_path),
        device=dev,
        loss=loss_name,
    )

    train_time_sec = time.time() - start_time
    best_epoch = history.get("best_epoch", 0) if history else 0

    # --- HP Selection: Threshold + Metrics on Validation (val_normal + val_anomaly) ---
    val_normal_errors = _compute_errors_for_loader(model, val_normal_loader, dev, metric=metric)
    val_anomaly_errors = _compute_errors_for_loader(model, val_anomaly_loader, dev, metric=metric)

    # Threshold: 99th percentile ONLY on val_normal
    threshold = percentile_threshold(val_normal_errors, threshold_percentile)

    # Combine for metrics (known order: normals first, then anomalies)
    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors)),
        np.ones(len(val_anomaly_errors))
    ])

    val_auc = calculate_auc(val_labels, val_errors)           # PR-AUC, ROC-AUC for HP selection
    val_pred = compute_anomaly_scores(val_errors, threshold)
    val_metrics = calculate_metrics(val_labels, val_pred)      # F1, precision, recall

    # --- Final Test (report only, NEVER used for HP selection) ---
    test_normal_errors = _compute_errors_for_loader(model, test_normal_loader, dev, metric=metric)
    test_anomaly_errors = _compute_errors_for_loader(model, test_anomaly_loader, dev, metric=metric)

    test_errors = np.concatenate([test_normal_errors, test_anomaly_errors])
    test_labels = np.concatenate([
        np.zeros(len(test_normal_errors)),
        np.ones(len(test_anomaly_errors))
    ])

    test_auc = calculate_auc(test_labels, test_errors)
    test_pred = compute_anomaly_scores(test_errors, threshold)
    test_metrics = calculate_metrics(test_labels, test_pred)

    # --- Cleanup temp checkpoint ---
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(
        "Run %d done: val_pr_auc=%.4f, val_f1=%.4f, val_roc_auc=%.4f, "
        "test_pr_auc=%.4f, test_f1=%.4f, test_roc_auc=%.4f, time=%.1fs",
        run_id,
        val_auc["pr_auc"], val_metrics["f1"], val_auc["roc_auc"],
        test_auc["pr_auc"], test_metrics["f1"], test_auc["roc_auc"],
        train_time_sec,
    )

    # --- Return result row (matches CSV format §4.8.8) ---
    # best_val_pr_auc now comes from VALIDATION set (not test!)
    result: dict[str, Any] = {
        "run_id": run_id,
        "W": window_size,
        "latent_dim": latent_dim,
        "encoder_channels": str(list(enc_channels)),
        "best_val_pr_auc": round(val_auc["pr_auc"], 6),       # HP selection metric
        "best_val_f1": round(val_metrics["f1"], 6),
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time_sec, 2),
        "seed": seed,
    }

    # Extended metrics for analysis (not written to the search CSV)
    result["_test_metrics"] = {
        "roc_auc": test_auc["roc_auc"],
        "pr_auc": test_auc["pr_auc"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "threshold": float(threshold),
        "val_mean_error": float(np.mean(val_normal_errors)),
        "val_std_error": float(np.std(val_normal_errors)),
        "test_normal_mean_error": float(test_normal_errors.mean()),
        "test_anomaly_mean_error": float(test_anomaly_errors.mean()),
    }

    return result
