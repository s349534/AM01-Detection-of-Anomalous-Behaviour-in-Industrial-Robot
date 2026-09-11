"""Single-run experiment: train + evaluate a vanilla AE (project_plan.md §3.1).

This module provides :func:`train_and_evaluate_ae`, the atomic unit invoked by
``run_search_ae.py`` for each sampled hyperparameter configuration.

Workflow per run:
    1. Set seed for reproducibility
    2. Build DataLoader from processed .npy files
    3. Instantiate ``SequenceAutoencoder`` from config
    4. Train with early stopping on validation loss
    5. Compute reconstruction errors on validation (normal) + test (normal + anomaly)
    6. Calibrate threshold (99th percentile on validation) and compute PR-AUC
    7. Return dict matching CSV format §4.8.8

Note on val_pr_auc: the validation set is normal-only (§2.3), so PR-AUC
cannot be computed there directly.  The threshold is calibrated on validation
errors (99th percentile → ~1% FPR), then PR-AUC and F1 are evaluated on the
test set (normal + anomaly).  The column name ``best_val_pr_auc`` follows the
project plan convention and represents the HP-selection metric.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

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
) -> DataLoader:
    """Build a shuffled DataLoader for the training set (normal only)."""
    train_data = np.load(processed_dir / "train.npy")
    train_ds = KukaDataset(train_data, window_size=window_size, label=0)
    return DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )


def _build_val_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
) -> DataLoader:
    """Build a DataLoader for the validation set (normal only)."""
    val_data = np.load(processed_dir / "val.npy")
    val_ds = KukaDataset(val_data, window_size=window_size, label=0)
    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )


def _build_test_loader(
    processed_dir: Path,
    window_size: int,
    batch_size: int,
) -> DataLoader:
    """Build a DataLoader for the test set (normal + anomaly concatenated)."""
    test_normal_data = np.load(processed_dir / "test_normal.npy")
    test_anomaly_data = np.load(processed_dir / "test_anomaly.npy")

    test_normal_ds = KukaDataset(test_normal_data, window_size=window_size, label=0)
    test_anomaly_ds = KukaDataset(test_anomaly_data, window_size=window_size, label=1)
    test_ds = ConcatDataset([test_normal_ds, test_anomaly_ds])

    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )


def _compute_reconstruction_errors(
    model: SequenceAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "mse",
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

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return errors, labels


def train_and_evaluate_ae(
    config: dict[str, Any],
    run_id: int = 0,
    seed: int = 42,
    max_val_epochs: int | None = None,
    threshold_percentile: float = 99.0,
    device: str | torch.device | None = None,
    patience: int = 5,
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
    patience : int
        Early Stopping patience for validation runs (shorter than final training).

    Returns
    -------
    dict
        Result row matching CSV format §4.8.8::

            {
                "run_id": int,
                "W": int,
                "latent_dim": int,
                "encoder_channels": str,   # e.g. "[128, 64]"
                "best_val_pr_auc": float,
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
    processed_dir = Path(get_param(config, "paths.data_processed", "data/processed/"))

    logger.info(
        "Run %d (seed=%d): W=%d, latent_dim=%d, channels=%s, epochs=%d",
        run_id, seed, window_size, latent_dim, list(enc_channels), epochs,
    )

    # --- Build dataloaders ---
    train_loader = _build_train_loader(processed_dir, window_size, batch_size)
    val_loader = _build_val_loader(processed_dir, window_size, batch_size)
    test_loader = _build_test_loader(processed_dir, window_size, batch_size)

    # --- Instantiate model ---
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    # --- Train ---
    start_time = time.time()

    # Use a temporary checkpoint path during validation (cleaned up afterwards)
    ckpt_path = Path(f"/tmp/_ae_val_run_{run_id}_{seed}.pth")
    history = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        checkpoint_path=str(ckpt_path),
        device=dev,
    )

    train_time_sec = time.time() - start_time
    best_epoch = len(history["val_loss"]) if history else 0

    # --- Calibrate threshold on validation (normal only) ---
    val_errors, _ = _compute_reconstruction_errors(model, val_loader, dev, metric="mse")
    threshold = percentile_threshold(val_errors, threshold_percentile)

    # --- Evaluate on test (normal + anomaly) ---
    test_errors, test_labels = _compute_reconstruction_errors(
        model, test_loader, dev, metric="mse"
    )

    auc_scores = calculate_auc(test_labels, test_errors)
    test_pred = compute_anomaly_scores(test_errors, threshold)
    test_metrics = calculate_metrics(test_labels, test_pred)

    # --- Cleanup temp checkpoint ---
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(
        "Run %d done: pr_auc=%.4f, f1=%.4f, roc_auc=%.4f, time=%.1fs",
        run_id,
        auc_scores["pr_auc"],
        test_metrics["f1"],
        auc_scores["roc_auc"],
        train_time_sec,
    )

    # --- Return result row (matches CSV format §4.8.8) ---
    result: dict[str, Any] = {
        "run_id": run_id,
        "W": window_size,
        "latent_dim": latent_dim,
        "encoder_channels": str(list(enc_channels)),
        "best_val_pr_auc": round(auc_scores["pr_auc"], 6),
        "best_val_f1": round(test_metrics["f1"], 6),
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time_sec, 2),
        "seed": seed,
    }

    # Extended metrics for analysis (not written to the search CSV)
    result["_test_metrics"] = {
        "roc_auc": auc_scores["roc_auc"],
        "pr_auc": auc_scores["pr_auc"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "threshold": float(threshold),
        "val_mean_error": float(np.mean(val_errors)),
        "val_std_error": float(np.std(val_errors)),
        "test_normal_mean_error": float(test_errors[test_labels == 0].mean()),
        "test_anomaly_mean_error": float(test_errors[test_labels == 1].mean()),
    }

    return result
