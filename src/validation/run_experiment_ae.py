"""Single-run experiment: train + evaluate a vanilla AE (project_plan.md §3.1).

This module provides :func:`train_and_evaluate_ae`, the atomic unit invoked by
``run_search_ae.py`` for each sampled hyperparameter configuration.

Workflow per run (Stage 1 - Model Selection):
    1. Set seed for reproducibility
    2. Build DataLoaders from processed .npy files (3 loaders: train, val_normal, val_anomaly)
    3. Instantiate ``SequenceAutoencoder`` from config
    4. Train with early stopping on validation normal reconstruction loss
    5. Compute reconstruction errors on val_normal + val_anomaly (HP selection set)
    6. Compute ROC-AUC and PR-AUC on validation (threshold-free, no binarization)
    7. Save raw errors + labels to .npz for Stage 2 threshold calibration
    8. Return dict with val_roc_auc, val_pr_auc, errors_file (no test metrics, no threshold)

Validation set composition (HP selection):
    - val_normal: 15% of KukaNormal (early stopping)
    - val_anomaly: 50% of KukaSlow (anomalies for AUC calculation)

Test set is NOT accessed in this stage.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import build_loader
from src.models.autoencoder import (
    SequenceAutoencoder,
    compute_errors_on_loader,
    fit_autoencoder,
)
from src.utils.config import get_param
from src.utils.metrics import calculate_auc
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Default epochs for validation search runs (full training happens in Fase 3.2)
DEFAULT_VAL_EPOCHS = 50


def train_and_evaluate_ae(
    config: dict[str, Any],
    run_id: int = 0,
    seed: int = 42,
    max_val_epochs: int | None = None,
    device: str | torch.device | None = None,
    patience: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train a single AE configuration and evaluate on validation (Stage 1).

    This is the Model Selection stage: computes ROC-AUC and PR-AUC on
    val_normal + val_anomaly WITHOUT any threshold. Saves raw errors+labels
    to .npz for Stage 2 threshold calibration.

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
        Result row for CSV (Stage 1 format)::

            {
                "run_id": int,
                "W": int,
                "latent_dim": int,
                "val_roc_auc": float,   # ROC-AUC on validation (threshold-free)
                "val_pr_auc": float,    # PR-AUC on validation (threshold-free)
                "best_epoch": int,
                "train_time_sec": float,
                "seed": int,
                "errors_file": str,     # path to .npz with raw errors+labels
            }

        No test metrics, no threshold, no F1.
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Extract hyperparameters from config ---
    window_size = int(get_param(config, "model.window_size", 16))
    latent_dim = int(get_param(config, "model.latent_dim", 16))
    enc_channels = tuple(get_param(config, "model.encoder.conv_channels", [64, 32, 16]))
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

    # --- Build dataloaders (3 loaders only: train, val_normal, val_anomaly) ---
    train_data = np.load(processed_dir / "train.npy")
    val_normal_data = np.load(processed_dir / "val_normal.npy")
    val_anomaly_data = np.load(processed_dir / "val_anomaly.npy")

    train_loader = build_loader(train_data, window_size=window_size,
                                batch_size=batch_size, label=0, shuffle=True, num_workers=num_workers)
    val_normal_loader = build_loader(val_normal_data, window_size=window_size,
                                     batch_size=batch_size, label=0, num_workers=num_workers)
    val_anomaly_loader = build_loader(val_anomaly_data, window_size=window_size,
                                      batch_size=batch_size, label=1, num_workers=num_workers)

    # --- Instantiate model ---
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    # --- Train + Early Stopping (uses val_normal only) ---
    start_time = time.time()

    # No checkpoint during validation search: early stopping keeps best weights
    # in memory and the checkpoint would only be a temp file /tmp that gets
    # deleted immediately.  The real checkpoint is saved by train_ae.py (Stage 2).
    history = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_normal_loader,        # early stopping on val_normal
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        device=dev,
        loss=loss_name,
    )

    train_time_sec = time.time() - start_time
    best_epoch = history.get("best_epoch", 0) if history else 0

    # --- HP Selection: ROC-AUC + PR-AUC on Validation (val_normal + val_anomaly) ---
    # Compute errors via shared utility (returns errors array; labels reconstructed manually below)
    val_normal_errors = compute_errors_on_loader(model, val_normal_loader, dev, metric=metric)
    val_anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, dev, metric=metric)

    # Concatenate errors and create labels (0 = normal, 1 = anomaly)
    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors), dtype=np.int32),
        np.ones(len(val_anomaly_errors), dtype=np.int32)
    ])

    # Compute AUC metrics (threshold-free: no binarization, no fixed threshold)
    val_auc = calculate_auc(val_labels, val_errors)   # returns {"roc_auc": ..., "pr_auc": ...}

    # --- Save raw errors + labels for Stage 2 threshold calibration ---
    errors_dir = Path("reports/errors")
    errors_dir.mkdir(parents=True, exist_ok=True)
    errors_filename = f"ae_errors_run{run_id}_seed{seed}_W{window_size}_latent{latent_dim}.npz"
    errors_path = errors_dir / errors_filename
    np.savez_compressed(errors_path, errors=val_errors, labels=val_labels)
    logger.info("Saved validation errors to %s", errors_path)

    logger.info(
        "Run %d done: val_roc_auc=%.4f, val_pr_auc=%.4f, best_epoch=%d, time=%.1fs",
        run_id,
        val_auc["roc_auc"], val_auc["pr_auc"], best_epoch, train_time_sec,
    )

    # --- Return result row (Stage 1 CSV format) ---
    result: dict[str, Any] = {
        "run_id": run_id,
        "W": window_size,
        "latent_dim": latent_dim,
        "val_roc_auc": round(val_auc["roc_auc"], 6),
        "val_pr_auc": round(val_auc["pr_auc"], 6),
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time_sec, 2),
        "seed": seed,
        "errors_file": str(errors_path),
    }

    return result
