"""Single-run experiment: train + evaluate an AAE (project_plan.md §4.1).

This module provides :func:`train_and_evaluate_aae`, the atomic unit invoked by
``run_search_aae.py`` for each sampled hyperparameter configuration.

Workflow per run:
    1. Set seed for reproducibility
    2. Build DataLoader from processed .npy files (same as AE)
    3. Instantiate ``AdversarialAutoencoder`` from config
    4. Train with early stopping on validation reconstruction loss
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
from src.models.adversarial_ae import AdversarialAutoencoder, fit_aae
from src.utils.config import get_param
from src.utils.metrics import (
    calculate_auc,
    calculate_metrics,
    compute_anomaly_scores,
    percentile_threshold,
)

logger = logging.getLogger(__name__)

# Default epochs for validation search runs (full training happens in Fase 4.2)
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
    model: AdversarialAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "mse",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction errors and labels from a DataLoader.

    Parameters
    ----------
    model : AdversarialAutoencoder
        Trained (or eval) model.  Reconstruction errors are computed via
        ``model.compute_reconstruction_error`` — the AAE discriminator is
        **not** involved at inference time (§4.1.3: scoring is reconstruction-based).
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


def train_and_evaluate_aae(
    config: dict[str, Any],
    run_id: int = 0,
    seed: int = 42,
    max_val_epochs: int | None = None,
    threshold_percentile: float = 99.0,
    device: str | torch.device | None = None,
    patience: int = 10,
) -> dict[str, Any]:
    """Train a single AAE configuration and evaluate on validation + test.

    Parameters
    ----------
    config : dict
        Full config dict.  Must contain ``model.window_size``,
        ``training.batch_size``, ``paths.data_processed``, plus the
        ``model.*`` keys that may have been overwritten by a
        ParameterSampler sample (including ``model.discriminator.hidden_layers``).
    run_id : int
        Identifier for this search run (written to CSV).
    seed : int
        Random seed for reproducibility.
    max_val_epochs : int or None
        Max epochs for this validation run.  Defaults to
        ``DEFAULT_VAL_EPOCHS`` (50).
    threshold_percentile : float
        Percentile of validation (normal) errors used as decision threshold.
        Default 99 → ~1% FPR on normal data.
    device : str | torch.device | None
        Device to train/evaluate on. ``None`` → auto-detect.
    patience : int
        Early Stopping patience for validation runs.

    Returns
    -------
    dict
        Result row matching CSV format §4.8.8::

            {
                "run_id": int,
                "reconstruction_weight": float,
                "adversarial_weight": float,
                "discriminator_hidden_layers": str,  # e.g. "[32, 16]"
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
    disc_hidden = tuple(get_param(config, "model.discriminator.hidden_layers", [32, 16]))
    batch_size = int(get_param(config, "training.batch_size", 256))
    epochs = max_val_epochs or int(get_param(config, "training.epochs", DEFAULT_VAL_EPOCHS))
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 0.0))
    reconstruction_weight = float(
        get_param(config, "training.reconstruction_weight", 1.0)
    )
    adversarial_weight = float(
        get_param(config, "training.adversarial_weight", 0.1)
    )
    disc_updates_per_gen = int(
        get_param(config, "training.discriminator_updates_per_gen", 1)
    )
    metric = str(get_param(config, "training.loss", "mse"))
    processed_dir = Path(get_param(config, "paths.data_processed", "data/processed/"))

    logger.info(
        "AAE Run %d (seed=%d): W=%d, latent_dim=%d, channels=%s, "
        "disc_hidden=%s, recon_w=%.2f, adv_w=%.3f, epochs=%d",
        run_id, seed, window_size, latent_dim, list(enc_channels),
        list(disc_hidden), reconstruction_weight, adversarial_weight, epochs,
    )

    # --- Build dataloaders ---
    train_loader = _build_train_loader(processed_dir, window_size, batch_size)
    val_loader = _build_val_loader(processed_dir, window_size, batch_size)
    test_loader = _build_test_loader(processed_dir, window_size, batch_size)

    # --- Instantiate model ---
    model = AdversarialAutoencoder.from_config(config)
    model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "AAE Model parameters: %d total, %d trainable (encoder+decoder+disc)",
        total_params, trainable_params,
    )

    # --- Train ---
    start_time = time.time()
    ckpt_path = Path(f"/tmp/_aae_val_run_{run_id}_{seed}.pth")
    history = fit_aae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        reconstruction_weight=reconstruction_weight,
        adversarial_weight=adversarial_weight,
        discriminator_updates_per_gen=disc_updates_per_gen,
        patience=patience,
        checkpoint_path=str(ckpt_path),
        device=dev,
        metric=metric,
    )

    train_time_sec = time.time() - start_time
    best_epoch = len(history["val_loss"]) if history else 0

    # --- Calibrate threshold on validation (normal only) ---
    val_errors, _ = _compute_reconstruction_errors(model, val_loader, dev, metric=metric)
    threshold = percentile_threshold(val_errors, threshold_percentile)

    # --- Evaluate on test (normal + anomaly) ---
    test_errors, test_labels = _compute_reconstruction_errors(
        model, test_loader, dev, metric=metric
    )

    auc_scores = calculate_auc(test_labels, test_errors)
    test_pred = compute_anomaly_scores(test_errors, threshold)
    test_metrics = calculate_metrics(test_labels, test_pred)

    # --- Cleanup temp checkpoint ---
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(
        "AAE Run %d done: pr_auc=%.4f, f1=%.4f, roc_auc=%.4f, time=%.1fs",
        run_id,
        auc_scores["pr_auc"],
        test_metrics["f1"],
        auc_scores["roc_auc"],
        train_time_sec,
    )

    # --- Return result row (matches CSV format §4.8.8) ---
    result: dict[str, Any] = {
        "run_id": run_id,
        "reconstruction_weight": round(reconstruction_weight, 6),
        "adversarial_weight": round(adversarial_weight, 6),
        "discriminator_hidden_layers": str(list(disc_hidden)),
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
