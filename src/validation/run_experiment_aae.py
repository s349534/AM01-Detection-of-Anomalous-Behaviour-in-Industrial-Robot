"""Single-run experiment: train + evaluate an AAE (project_plan.md §4.1).

This module provides :func:`train_and_evaluate_aae`, the atomic unit invoked by
``run_search_aae.py`` for each sampled hyperparameter configuration.

Workflow per run (Stage 1 - Model Selection):
    1. Set seed for reproducibility
    2. Build DataLoaders from processed artifacts via build_dataloaders (train, val_normal, val_anomaly)
    3. Instantiate ``AdversarialAutoencoder`` from config
    4. Train with early stopping on validation normal reconstruction loss
    5. Compute reconstruction errors on val_normal + val_anomaly (HP selection set)
    6. Compute ROC-AUC and PR-AUC on validation (threshold-free, no binarization)
    7. Return dict with val_roc_auc, val_pr_auc, etc. (no test leakage)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import build_dataloaders
from src.models.inference import compute_errors_on_loader
from src.models.adversarial_ae import AdversarialAutoencoder, fit_aae
from src.utils.config import get_param
from src.utils.device import check_cuda_available
from src.evaluation.metrics import calculate_auc
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

def train_and_evaluate_aae(
    config: dict[str, Any],
    run_id: int = 0,
    seed: int = 42,
    max_val_epochs: int | None = None,
    device: str | torch.device | None = None,
    patience: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train a single AAE configuration and evaluate on validation (Stage 1).

    Parameters
    ----------
    config : dict
        Full config dict. Must contain data.window_size, training.batch_size, etc.
    run_id : int
        Identifier for this search run (written to CSV).
    seed : int
        Random seed for reproducibility.
    max_val_epochs : int or None
        Max epochs for this validation run. Overrides config ``training.validation_epochs``
        if provided. If not provided, reads from ``config["training"]["validation_epochs"]``.
    device : str | torch.device | None
        Device to train/evaluate on. None -> auto-detect.
    patience : int or None
        Early Stopping patience. If None, reads from config.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    dict
        Result row matching CSV format.
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Early GPU check: fail fast if CUDA not actually usable ---
    if dev.type == "cuda":
        try:
            check_cuda_available(dev)
        except RuntimeError as e:
            logger.error("GPU check failed in run_id=%d: %s", run_id, e)
            raise

    # --- Extract hyperparameters ---
    window_size = int(get_param(config, "data.window_size", 256))
    latent_dim = int(get_param(config, "model.latent_dim", 16))
    # Read discriminator hidden layers from aae.discriminator (new unified config)
    # with fallback to model.discriminator (legacy)
    disc_hidden = tuple(
        get_param(config, "aae.discriminator.hidden_layers", get_param(config, "model.discriminator.hidden_layers", [32, 16]))
    )
    batch_size = int(get_param(config, "training.batch_size", 256))
    # Use validation_epochs from config; CLI max_val_epochs overrides if provided
    val_epochs_cfg = get_param(config, "training.validation_epochs", None)
    if val_epochs_cfg is not None:
        epochs = int(val_epochs_cfg)
    elif max_val_epochs is not None:
        epochs = int(max_val_epochs)
    else:
        raise ValueError("training.validation_epochs not set in config and --max-val-epochs not provided")
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 0.0))
    patience = patience if patience is not None else int(
        get_param(config, "training.early_stopping.patience", 10)
    )
    min_delta = float(get_param(config, "training.early_stopping.min_delta", 1e-4))
    reconstruction_weight = float(
        get_param(config, "training.reconstruction_weight", 1.0)
    )
    adversarial_weight = float(
        get_param(config, "training.adversarial_weight", 0.1)
    )
    disc_updates_per_gen = int(
        get_param(config, "training.discriminator_updates_per_gen", 1)
    )
    loss_name = str(get_param(config, "training.loss", "mae"))
    metric = str(get_param(config, "training.reconstruction_metric", "mae"))

    logger.info(
        "AAE Run %d (seed=%d): W=%d, latent_dim=%d, disc_hidden=%s, recon_w=%.2f, adv_w=%.3f, epochs=%d",
        run_id, seed, window_size, latent_dim, list(disc_hidden),
        reconstruction_weight, adversarial_weight, epochs,
    )

    # --- Build dataloaders (train, val_normal, val_anomaly) ---
    loaders = build_dataloaders(
        config,
        partitions=["train", "val_normal", "val_anomaly"],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_loader = loaders["train"]
    val_normal_loader = loaders["val_normal"]
    val_anomaly_loader = loaders["val_anomaly"]

    # --- Instantiate model ---
    model = AdversarialAutoencoder.from_config(config)
    model.to(dev)

    # --- Train (early stopping on val_normal) ---
    start_time = time.time()
    history = fit_aae(
        model=model,
        train_loader=train_loader,
        val_loader=val_normal_loader,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        reconstruction_weight=reconstruction_weight,
        adversarial_weight=adversarial_weight,
        discriminator_updates_per_gen=disc_updates_per_gen,
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=None,
        device=dev,
        metric=metric,
    )
    train_time_sec = time.time() - start_time
    best_epoch = len(history["val_loss"]) if history else 0

    # --- Compute validation AUC (val_normal + val_anomaly) ---
    val_normal_errors = compute_errors_on_loader(model, val_normal_loader, dev, metric=metric)
    val_anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, dev, metric=metric)
    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors), dtype=np.int32),
        np.ones(len(val_anomaly_errors), dtype=np.int32),
    ])

    auc_scores = calculate_auc(val_labels, val_errors)
    val_roc_auc = float(auc_scores["roc_auc"])
    val_pr_auc = float(auc_scores["pr_auc"])

    logger.info(
        "AAE Run %d done: val_pr_auc=%.4f, val_roc_auc=%.4f, time=%.1fs, epoch=%d",
        run_id, val_pr_auc, val_roc_auc, train_time_sec, best_epoch,
    )

    return {
        "run_id": run_id,
        "reconstruction_weight": round(reconstruction_weight, 6),
        "adversarial_weight": round(adversarial_weight, 6),
        "discriminator_hidden_layers": str(list(disc_hidden)),
        "val_roc_auc": round(val_roc_auc, 6),
        "val_pr_auc": round(val_pr_auc, 6),
        "best_val_pr_auc": round(val_pr_auc, 6),
        "best_val_f1": 0.0,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time_sec, 2),
        "seed": seed,
    }
