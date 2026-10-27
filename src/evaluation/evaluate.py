"""High-level evaluation pipeline for trained models.

This is the main entry point for evaluating a trained model on test/validation data.
Can be used as a library function or via CLI (python -m src.evaluation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.autoencoder import SequenceAutoencoder
from src.data.dataset import build_dataloaders
from src.evaluation.threshold import calibrate_threshold
from src.evaluation.metrics import evaluate_threshold
from src.utils.config import get_param, load_config

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
                    test_precision, test_recall, threshold, confusion_matrix,
                    confusion_matrix_array
    """
    from src.models.autoencoder import compute_errors_on_loader

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
    config: dict[str, Any],
    threshold: float | None,
    device: torch.device,
    metric: str = "mae",
    calibration_method: str = "f1_optimal",
) -> dict[str, Any]:
    """Evaluate model on all splits (val + test) using build_dataloaders.

    Loads data via build_dataloaders, calibrates threshold on validation,
    then evaluates on test. Returns comprehensive metrics.
    """
    # Build loaders for val and test partitions
    loaders = build_dataloaders(
        config,
        partitions=["val_normal", "val_anomaly", "test_normal", "test_anomaly"],
    )

    val_normal_loader = loaders["val_normal"]
    val_anomaly_loader = loaders["val_anomaly"]
    test_normal_loader = loaders["test_normal"]
    test_anomaly_loader = loaders["test_anomaly"]

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
    config: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    threshold: float | None = None,
    metric: str = "mae",
    calibration_method: str = "f1_optimal",
) -> dict[str, Any]:
    """Full evaluation pipeline from checkpoint.

    Loads model from checkpoint, builds data loaders via build_dataloaders,
    calibrates threshold on validation (or uses provided), evaluates on test.

    This is the library API entry point. The checkpoint's architecture metadata
    (model_config) is used to build the model, ensuring no mismatch.
    """
    if config is None:
        config = load_config()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=dev)

    # Extract architecture from checkpoint (model_config takes precedence)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # Legacy checkpoint without architecture metadata
        raise RuntimeError(
            "Checkpoint does not contain 'model_config' metadata. "
            "Please retrain the model with the current codebase (which saves "
            "architecture metadata) or use a compatible checkpoint."
        )

    # Build effective config: use checkpoint's architecture for model,
    # and config for data/partitioning/training params
    effective_config = {
        **config,
        "data": {**config.get("data", {}), "window_size": model_config["window_size"]},
        "model": {
            **config.get("model", {}),
            "window_size": model_config["window_size"],
            "latent_dim": model_config["latent_dim"],
            "input_dim": model_config["input_dim"],
            "encoder_channels": tuple(model_config["encoder_channels"]),
            "encoder_kernels": tuple(model_config["encoder_kernels"]),
            "pool_size": model_config["pool_size"],
            "activation": model_config["activation"],
        },
    }

    # Validate checkpoint window_size matches config (warn, don't fail)
    config_window_size = int(get_param(config, "data.window_size", 256))
    checkpoint_window_size = model_config["window_size"]
    if checkpoint_window_size != config_window_size:
        logger.warning(
            "Checkpoint window_size (%d) != config data.window_size (%d). "
            "Using checkpoint's window_size for model and loaders.",
            checkpoint_window_size, config_window_size
        )

    # Verify manifest if present
    manifest_path = Path(get_param(config, "paths.data_processed", "data/processed/")) / "manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        m_window = manifest.get("window_size")
        if m_window and m_window != checkpoint_window_size:
            logger.warning(
                "Manifest window_size (%d) != checkpoint window_size (%d). "
                "Partition plan may differ from what was used during training.",
                m_window, checkpoint_window_size
            )
        # Check partitioning parameters
        part_cfg = config.get("data", {}).get("partitioning", {})
        manifest_part = manifest.get("partitioning", {})
        if manifest_part:
            for key in ("chunk_size", "gap", "quotas", "stride_multipliers"):
                if key in manifest_part and key in part_cfg:
                    if manifest_part[key] != part_cfg[key]:
                        logger.warning(
                            "Manifest partitioning '%s' (%s) != config (%s). "
                            "The evaluation may use a different partition plan.",
                            key, manifest_part[key], part_cfg[key]
                        )

    # Build model from checkpoint's architecture
    model = SequenceAutoencoder.from_architecture(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return evaluate_on_splits(
        model=model,
        config=effective_config,
        threshold=threshold,
        device=dev,
        metric=metric,
        calibration_method=calibration_method,
    )


def evaluate_checkpoints(
    checkpoint_paths: list[str | Path],
    config: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    threshold: float | None = None,
    metric: str = "mae",
    calibration_method: str = "f1_optimal",
) -> dict[str, Any]:
    """Evaluate multiple checkpoints and aggregate results (mean +/- std).

    Parameters
    ----------
    checkpoint_paths : list[str | Path]
        Paths to checkpoint files.
    config : dict | None
        Configuration dict (merged config + params).
    device : str | torch.device | None
        Device for inference.
    threshold : float | None
        Fixed threshold (if None, calibrated on validation).
    metric : str
        Reconstruction metric (mae/mse).
    calibration_method : str
        Threshold calibration method.

    Returns
    -------
    dict with keys:
        - "per_seed": list of per-seed result dicts
        - "aggregate": dict with mean/std for each metric
    """
    from src.evaluation.metrics import aggregate_metrics

    if config is None:
        config = load_config()

    per_seed_results = []

    for ckpt_path in checkpoint_paths:
        result = run_evaluation(
            checkpoint_path=ckpt_path,
            config=config,
            device=device,
            threshold=threshold,
            metric=metric,
            calibration_method=calibration_method,
        )
        # Add seed info from checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        seed = ckpt.get("seed", "unknown")
        per_seed_results.append({
            "seed": seed,
            **result["test_metrics"],
        })

    # Aggregate metrics
    aggregate = aggregate_metrics(per_seed_results)

    return {
        "per_seed": per_seed_results,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    # This module is not intended to be run directly.
    # Use `python -m src.evaluation` instead (see src/evaluation/cli.py).
    pass