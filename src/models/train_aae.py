"""CLI for final AAE training with validated hyperparameters (§4.2).

After Fase 4.1 validation selects HP_AAE_best and writes
``config/params_validated_aae.yaml``, this script performs the final training:

    - 3 runs with different seeds (nested validation, §4.8.1)
    - Calibrates decision threshold on validation set
    - Evaluates on the held-out test set
    - Saves best checkpoint to reports/checkpoints/aae_final.pth
    - Writes reports/tables/aae_final_metrics.csv with mean +/- std

Usage:
    python -m src.models.train_aae --config config/params_validated_aae.yaml
    python -m src.models.train_aae --config config/params_validated_aae.yaml --seeds 42 123 7
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import build_dataloaders
from src.models.autoencoder import compute_errors_on_loader
from src.models.adversarial_ae import AdversarialAutoencoder, fit_aae
from src.evaluation.metrics import calculate_auc, evaluate_threshold
from src.evaluation.threshold import calibrate_threshold
from src.utils.config import get_param, load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 7]

FINAL_METRICS_COLUMNS = [
    "seed",
    "train_time_sec",
    "best_epoch",
    "threshold",
    "test_pr_auc",
    "test_roc_auc",
    "test_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_fpr",
    "test_fnr",
    "test_balanced_accuracy",
    "test_mcc",
    "test_cohen_kappa",
    "test_npv",
    "test_jaccard",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "n_train_windows",
    "n_val_normal_windows",
    "n_val_anomaly_windows",
    "n_test_normal_windows",
    "n_test_anomaly_windows",
]


def train_single_seed(
    config: dict[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a single AAE with a given seed and evaluate on test set."""
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    window_size = int(get_param(config, "data.window_size", 256))
    batch_size = int(get_param(config, "training.batch_size", 256))
    epochs = int(get_param(config, "training.epochs", 200))
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 0.0))
    patience = int(get_param(config, "training.early_stopping.patience", 15))
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

    logger.info("Training AAE (seed=%d) on %s, epochs=%d, patience=%d", seed, dev, epochs, patience)

    # --- Build dataloaders ---
    loaders = build_dataloaders(
        config,
        partitions=["train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"],
        batch_size=batch_size,
    )

    train_loader = loaders["train"]
    val_normal_loader = loaders["val_normal"]
    val_anomaly_loader = loaders["val_anomaly"]
    test_normal_loader = loaders["test_normal"]
    test_anomaly_loader = loaders["test_anomaly"]

    # --- Instantiate model ---
    model = AdversarialAutoencoder.from_config(config)
    model.to(dev)

    # --- Train ---
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
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=dev,
        metric=metric,
    )
    train_time_sec = time.time() - start_time
    best_epoch = len(history["val_loss"]) if history else 0

    # --- Validation AUC for model selection ---
    val_normal_errors = compute_errors_on_loader(model, val_normal_loader, dev, metric=metric)
    val_anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, dev, metric=metric)
    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors), dtype=np.int32),
        np.ones(len(val_anomaly_errors), dtype=np.int32),
    ])
    val_auc = calculate_auc(val_labels, val_errors)
    val_roc_auc = float(val_auc["roc_auc"])
    val_pr_auc = float(val_auc["pr_auc"])

    # --- Calibrate threshold on validation and evaluate on TEST set ---
    threshold, _ = calibrate_threshold(
        model, val_normal_loader, val_anomaly_loader, dev, metric=metric, method="f1_optimal"
    )

    test_normal_errors = compute_errors_on_loader(model, test_normal_loader, dev, metric=metric)
    test_anomaly_errors = compute_errors_on_loader(model, test_anomaly_loader, dev, metric=metric)
    test_errors = np.concatenate([test_normal_errors, test_anomaly_errors])
    test_labels = np.concatenate([
        np.zeros(len(test_normal_errors), dtype=np.int32),
        np.ones(len(test_anomaly_errors), dtype=np.int32),
    ])

    test_metrics = evaluate_threshold(test_labels, test_errors, threshold)
    cm = test_metrics["confusion_matrix"]

    logger.info(
        "AAE Seed %d: ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, threshold=%.6f [TN=%d FP=%d FN=%d TP=%d]",
        seed, test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"],
        threshold, cm["tn"], cm["fp"], cm["fn"], cm["tp"],
    )

    result_dict = {
        "seed": seed,
        "train_time_sec": round(train_time_sec, 2),
        "best_epoch": best_epoch,
        "threshold": round(float(threshold), 6),
        "test_pr_auc": round(test_metrics["pr_auc"], 6),
        "test_roc_auc": round(test_metrics["roc_auc"], 6),
        "test_f1": round(test_metrics["f1"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
        "test_specificity": round(test_metrics["specificity"], 6),
        "test_fpr": round(test_metrics["false_positive_rate"], 6),
        "test_fnr": round(test_metrics["false_negative_rate"], 6),
        "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 6),
        "test_mcc": round(test_metrics["mcc"], 6),
        "test_cohen_kappa": round(test_metrics["cohen_kappa"], 6),
        "test_npv": round(test_metrics["npv"], 6),
        "test_jaccard": round(test_metrics["jaccard"], 6),
        "test_tn": cm["tn"],
        "test_fp": cm["fp"],
        "test_fn": cm["fn"],
        "test_tp": cm["tp"],
        "n_train_windows": len(train_loader.dataset),
        "n_val_normal_windows": len(val_normal_loader.dataset),
        "n_val_anomaly_windows": len(val_anomaly_loader.dataset),
        "n_test_normal_windows": len(test_normal_loader.dataset),
        "n_test_anomaly_windows": len(test_anomaly_loader.dataset),
        "_val_roc_auc": val_roc_auc,
        "_val_pr_auc": val_pr_auc,
    }

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "config": config,
                "seed": seed,
                "threshold": threshold,
            },
            ckpt,
        )

    return result_dict


def run_final_training(
    config_path: str | Path = "config/params_validated_aae.yaml",
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
) -> None:
    """Run final AAE training across multiple seeds and record final metrics."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        cfg_file = Path("config/params_validated_aae.yaml")
        if not cfg_file.exists():
            cfg_file = Path("config/params.yaml")

    config = load_config(config_path=Path("config/config.yaml"), params_path=cfg_file)

    reports_dir = Path(get_param(config, "paths.reports_tables", "reports/tables/"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "aae_final_metrics.csv"

    ckpt_dir = Path(get_param(config, "paths.reports_checkpoints", "reports/checkpoints/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best_val_auc = -1.0
    best_seed = seeds[0]

    for seed in seeds:
        seed_ckpt = ckpt_dir / f"aae_final_seed{seed}.pth"
        res = train_single_seed(config, seed=seed, checkpoint_path=seed_ckpt, device=device)
        results.append(res)

        if res["_val_roc_auc"] > best_val_auc:
            best_val_auc = res["_val_roc_auc"]
            best_seed = seed
            final_ckpt = ckpt_dir / "aae_final.pth"
            shutil.copy2(seed_ckpt, final_ckpt)
            logger.info("New best AAE model (seed=%d, val_roc_auc=%.4f) -> %s", seed, best_val_auc, final_ckpt)

    # Write metrics CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_METRICS_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in FINAL_METRICS_COLUMNS if k in r}
            writer.writerow(row)

    logger.info("AAE Final metrics saved to %s", csv_path)


run_final_training_aae = run_final_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final AAE models across multiple seeds")
    parser.add_argument("--config", type=Path, default=Path("config/params_validated_aae.yaml"), help="Config path")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    run_final_training(config_path=args.config, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
