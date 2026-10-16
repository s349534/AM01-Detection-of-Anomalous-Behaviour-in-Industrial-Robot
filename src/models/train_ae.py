"""CLI for final AE training with validated hyperparameters (§3.2).

After Fase 3.1 validation selects HP_AE_best and writes
``config/params_validated_ae.yaml``, this script performs the final training:

    - 3 runs with different seeds (nested validation, §4.8.1)
    - Saves best checkpoint to reports/checkpoints/ae_baseline.pth
    - Writes reports/tables/ae_final_metrics.csv with mean +/- std

Threshold (soglia):
    La soglia NON viene più calcolata nello Stage 2 (select_threshold.py).
    Viene calcolata **dopo l'addestramento** di ogni modello, su dati di
    validazione (val_normal + val_anomaly) che il modello non ha mai visto
    durante il training. Questo garantisce che la soglia corrisponda al modello
    che effettivamente farà inferenza.

    Ogni seed produce un modello con pesi diversi → soglia F1-ottimale diversa.

Usage:
    python -m src.models.train_ae --config config/params_validated_ae.yaml
    python -m src.models.train_ae --config config/params_validated_ae.yaml --seeds 42 123 7
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
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
from src.utils.config import get_param, load_config
from src.utils.metrics import (
    calculate_auc,
    calculate_metrics,
    compute_anomaly_scores,
    find_optimal_threshold,
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Nested validation: 3 runs with different seeds (§3.2)
DEFAULT_SEEDS = [42, 123, 7]


def train_single_seed(
    config: dict[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a single AE with a given seed and evaluate on test set.

    The decision threshold is **calibrated in-place** on this model's own
    validation errors (val_normal + val_anomaly), not inherited from a
    prior Stage-2 calibration. This guarantees the threshold matches the
    model that will perform inference.

    Parameters
    ----------
    config : dict
        Validated config (from params_validated_ae.yaml).
    seed : int
        Random seed for this run.
    checkpoint_path : str | Path | None
        Where to save the model checkpoint.
    device : str | torch.device | None
        Device to train on.

    Returns
    -------
    dict
        Metrics dict for this run: {seed, train_time_sec, test_pr_auc,
        test_roc_auc, test_f1, test_accuracy, test_precision, test_recall,
        best_epoch, threshold, val_threshold}
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    window_size = int(get_param(config, "model.window_size", 16))
    batch_size = int(get_param(config, "training.batch_size", 256))
    epochs = int(get_param(config, "training.epochs", 100))
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 0.0))
    patience = int(get_param(config, "training.early_stopping.patience", 10))
    min_delta = float(get_param(config, "training.early_stopping.min_delta", 1e-4))
    processed_dir = Path(get_param(config, "paths.data_processed", "data/processed/"))

    logger.info("Training AE (seed=%d) on %s, epochs=%d, patience=%d", seed, dev, epochs, patience)

    # --- Build dataloaders via shared build_loader ---
    train_data = np.load(processed_dir / "train.npy")
    val_normal_data = np.load(processed_dir / "val_normal.npy")
    val_anomaly_data = np.load(processed_dir / "val_anomaly.npy")
    test_normal_data = np.load(processed_dir / "test_normal.npy")
    test_anomaly_data = np.load(processed_dir / "test_anomaly.npy")

    train_loader = build_loader(train_data, window_size=window_size,
                                batch_size=batch_size, label=0, shuffle=True)
    val_normal_loader = build_loader(val_normal_data, window_size=window_size,
                                     batch_size=batch_size, label=0)
    val_anomaly_loader = build_loader(val_anomaly_data, window_size=window_size,
                                      batch_size=batch_size, label=1)
    test_normal_loader = build_loader(test_normal_data, window_size=window_size,
                                      batch_size=batch_size, label=0)
    test_anomaly_loader = build_loader(test_anomaly_data, window_size=window_size,
                                       batch_size=batch_size, label=1)

    # --- Instantiate model ---
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", total_params, trainable_params)

    # --- Read loss config ---
    loss_name = get_param(config, "training.loss", "mae")

    # --- Train ---
    start_time = time.time()
    history = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_normal_loader,   # early stopping on val_normal only
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=dev,
        loss=loss_name,
    )
    train_time_sec = time.time() - start_time
    best_epoch = history.get("best_epoch", 0) if history else 0

    # --- Calibrate threshold on this model's OWN validation errors ---
    # (val_normal + val_anomaly) — NOT on Stage-1 search model errors.
    # This guarantees the threshold matches the model that will do inference.
    val_normal_errors = compute_errors_on_loader(model, val_normal_loader, dev, metric=loss_name)
    val_anomaly_errors = compute_errors_on_loader(model, val_anomaly_loader, dev, metric=loss_name)
    val_errors = np.concatenate([val_normal_errors, val_anomaly_errors])
    val_labels = np.concatenate([
        np.zeros(len(val_normal_errors), dtype=np.int32),
        np.ones(len(val_anomaly_errors), dtype=np.int32),
    ])

    threshold, val_metrics = find_optimal_threshold(val_labels, val_errors)
    logger.info(
        "Seed %d: calibrated threshold=%.6f (val F1=%.4f, precision=%.4f, recall=%.4f, "
        "val_normal_samples=%d, val_anomaly_samples=%d)",
        seed, threshold, val_metrics["f1"], val_metrics["precision"], val_metrics["recall"],
        len(val_normal_errors), len(val_anomaly_errors),
    )

    # --- Evaluate on test (normal + anomaly) ---
    test_normal_errors = compute_errors_on_loader(model, test_normal_loader, dev)
    test_anomaly_errors = compute_errors_on_loader(model, test_anomaly_loader, dev)
    test_errors = np.concatenate([test_normal_errors, test_anomaly_errors])
    test_labels = np.concatenate([
        np.zeros(len(test_normal_errors), dtype=np.int32),
        np.ones(len(test_anomaly_errors), dtype=np.int32),
    ])

    auc_scores = calculate_auc(test_labels, test_errors)
    test_pred = compute_anomaly_scores(test_errors, threshold)
    test_metrics = calculate_metrics(test_labels, test_pred)

    logger.info(
        "Seed %d: train_time=%.1fs, best_epoch=%d, "
        "PR-AUC=%.4f, ROC-AUC=%.4f, F1=%.4f, threshold=%.6f",
        seed, train_time_sec, best_epoch,
        auc_scores["pr_auc"], auc_scores["roc_auc"], test_metrics["f1"], threshold,
    )

    # --- Save checkpoint (only for first seed / baseline) ---
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        # Load existing checkpoint to get model state and history
        if ckpt.exists():
            existing = torch.load(ckpt, map_location="cpu")
            model_state = existing.get("model_state_dict", model.state_dict())
            history_dict = existing.get("history", history)
        else:
            model_state = model.state_dict()
            history_dict = history

        # Save checkpoint (no channel stats — using classic MAE scoring)
        torch.save(
            {
                "model_state_dict": model_state,
                "history": history_dict,
                "input_dim": model.input_dim,
                "window_size": model.window_size,
                "latent_dim": model.latent_dim,
            },
            ckpt,
        )
        logger.info("Saved checkpoint to %s", ckpt)

    return {
        "seed": seed,
        "train_time_sec": round(train_time_sec, 2),
        "best_epoch": best_epoch,
        "test_pr_auc": round(auc_scores["pr_auc"], 6),
        "test_roc_auc": round(auc_scores["roc_auc"], 6),
        "test_f1": round(test_metrics["f1"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
        "threshold": round(float(threshold), 6),
        "val_threshold": round(float(threshold), 6),
        "val_f1": round(val_metrics["f1"], 6),
        "n_train_windows": len(train_loader.dataset),  # type: ignore[arg-type]
        "n_val_normal_windows": len(val_normal_loader.dataset),  # type: ignore[arg-type]
        "n_val_anomaly_windows": len(val_anomaly_loader.dataset),  # type: ignore[arg-type]
        "n_test_normal_windows": len(test_normal_loader.dataset),  # type: ignore[arg-type]
        "n_test_anomaly_windows": len(test_anomaly_loader.dataset),  # type: ignore[arg-type]
    }


def run_final_training(
    config_path: str | Path,
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
) -> None:
    """Run final AE training with nested validation across multiple seeds.

    Parameters
    ----------
    config_path : str | Path
        Path to ``params_validated_ae.yaml``.
    seeds : list[int] or None
        Seeds for nested validation runs. Defaults to ``[42, 123, 7]``.
    device : str | torch.device | None
        Device to train on.
    max_epochs : int or None
        Override ``training.epochs`` from config (useful for quick smoke tests).
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    config = load_config(
        config_path=Path("config/config.yaml"),
        params_path=Path(config_path) if isinstance(config_path, str) else config_path,
    )

    if max_epochs is not None:
        config["training"]["epochs"] = max_epochs
        logger.info("Overriding training.epochs -> %d (max-epochs)", max_epochs)

    reports_dir = Path(get_param(config, "paths.reports", "reports/"))
    ckpt_dir = reports_dir / "checkpoints"
    tables_dir = reports_dir / "tables"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ckpt_dir / "ae_baseline.pth"
    metrics_path = tables_dir / "ae_final_metrics.csv"

    # Read model HP for logging
    window_size = int(get_param(config, "model.window_size", 16))
    latent_dim = int(get_param(config, "model.latent_dim", 16))
    enc_channels = list(get_param(config, "model.encoder.conv_channels", [64, 32, 16]))

    # Threshold is calibrated per-seed inside train_single_seed (Stage 3),
    # not inherited from Stage 2 (select_threshold.py).

    logger.info("=" * 60)
    logger.info("Fase 3.2 -- Final AE Training")
    logger.info("HP_AE_best: W=%d, latent_dim=%d, encoder_channels=%s",
                window_size, latent_dim, enc_channels)
    logger.info("Threshold: calibrated per-seed after training (not from Stage 2)")
    logger.info("Nested validation: %d seeds %s", len(seeds), seeds)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        # Save checkpoint only for the first seed (the baseline)
        save_ckpt = str(checkpoint_path) if i == 0 else None

        result = train_single_seed(config, seed,
                                   checkpoint_path=save_ckpt, device=device)
        all_results.append(result)

    # --- Write metrics CSV ---
    csv_columns = [
        "seed", "train_time_sec", "best_epoch", "test_pr_auc", "test_roc_auc",
        "test_f1", "test_accuracy", "test_precision", "test_recall", "threshold",
        "val_f1",
        "n_train_windows", "n_val_normal_windows", "n_val_anomaly_windows",
        "n_test_normal_windows", "n_test_anomaly_windows",
    ]

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in all_results:
            writer.writerow({col: row.get(col, "") for col in csv_columns})

    # --- Compute mean +/- std ---
    pr_aucs = [r["test_pr_auc"] for r in all_results]
    f1s = [r["test_f1"] for r in all_results]
    roc_aucs = [r["test_roc_auc"] for r in all_results]

    logger.info("=" * 60)
    logger.info("FINAL METRICS (mean +/- std over %d seeds)", len(seeds))
    logger.info("  PR-AUC:  %.4f +/- %.4f", float(np.mean(pr_aucs)), float(np.std(pr_aucs)))
    logger.info("  ROC-AUC: %.4f +/- %.4f", float(np.mean(roc_aucs)), float(np.std(roc_aucs)))
    logger.info("  F1:      %.4f +/- %.4f", float(np.mean(f1s)), float(np.std(f1s)))
    logger.info("  Threshold (calibrated per-seed):")
    for r in all_results:
        logger.info("    seed %d: threshold=%.6f (val_f1=%.4f)", r["seed"], r["threshold"], r["val_f1"])
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Metrics:    %s", metrics_path)

    # Append summary row to CSV
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([])
        csv_writer.writerow(csv_columns)
        # Write summary row
        thresholds = [r["threshold"] for r in all_results]
        val_f1s = [r["val_f1"] for r in all_results]
        summary_row: dict[str, str] = {
            "seed": "MEAN+/-STD",
            "train_time_sec": f"{float(np.mean([r['train_time_sec'] for r in all_results])):.2f} +/- {float(np.std([r['train_time_sec'] for r in all_results])):.2f}",
            "best_epoch": f"{float(np.mean([r['best_epoch'] for r in all_results])):.1f} +/- {float(np.std([r['best_epoch'] for r in all_results])):.1f}",
            "test_pr_auc": f"{float(np.mean(pr_aucs)):.4f} +/- {float(np.std(pr_aucs)):.4f}",
            "test_roc_auc": f"{float(np.mean(roc_aucs)):.4f} +/- {float(np.std(roc_aucs)):.4f}",
            "test_f1": f"{float(np.mean(f1s)):.4f} +/- {float(np.std(f1s)):.4f}",
            "threshold": f"{float(np.mean(thresholds)):.6f} +/- {float(np.std(thresholds)):.6f}",
            "val_f1": f"{float(np.mean(val_f1s)):.4f} +/- {float(np.std(val_f1s)):.4f}",
        }
        csv_writer.writerow([summary_row.get(col, "") for col in csv_columns])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final AE training with validated hyperparameters (§3.2)"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/params_validated_ae.yaml"),
        help="Path to params_validated_ae.yaml (default: config/params_validated_ae.yaml)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds for nested validation (default: 42 123 7)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu), auto-detect if not specified",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override training.epochs (useful for quick smoke tests on HPC)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    run_final_training(
        config_path=args.config,
        seeds=args.seeds,
        device=device,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
