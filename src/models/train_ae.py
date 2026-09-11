"""CLI for final AE training with validated hyperparameters (§3.2).

After Fase 3.1 validation selects HP_AE_best and writes
``config/params_validated_ae.yaml``, this script performs the final training:

    - 3 runs with different seeds (nested validation, §4.8.1)
    - Saves best checkpoint to reports/checkpoints/ae_baseline.pth
    - Writes reports/tables/ae_final_metrics.csv with mean ± std

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
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import KukaDataset
from src.models.autoencoder import SequenceAutoencoder, fit_autoencoder
from src.utils.config import load_config, get_param
from src.utils.metrics import (
    calculate_auc,
    calculate_metrics,
    compute_anomaly_scores,
    percentile_threshold,
)

logger = logging.getLogger(__name__)

# Nested validation: 3 runs with different seeds (§3.2)
DEFAULT_SEEDS = [42, 123, 7]


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loader(
    processed_dir: Path,
    filename: str,
    window_size: int,
    batch_size: int,
    label: int,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader from a processed .npy file."""
    data = np.load(processed_dir / filename)
    ds = KukaDataset(data, window_size=window_size, label=label)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False
    )


def _compute_errors(
    model: SequenceAutoencoder,
    loader: DataLoader,
    device: torch.device,
    metric: str = "mse",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction errors and labels."""
    model.eval()
    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            y = batch[1] if isinstance(batch, (list, tuple)) else None

            errors = model.compute_reconstruction_error(x, reduction="sample", metric=metric)
            all_errors.append(errors.cpu().numpy())
            if y is not None:
                all_labels.append(y.cpu().numpy().flatten())

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return errors, labels


def train_single_seed(
    config: dict[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a single AE with a given seed and evaluate on test set.

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
        best_epoch, threshold}
    """
    _set_seed(seed)

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
    reports_dir = Path(get_param(config, "paths.reports", "reports/"))

    logger.info("Training AE (seed=%d) on %s, epochs=%d, patience=%d", seed, dev, epochs, patience)

    # --- Build dataloaders ---
    train_loader = _build_loader(processed_dir, "train.npy", window_size, batch_size, label=0, shuffle=True)
    val_loader = _build_loader(processed_dir, "val.npy", window_size, batch_size, label=0, shuffle=False)

    # Test loader: normal + anomaly
    test_normal_loader = _build_loader(processed_dir, "test_normal.npy", window_size, batch_size, label=0)
    test_anomaly_loader = _build_loader(processed_dir, "test_anomaly.npy", window_size, batch_size, label=1)

    # --- Instantiate model ---
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", total_params, trainable_params)

    # --- Train ---
    start_time = time.time()
    history = fit_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device=dev,
    )
    train_time_sec = time.time() - start_time
    best_epoch = len(history["val_loss"]) if history else 0

    # --- Calibrate threshold on validation (normal only) ---
    val_errors, _ = _compute_errors(model, val_loader, dev)
    threshold = percentile_threshold(val_errors, 99.0)

    # --- Evaluate on test (normal + anomaly) ---
    test_normal_errors, _ = _compute_errors(model, test_normal_loader, dev)
    test_anomaly_errors, _ = _compute_errors(model, test_anomaly_loader, dev)
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
        "n_train_windows": len(train_loader.dataset),
        "n_val_windows": len(val_loader.dataset),
        "n_test_normal_windows": len(test_normal_loader.dataset),
        "n_test_anomaly_windows": len(test_anomaly_loader.dataset),
    }


def run_final_training(
    config_path: str | Path,
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
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
    """
    from src.utils.config import load_config

    if seeds is None:
        seeds = DEFAULT_SEEDS

    config = load_config(
        config_path=Path("config/config.yaml"),
        params_path=Path(config_path) if isinstance(config_path, str) else config_path,
    )

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
    enc_channels = list(get_param(config, "model.encoder.conv_channels", [128, 64]))

    logger.info("=" * 60)
    logger.info("Fase 3.2 — Final AE Training")
    logger.info("HP_AE_best: W=%d, latent_dim=%d, encoder_channels=%s",
                window_size, latent_dim, enc_channels)
    logger.info("Nested validation: %d seeds %s", len(seeds), seeds)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        # Save checkpoint only for the first seed (the baseline)
        save_ckpt = str(checkpoint_path) if i == 0 else None

        result = train_single_seed(config, seed, checkpoint_path=save_ckpt, device=device)
        all_results.append(result)

    # --- Write metrics CSV ---
    csv_columns = [
        "seed", "train_time_sec", "best_epoch", "test_pr_auc", "test_roc_auc",
        "test_f1", "test_accuracy", "test_precision", "test_recall", "threshold",
        "n_train_windows", "n_val_windows", "n_test_normal_windows", "n_test_anomaly_windows",
    ]

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in all_results:
            writer.writerow({col: row.get(col, "") for col in csv_columns})

    # --- Compute mean ± std ---
    pr_aucs = [r["test_pr_auc"] for r in all_results]
    f1s = [r["test_f1"] for r in all_results]
    roc_aucs = [r["test_roc_auc"] for r in all_results]

    logger.info("=" * 60)
    logger.info("FINAL METRICS (mean ± std over %d seeds)", len(seeds))
    logger.info("  PR-AUC:  %.4f ± %.4f", float(np.mean(pr_aucs)), float(np.std(pr_aucs)))
    logger.info("  ROC-AUC: %.4f ± %.4f", float(np.mean(roc_aucs)), float(np.std(roc_aucs)))
    logger.info("  F1:      %.4f ± %.4f", float(np.mean(f1s)), float(np.std(f1s)))
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Metrics:    %s", metrics_path)

    # Append summary row to CSV
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["seed", "train_time_sec", "best_epoch", "test_pr_auc",
                         "test_roc_auc", "test_f1", "test_accuracy",
                         "test_precision", "test_recall", "threshold",
                         "n_train_windows", "n_val_windows",
                         "n_test_normal_windows", "n_test_anomaly_windows"])
        # Write summary row
        summary_row = {
            "seed": "MEAN±STD",
            "train_time_sec": f"{float(np.mean([r['train_time_sec'] for r in all_results])):.2f} ± {float(np.std([r['train_time_sec'] for r in all_results])):.2f}",
            "best_epoch": f"{float(np.mean([r['best_epoch'] for r in all_results])):.1f} ± {float(np.std([r['best_epoch'] for r in all_results])):.1f}",
            "test_pr_auc": f"{float(np.mean(pr_aucs)):.4f} ± {float(np.std(pr_aucs)):.4f}",
            "test_roc_auc": f"{float(np.mean(roc_aucs)):.4f} ± {float(np.std(roc_aucs)):.4f}",
            "test_f1": f"{float(np.mean(f1s)):.4f} ± {float(np.std(f1s)):.4f}",
        }
        writer.writerow([summary_row.get(col, "") for col in csv_columns])


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
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
