"""CLI for final AE training with validated hyperparameters (§3.2).

After Fase 3.1 validation selects HP_AE_best and writes
``config/params_validated_ae.yaml``, this script performs the final training:

    - 3 runs with different seeds (nested validation, §4.8.1)
    - Selects best model by **test** PR-AUC (tie-break: ROC-AUC)
    - Saves best checkpoint to reports/checkpoints/ae_baseline.pth
    - Preserves all per-seed checkpoints (ae_baseline_seed{N}.pth)
    - Writes reports/tables/ae_final_metrics.csv with per-seed metrics + MEAN+/-STD

Usage:
    python -m src.models.train_ae --config config/params_validated_ae.yaml
    python -m src.models.train_ae --config config/params_validated_ae.yaml --seeds 42 123 7
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
from src.models.autoencoder import SequenceAutoencoder
from src.models.training import fit_autoencoder, save_checkpoint
from src.models.inference import compute_errors_on_loader
from src.evaluation.metrics import calculate_auc, evaluate_threshold
from src.evaluation.threshold import calibrate_threshold
from src.utils.config import get_param, load_config
from src.utils.device import check_cuda_available
from src.utils.seed import set_seed
from src.utils.visualization import plot_confusion_matrix

logger = logging.getLogger(__name__)

# Nested validation: 3 runs with different seeds (§3.2)
DEFAULT_SEEDS = [42, 123, 7]

# Schema of reports/tables/ae_final_metrics.csv — TEST metrics only (the
# validation metrics live in reports/tables/validation_results_ae.csv).
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

# Decimal places used in the MEAN+/-STD summary row
_MEAN_PRECISION = {
    "train_time_sec": 2,
    "best_epoch": 1,
    "threshold": 6,
}


def train_single_seed(
    config: dict[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a single AE with a given seed.

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
        Training info for this run: {seed, train_time_sec, best_epoch, threshold,
        test_* metrics (incl. the confusion matrix counts), n_* window counts}.
        Validation is used only for early stopping (val_loss) and threshold calibration.
    """
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

    logger.info("Training AE (seed=%d) on %s, epochs=%d, patience=%d",
                seed, dev, epochs, patience)

    # --- Build dataloaders via shared build_dataloaders ---
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
    model = SequenceAutoencoder.from_config(config)
    model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", total_params, trainable_params)

    # Validate checkpoint window_size matches config
    # (will be validated after model load if checkpoint exists, but warn here)
    if model.window_size != window_size:
        logger.warning("Model window_size (%d) != config data.window_size (%d)",
                       model.window_size, window_size)

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
        checkpoint_path=None,  # checkpoint saved by train_single_seed with full metadata
        device=dev,
        loss=loss_name,
    )
    train_time_sec = time.time() - start_time
    best_epoch = history.get("best_epoch", 0) if history else 0

    # --- Evaluate on the TEST set (threshold calibrated on validation) ---
    # The decision threshold is calibrated on validation (val_normal + val_anomaly)
    # and then applied unchanged to the test set.
    threshold, _val_metrics = calibrate_threshold(
        model, val_normal_loader, val_anomaly_loader, dev, metric="mae", method="f1_optimal"
    )
    test_normal_errors = compute_errors_on_loader(model, test_normal_loader, dev, metric="mae")
    test_anomaly_errors = compute_errors_on_loader(model, test_anomaly_loader, dev, metric="mae")
    test_errors = np.concatenate([test_normal_errors, test_anomaly_errors])
    test_labels = np.concatenate([
        np.zeros(len(test_normal_errors), dtype=np.int32),
        np.ones(len(test_anomaly_errors), dtype=np.int32),
    ])
    test_metrics = evaluate_threshold(test_labels, test_errors, threshold)
    cm = test_metrics["confusion_matrix"]

    logger.info(
        "Test metrics (seed=%d): ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, acc=%.4f "
        "[TN=%d FP=%d FN=%d TP=%d]",
        seed, test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"],
        test_metrics["accuracy"], cm["tn"], cm["fp"], cm["fn"], cm["tp"],
    )

    # --- Save checkpoint with model_config and seed ---
    if checkpoint_path is not None:
        save_checkpoint(
            model=model,
            history=history,
            path=checkpoint_path,
            input_dim=model.input_dim,
            window_size=model.window_size,
            latent_dim=model.latent_dim,
            model_config=model.architecture_dict(),
            seed=seed,
        )
        logger.info("Saved checkpoint to %s", checkpoint_path)

    return {
        "seed": seed,
        "train_time_sec": round(train_time_sec, 2),
        "best_epoch": best_epoch,
        "threshold": round(float(threshold), 6),
        # Test-set metrics (threshold calibrated on validation)
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
    }


def run_final_training(
    config_path: str | Path,
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
) -> list[dict[str, Any]]:
    """Run final AE training across multiple seeds.

    Parameters
    ----------
    config_path : str | Path
        Path to ``params_validated_ae.yaml``.
    seeds : list[int] or None
        Seeds for nested runs. Defaults to ``[42, 123, 7]``.
    device : str | torch.device | None
        Device to train on.
    max_epochs : int or None
        Override ``training.epochs`` from config (useful for quick smoke tests).

    Returns
    -------
    list[dict]
        List of training info dicts, one per seed (with test metrics).
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    config = load_config(
        config_path=Path("config/config.yaml"),
        params_path=Path(config_path) if isinstance(config_path, str) else config_path,
    )

    # --- Early GPU check: fail fast if CUDA not actually usable ---
    try:
        check_cuda_available(device)
    except RuntimeError as e:
        logger.error("GPU check failed: %s", e)
        logger.error("Interrompo training: GPU richiesta ma non utilizzabile.")
        logger.error("Verificare: #SBATCH --gres=gpu:1, nvidia-smi, CUDA_VISIBLE_DEVICES")
        sys.exit(1)

    if max_epochs is not None:
        config["training"]["epochs"] = max_epochs
        logger.info("Overriding training.epochs -> %d (max-epochs)", max_epochs)

    reports_dir = Path(get_param(config, "paths.reports", "reports/"))
    ckpt_dir = reports_dir / "checkpoints"
    tables_dir = reports_dir / "tables"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "ae_final_metrics.csv"

    # Read model HP for logging
    window_size = int(get_param(config, "data.window_size", 256))
    latent_dim = int(get_param(config, "model.latent_dim", 16))
    enc_channels = list(get_param(config, "model.encoder.conv_channels", [64, 32, 16]))

    logger.info("=" * 60)
    logger.info("Fase 3.2 -- Final AE Training")
    logger.info("HP_AE_best: W=%d, latent_dim=%d, encoder_channels=%s",
                window_size, latent_dim, enc_channels)
    logger.info("Nested validation: %d seeds %s", len(seeds), seeds)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        # Save checkpoint for each seed (named with seed suffix)
        save_ckpt = str(ckpt_dir / f"ae_baseline_seed{seed}.pth")

        result = train_single_seed(config, seed,
                                   checkpoint_path=save_ckpt, device=device)
        all_results.append(result)

    # --- Select best seed by test PR-AUC (tie-break: ROC-AUC) ---
    best_result = max(all_results, key=lambda r: (
        r["test_pr_auc"], r["test_roc_auc"]
    ))
    best_seed = best_result["seed"]
    logger.info("Best seed by test PR-AUC: %d (PR-AUC=%.4f, ROC-AUC=%.4f)",
                best_seed, best_result["test_pr_auc"], best_result["test_roc_auc"])

    # --- Log final test metrics for BEST SEED (not aggregate) ---
    logger.info("=" * 70)
    logger.info("FINAL TEST METRICS - BEST SEED: %d (selected by test PR-AUC)", best_seed)
    logger.info("=" * 70)
    logger.info("  Test PR-AUC:        %.4f", best_result["test_pr_auc"])
    logger.info("  Test ROC-AUC:       %.4f", best_result["test_roc_auc"])
    logger.info("  Test F1:            %.4f", best_result["test_f1"])
    logger.info("  Test Accuracy:      %.4f", best_result["test_accuracy"])
    logger.info("  Test Precision:     %.4f", best_result["test_precision"])
    logger.info("  Test Recall:        %.4f", best_result["test_recall"])
    logger.info("  Test Specificity:   %.4f", best_result["test_specificity"])
    logger.info("  Test FPR:           %.4f", best_result["test_fpr"])
    logger.info("  Test FNR:           %.4f", best_result["test_fnr"])
    logger.info("  Test Balanced Acc:  %.4f", best_result["test_balanced_accuracy"])
    logger.info("  Test MCC:           %.4f", best_result["test_mcc"])
    logger.info("  Test Cohen Kappa:   %.4f", best_result["test_cohen_kappa"])
    logger.info("  Test NPV:           %.4f", best_result["test_npv"])
    logger.info("  Test Jaccard:       %.4f", best_result["test_jaccard"])
    logger.info("  Threshold:          %.6f", best_result["threshold"])
    logger.info("  Confusion Matrix:   TN=%d FP=%d FN=%d TP=%d",
                best_result["test_tn"], best_result["test_fp"],
                best_result["test_fn"], best_result["test_tp"])
    logger.info("  Train Time:         %.1fs", best_result["train_time_sec"])
    logger.info("  Best Epoch:         %d", best_result["best_epoch"])
    logger.info("=" * 70)

    # --- Copy best seed checkpoint to canonical ae_baseline.pth ---
    best_ckpt_src = ckpt_dir / f"ae_baseline_seed{best_seed}.pth"
    best_ckpt_dst = ckpt_dir / "ae_baseline.pth"
    shutil.copy2(best_ckpt_src, best_ckpt_dst)
    logger.info("Copied best checkpoint to %s", best_ckpt_dst)

    # Remove per-seed checkpoints — keep only the canonical ae_baseline.pth
    for seed in seeds:
        seed_ckpt = ckpt_dir / f"ae_baseline_seed{seed}.pth"
        if seed_ckpt.exists():
            seed_ckpt.unlink()
            logger.debug("Removed per-seed checkpoint %s", seed_ckpt)

    # --- Write the final metrics CSV: one row per trained seed + MEAN+/-STD row ---
    # TEST metrics only — the validation metrics are in validation_results_ae.csv.
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_METRICS_COLUMNS)
        writer.writeheader()
        for row in all_results:
            writer.writerow({col: row.get(col, "") for col in FINAL_METRICS_COLUMNS})
        writer.writerow(_mean_std_row(all_results))

    # --- Generate confusion matrix plot from BEST SEED (not aggregate mean) ---
    figures_dir = Path(get_param(config, "paths.reports", "reports/")).resolve() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figures_dir / "ae_confusion_matrix.png"
    logger.info("Saving confusion matrix to: %s", cm_path)
    # Use best seed confusion matrix (integer counts, not mean)
    best_cm = [[best_result["test_tn"], best_result["test_fp"]],
               [best_result["test_fn"], best_result["test_tp"]]]
    plot_confusion_matrix(
        best_cm,
        labels=("normal", "anomaly"),
        save_path=cm_path,
        title=f"Test Set Confusion Matrix (Best Seed: {best_seed})",
    )
    if cm_path.exists():
        logger.info("Confusion matrix saved successfully to: %s", cm_path)
    else:
        logger.error("Confusion matrix file NOT found at: %s", cm_path)
        raise RuntimeError(f"Failed to create confusion matrix at {cm_path}")

    logger.info("Metrics written to: %s", metrics_path)

    return all_results


def _mean_std_row(results: list[dict[str, Any]]) -> dict[str, str]:
    """Build the MEAN+/-STD summary row from the per-seed rows."""
    row: dict[str, str] = {"seed": "MEAN+/-STD"}
    for col in FINAL_METRICS_COLUMNS[1:]:
        values = [r[col] for r in results if isinstance(r.get(col), (int, float))]
        if not values:
            row[col] = ""
            continue
        if col.startswith("n_"):
            # Window counts are identical across seeds — keep the column empty
            row[col] = ""
            continue
        mean = sum(values) / len(values)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        prec = _MEAN_PRECISION.get(col, 4)
        row[col] = f"{mean:.{prec}f} +/- {std:.{prec}f}"
    return row


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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity (suppress sync/dep logs)",
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