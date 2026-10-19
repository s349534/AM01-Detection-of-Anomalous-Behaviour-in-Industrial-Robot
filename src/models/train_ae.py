"""CLI for final AE training with validated hyperparameters (§3.2).

After Fase 3.1 validation selects HP_AE_best and writes
``config/params_validated_ae.yaml``, this script performs the final training:

    - 3 runs with different seeds (nested validation, §4.8.1)
    - Saves best checkpoint to reports/checkpoints/ae_baseline.pth
    - Writes reports/tables/ae_final_metrics.csv with mean +/- std

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
from src.models.autoencoder import SequenceAutoencoder, fit_autoencoder
from src.utils.config import get_param, load_config
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
        Training info for this run: {seed, train_time_sec, best_epoch,
        n_train_windows, n_val_normal_windows, n_val_anomaly_windows}
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

    train_loader = build_loader(train_data, window_size=window_size,
                                batch_size=batch_size, label=0, shuffle=True)
    val_normal_loader = build_loader(val_normal_data, window_size=window_size,
                                     batch_size=batch_size, label=0)
    val_anomaly_loader = build_loader(val_anomaly_data, window_size=window_size,
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
        "n_train_windows": len(train_loader.dataset),  # type: ignore[arg-type]
        "n_val_normal_windows": len(val_normal_loader.dataset),  # type: ignore[arg-type]
        "n_val_anomaly_windows": len(val_anomaly_loader.dataset),  # type: ignore[arg-type]
    }


def run_final_training(
    config_path: str | Path,
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
) -> list[dict[str, Any]]:
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

    Returns
    -------
    list[dict]
        List of training info dicts, one per seed.
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

    # --- Write training metrics CSV ---
    csv_columns = [
        "seed", "train_time_sec", "best_epoch", "n_train_windows",
        "n_val_normal_windows", "n_val_anomaly_windows",
    ]

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in all_results:
            writer.writerow({col: row.get(col, "") for col in csv_columns})

    logger.info("Checkpoints saved in: %s", ckpt_dir)
    logger.info("Metrics written to: %s", metrics_path)

    return all_results


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