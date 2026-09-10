"""Main entry point for the AM01 anomaly detection pipeline.

This script orchestrates:
1. Data loading and preprocessing
2. Model training
3. Evaluation and comparison
4. Results visualization

Usage:
    uv run python src/main.py [--config CONFIG] [--phase {preprocess,train,evaluate,all}]

Note: model/data modules are currently scaffolding (placeholders) for Fases 3–5.
Fase 2 (preprocessing) is fully implemented and wired up — use ``--phase preprocess``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AM01 anomaly detection pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "params.yaml",
        help="Path to params.yaml (default: config/params.yaml)",
    )
    parser.add_argument(
        "--phase",
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Which pipeline phase to run (default: all)",
    )
    return parser.parse_args()


def report_device() -> None:
    print("--- PyTorch / device ---")
    print(f"torch:           {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:    {torch.version.cuda}")
    else:
        print("Running in CPU-only mode.")


def import_modules() -> None:
    print("--- Importing project modules ---")
    from src.data import dataset, preprocessing  # noqa: F401
    try:
        from src.models import autoencoder, adversarial_ae, compare_models  # noqa: F401
        print("models import OK.")
    except ImportError as e:
        print(f"models import skipped (scaffolding): {e}")
    from src.utils import metrics, visualization  # noqa: F401
    print("All src.* modules import OK.")


def run_phase_preprocess(config: dict) -> None:
    """Run Fase 2 — preprocessing pipeline (fully implemented)."""
    from src.data.preprocessing import run_preprocessing

    print("\n=== Fase 2: Preprocessing ===")
    run_preprocessing(config)


def run_phase_train(config: dict) -> None:
    """Run Fase 3 — Train baseline SequenceAutoencoder."""
    from src.data.dataset import build_datasets
    from src.models.autoencoder import SequenceAutoencoder, fit_autoencoder
    from src.utils.config import get_param

    print("\n=== Fase 3: Train Baseline SequenceAutoencoder ===")
    processed_dir = Path(config["paths"]["data_processed"])
    if not (processed_dir / "train.npy").exists():
        print(f"Processed data not found at {processed_dir}. Running preprocessing first...")
        from src.data.preprocessing import run_preprocessing
        run_preprocessing(config)

    window_size = int(get_param(config, "data.window_size", 16))
    window_stride = int(get_param(config, "data.window_stride", 1))
    batch_size = int(get_param(config, "training.batch_size", 256))
    epochs = int(get_param(config, "training.epochs", 100))
    lr = float(get_param(config, "training.learning_rate", 1e-3))
    weight_decay = float(get_param(config, "training.weight_decay", 1e-4))
    patience = int(get_param(config, "training.early_stopping.patience", 10))
    min_delta = float(get_param(config, "training.early_stopping.min_delta", 1e-4))

    print(f"Loading datasets from {processed_dir} (W={window_size}, batch_size={batch_size})...")
    loaders = build_datasets(
        processed_dir=processed_dir,
        window_size=window_size,
        stride=window_stride,
        batch_size=batch_size,
        num_workers=0,
    )

    model = SequenceAutoencoder.from_config(config)
    checkpoint_path = Path("reports/checkpoints/ae_baseline.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    history = fit_autoencoder(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print(f"Baseline Autoencoder training complete. Checkpoint: {checkpoint_path}")
    print("[Pending] Fase 4: Train Adversarial Autoencoder")



def run_phase_evaluate(config: dict) -> None:
    """Fase 5 placeholder (model comparison + metrics)."""
    print("\n[Not yet implemented] Fase 5: Evaluate and compare models")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.params)

    # Ensure UTF-8 output on Windows (cp1252 can't encode em-dashes etc.)
    # On Linux this is a no-op.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    print("AM01 Anomaly Detection Pipeline")
    print("=" * 40)
    print(f"config:          {args.config}")
    print(f"params:          {args.params}")
    print(f"phase:           {args.phase}")
    if config:
        project = config.get("project", {})
        print(f"project name:    {project.get('name', '<unset>')}")
        print(f"project version: {project.get('version', '<unset>')}")

    report_device()
    import_modules()

    print("=" * 40)

    phases = {
        "preprocess": run_phase_preprocess,
        "train": run_phase_train,
        "evaluate": run_phase_evaluate,
    }

    if args.phase == "all":
        for phase_name, phase_fn in phases.items():
            phase_fn(config)
    else:
        phases[args.phase](config)

    print("\nPipeline finished.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
