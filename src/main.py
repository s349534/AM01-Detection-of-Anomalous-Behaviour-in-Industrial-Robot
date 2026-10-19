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
        from src.models import adversarial_ae, autoencoder  # noqa: F401
        print("models import OK.")
    except ImportError as e:
        print(f"models import skipped (scaffolding): {e}")
    from src.utils import visualization  # noqa: F401
    print("All src.* modules import OK.")


def run_phase_preprocess(config: dict) -> None:
    """Run Fase 2 — preprocessing pipeline (fully implemented)."""
    from src.data.preprocessing import run_preprocessing

    print("\n=== Fase 2: Preprocessing ===")
    run_preprocessing(config)


def run_phase_train(config: dict) -> None:
    """Run Fase 3 — Train baseline SequenceAutoencoder."""
    from src.models.train_ae import run_final_training

    print("\n=== Fase 3: Train Baseline SequenceAutoencoder ===")
    run_final_training(
        config_path="config/params_validated_ae.yaml",
        seeds=[42, 123, 7],
    )


def run_phase_evaluate(config: dict) -> None:
    """Fase 5 — Evaluate trained model on test set."""
    from src.evaluation.evaluate import run_evaluation
    from pathlib import Path

    checkpoint_path = Path("reports/checkpoints/ae_baseline.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}. Train first with --phase train")
        return

    print("\n=== Fase 5: Evaluate Model ===")
    result = run_evaluation(
        checkpoint_path=checkpoint_path,
        processed_dir=Path(config["paths"]["data_processed"]),
        config=config,
    )
    # Results printed by run_evaluation


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.params)

    # Ensure UTF-8 output on Windows (cp1252 can't encode em-dashes etc.)
    # On Linux this is a no-op.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
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
        for phase_fn in phases.values():
            phase_fn(config)
    else:
        phases[args.phase](config)

    print("\nPipeline finished.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()