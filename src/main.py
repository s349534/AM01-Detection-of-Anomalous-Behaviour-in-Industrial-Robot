"""Main entry point for the anomaly detection project.

This script orchestrates:
1. Data loading and preprocessing
2. Model training
3. Evaluation and comparison
4. Results visualization

Usage:
    python src/main.py [--config CONFIG_PATH]

Note: model/data modules are currently scaffolding (placeholders). This entry
point wires up configuration parsing, environment/device reporting and module
imports so the pipeline can be launched end-to-end (locally or via SLURM).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AM01 anomaly detection pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "config.yaml",
        help="Path to the YAML configuration file (default: config/config.yaml)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"WARNING: config file not found: {path} (continuing with defaults)", file=sys.stderr)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


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
    # Importing here (rather than at top level) surfaces import errors in the
    # SLURM log only after the heavier torch import has succeeded.
    from src.data import dataset, preprocessing  # noqa: F401
    from src.models import autoencoder, adversarial_ae, compare_models  # noqa: F401
    from src.utils import metrics, visualization  # noqa: F401
    print("All src.* modules import OK.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("AM01 Anomaly Detection Pipeline")
    print("=" * 40)
    print(f"config:          {args.config}")
    if config:
        project = config.get("project", {})
        print(f"project name:    {project.get('name', '<unset>')}")
        print(f"project version: {project.get('version', '<unset>')}")

    report_device()
    import_modules()

    print("=" * 40)
    print("[Scaffolding] No training step implemented yet.")
    print("Run notebooks/01_data_exploration.ipynb to start the data exploration,")
    print("then implement the model/training logic in src/models and src/data.")


if __name__ == "__main__":
    # Allow `python src/main.py` to resolve `src` as an importable package
    # when launched from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
