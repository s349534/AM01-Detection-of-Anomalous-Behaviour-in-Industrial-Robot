"""CLI for model evaluation.

Usage:
    python -m src.evaluation --checkpoint reports/checkpoints/ae_baseline.pth --data-dir data/processed
    python -m src.evaluation --checkpoint reports/checkpoints/ae_baseline.pth --data-dir data/processed --threshold 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.evaluation.evaluate import run_evaluation
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained AE model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"), help="Params file")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (default: calibrate on validation)")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "mse"], help="Reconstruction metric")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--calibration-method", type=str, default="f1_optimal", choices=["f1_optimal", "percentile_99", "percentile_95"], help="Threshold calibration method")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = load_config(args.config, args.params)

    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        processed_dir=args.data_dir,
        config=config,
        device=args.device,
        threshold=args.threshold,
        metric=args.metric,
        calibration_method=args.calibration_method,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Threshold: {result['threshold']:.6f}")
    if result['val_metrics']:
        print(f"Val F1:    {result['val_metrics']['f1']:.4f}")
    tm = result['test_metrics']
    print(f"Test PR-AUC: {tm['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {tm['roc_auc']:.4f}")
    print(f"Test F1:     {tm['f1']:.4f}")
    print(f"Test Acc:    {tm['accuracy']:.4f}")
    print(f"Test Prec:   {tm['precision']:.4f}")
    print(f"Test Rec:    {tm['recall']:.4f}")
    cm = tm['confusion_matrix']
    print(f"CM: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print("=" * 60)


if __name__ == "__main__":
    main()