"""CLI for model evaluation.

Usage:
    python -m src.evaluation --checkpoint reports/checkpoints/ae_baseline.pth --data-dir data/processed
    python -m src.evaluation --checkpoint reports/checkpoints/ae_baseline_seed42.pth reports/checkpoints/ae_baseline_seed123.pth reports/checkpoints/ae_baseline_seed7.pth --data-dir data/processed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.evaluation.evaluate import evaluate_checkpoints
from src.utils.visualization import plot_confusion_matrix
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained AE model(s)")
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+", required=True,
        help="Path(s) to model checkpoint(s) (.pth). Multiple checkpoints will be evaluated and aggregated."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument(
        "--config", type=Path, default=Path("config/params_validated_ae.yaml"), help="Config file (default: params_validated_ae.yaml)")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (default: calibrate on validation)")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "mse"], help="Reconstruction metric")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--calibration-method", type=str, default="f1_optimal", choices=["f1_optimal", "percentile_99", "percentile_95"], help="Threshold calibration method")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Set logging level based on --quiet flag
    if args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = load_config(args.config, None)  # params_validated_ae.yaml already contains everything

    # Evaluate checkpoints
    result = evaluate_checkpoints(
        checkpoint_paths=args.checkpoint,
        config=config,
        device=args.device,
        threshold=args.threshold,
        metric=args.metric,
        calibration_method=args.calibration_method,
    )

    # Print per-seed results
    for seed_result in result["per_seed"]:
        print(f"\n=== Seed {seed_result['seed']} ===")
        print(f"  Threshold: {seed_result.get('threshold', 0.0):.6f}")
        print(f"  Test ROC-AUC: {seed_result['roc_auc']:.4f}")
        print(f"  Test PR-AUC:  {seed_result['pr_auc']:.4f}")
        print(f"  Test F1:      {seed_result['f1']:.4f}")
        print(f"  Test Acc:     {seed_result['accuracy']:.4f}")

    # Print aggregate
    agg = result["aggregate"]
    print("\n" + "=" * 60)
    print("AGGREGATE TEST METRICS (mean ± std)")
    print("=" * 60)
    for key in [
        "roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall",
        "specificity", "false_positive_rate", "false_negative_rate",
        "balanced_accuracy", "mcc", "cohen_kappa", "npv", "jaccard",
    ]:
        if key in agg:
            m = agg[key]
            print(f"  {key}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Confusion matrix (mean over the evaluated checkpoints)
    cm = agg.get("confusion_matrix", {})
    if cm:
        print("\nConfusion Matrix (mean):")
        print(f"  [[TN={cm['tn']:.1f}, FP={cm['fp']:.1f}]")
        print(f"   [FN={cm['fn']:.1f}, TP={cm['tp']:.1f}]]")

    # Save confusion matrix image
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_array = [
        [cm.get("tn", 0), cm.get("fp", 0)],
        [cm.get("fn", 0), cm.get("tp", 0)],
    ]
    plot_confusion_matrix(
        cm_array,
        labels=("normal", "anomaly"),
        save_path=figures_dir / "ae_confusion_matrix.png",
        title="Test Set Confusion Matrix (Aggregate)",
    )
    print("\nConfusion matrix saved to: reports/figures/ae_confusion_matrix.png")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()