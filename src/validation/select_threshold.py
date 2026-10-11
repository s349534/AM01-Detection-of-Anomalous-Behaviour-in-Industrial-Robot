"""Stage 2: Report best configuration from HP validation search.

Usage:
    python -m src.validation.select_threshold
    python -m src.validation.select_threshold --input reports/tables/validation_results_ae.csv

Actions:
    1. Load the best row from the validation results CSV (produced by Stage 1)
    2. Report the best configuration and validation metrics

NOTE:
    The threshold is NO LONGER calibrated here. It is calibrated in Stage 3
    (train_ae.py) on the final trained model's own validation errors
    (val_normal + val_anomaly). This ensures the threshold matches the model
    that will actually perform inference.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def get_default_csv_path(config: dict | None = None) -> Path:
    """Resolve default validation CSV path."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    return reports_dir / "tables" / "validation_results_ae.csv"


def get_default_params_path() -> Path:
    """Resolve default params_validated_ae.yaml path."""
    return Path("config/params_validated_ae.yaml")


def load_best_row(csv_path: Path) -> dict:
    """Load the best row from the validation CSV.

    Selects the row with highest val_pr_auc, tie-break on val_roc_auc,
    then best_epoch (min).

    Parameters
    ----------
    csv_path : Path
        Path to validation_results_ae.csv.

    Returns
    -------
    dict
        The best row (as a dict of string values from CSV).
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pr_auc = float(row["val_pr_auc"])
                if pr_auc < 0:
                    continue  # skip failed runs
                rows.append(row)
            except (ValueError, KeyError):
                continue

    if not rows:
        raise ValueError(f"No valid rows found in {csv_path}")

    best = max(rows, key=lambda r: (
        float(r["val_pr_auc"]),       # primary: PR-AUC
        float(r["val_roc_auc"]),      # tie-break 1: ROC-AUC
        -int(r["best_epoch"]),        # tie-break 2: fewer epochs
    ))
    logger.info("Best row: run_id=%s, W=%s, latent_dim=%s, val_pr_auc=%s",
                best["run_id"], best["W"], best["latent_dim"], best["val_pr_auc"])
    return best


def run_threshold_selection(
    csv_path: Path | None = None,
    params_path: Path | None = None,
) -> dict:
    """Report the best AE configuration from Stage 1 (no threshold calibration).

    Threshold calibration has been moved to Stage 3 (train_ae.py) where it is
    performed on the final trained model's own validation errors.  This
    function now serves as a summary/report of the best HP configuration
    only.

    Parameters
    ----------
    csv_path : Path or None
        Path to validation_results_ae.csv. Defaults to reports/tables/.
    params_path : Path or None
        Path to params_validated_ae.yaml. (No longer written — kept for
        API backward-compat; used only to display the validated config.)

    Returns
    -------
    dict
        Summary with best_row info.
    """
    config = load_config()

    if csv_path is None:
        csv_path = get_default_csv_path(config)
    if params_path is None:
        params_path = get_default_params_path()

    if not csv_path.exists():
        logger.error("Validation CSV not found: %s", csv_path)
        logger.error("Run Stage 1 first: python -m src.validation.run_search_ae --no-resume")
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    # --- Load best row from CSV ---
    best_row = load_best_row(csv_path)
    logger.info("Selected best config: W=%s, latent_dim=%s (run_id=%s)",
                best_row["W"], best_row["latent_dim"], best_row["run_id"])
    logger.info("  val_roc_auc=%.4f, val_pr_auc=%.4f",
                float(best_row["val_roc_auc"]), float(best_row["val_pr_auc"]))

    # --- Report (threshold is calibrated in Stage 3 / train_ae.py) ---
    logger.info("Threshold is calibrated in Stage 3 (train_ae.py) on the final "
                "trained model's validation errors, not here.")

    return {
        "run_id": best_row["run_id"],
        "W": best_row["W"],
        "latent_dim": best_row["latent_dim"],
        "val_roc_auc": float(best_row["val_roc_auc"]),
        "val_pr_auc": float(best_row["val_pr_auc"]),
        "params_path": str(params_path) if params_path else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Report best AE configuration from validation search"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to validation_results_ae.csv (default: reports/tables/validation_results_ae.csv)",
    )
    parser.add_argument(
        "--params", type=Path, default=None,
        help="Path to params_validated_ae.yaml (informational only; not modified)",
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
    logger.info("=" * 60)
    logger.info("Stage 2: Best Configuration Report")
    logger.info("=" * 60)

    result = run_threshold_selection(
        csv_path=args.input,
        params_path=args.params,
    )

    logger.info("=" * 60)
    logger.info("Best AE configuration (Stage 1):")
    logger.info("  W=%s, latent_dim=%s (run_id=%s)",
                result["W"], result["latent_dim"], result["run_id"])
    logger.info("  val_roc_auc=%.4f, val_pr_auc=%.4f",
                result["val_roc_auc"], result["val_pr_auc"])
    logger.info("NOTE: Threshold is calibrated in Stage 3 (train_ae.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()