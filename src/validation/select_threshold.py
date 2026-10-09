"""Stage 2: Report best configuration from HP validation search.

Usage:
    python -m src.validation.select_threshold
    python -m src.validation.select_threshold --input reports/tables/validation_results_ae.csv

Actions:
    1. Load the best row from the validation results CSV (produced by Stage 1)
    2. Load the saved raw errors + labels from the .npz file for the best run
    3. Report the best configuration and validation error statistics

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

import numpy as np

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


def find_errors_file(best_row: dict, errors_dir: Path) -> Path:
    """Locate the .npz file for the best run.

    Tries the `errors_file` column from the CSV first. If not present,
    reconstructs the filename from run_id, seed, W, latent_dim.

    Parameters
    ----------
    best_row : dict
        Best row from CSV.
    errors_dir : Path
        Directory where errors .npz files are stored.

    Returns
    -------
    Path
        Path to the .npz file.
    """
    errors_file = best_row.get("errors_file", "").strip()
    if errors_file:
        path = Path(errors_file)
        if path.exists():
            return path
        logger.warning("errors_file path in CSV does not exist: %s", path)

    # Fallback: reconstruct from run_id + seed + W + latent_dim
    run_id = best_row["run_id"]
    seed = best_row["seed"]
    W = best_row["W"]
    latent_dim = best_row["latent_dim"]
    fallback = errors_dir / f"ae_errors_run{run_id}_seed{seed}_W{W}_latent{latent_dim}.npz"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Could not find errors file for run_id={run_id}. "
            f"Tried CSV path '{errors_file}' and fallback '{fallback}'."
        )
    logger.warning("Using reconstructed errors file: %s", fallback)
    return fallback


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
        Summary with best_row info and validation error statistics.
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

    # --- Load raw errors + labels for reporting ---
    errors_dir = Path("reports/errors")
    if "errors_file" in best_row and best_row["errors_file"].strip():
        errors_path = Path(best_row["errors_file"])
    else:
        errors_path = find_errors_file(best_row, errors_dir)

    if errors_path and errors_path.exists():
        data = np.load(errors_path)
        errors = data["errors"]
        labels = data["labels"]
        n_normal = int(np.sum(labels == 0))
        n_anomaly = int(np.sum(labels == 1))
        logger.info("Validation errors from %s: %d samples "
                    "(normal=%d, anomaly=%d)", errors_path.name,
                    len(errors), n_normal, n_anomaly)
        logger.info("Error distribution: min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
                    float(np.min(errors)), float(np.max(errors)),
                    float(np.mean(errors)), float(np.median(errors)))
    else:
        errors = None
        labels = None

    # --- Report (threshold is calibrated in Stage 3 / train_ae.py) ---
    logger.info("Threshold is calibrated in Stage 3 (train_ae.py) on the final "
                "trained model's validation errors, not here.")

    return {
        "run_id": best_row["run_id"],
        "W": best_row["W"],
        "latent_dim": best_row["latent_dim"],
        "val_roc_auc": float(best_row["val_roc_auc"]),
        "val_pr_auc": float(best_row["val_pr_auc"]),
        "n_val_errors": len(errors) if errors is not None else 0,
        "params_path": str(params_path) if params_path else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Calibrate optimal threshold on best AE config (§4.7)"
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
    logger.info("  Validation error samples: %d", result["n_val_errors"])
    logger.info("NOTE: Threshold is calibrated in Stage 3 (train_ae.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
