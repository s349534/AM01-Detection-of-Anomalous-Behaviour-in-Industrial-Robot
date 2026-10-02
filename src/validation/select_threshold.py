"""Stage 2: Threshold calibration for the best AE configuration.

Usage:
    python -m src.validation.select_threshold
    python -m src.validation.select_threshold --input reports/tables/validation_results_ae.csv
    python -m src.validation.select_threshold --input reports/tables/validation_results_ae.csv \
        --params config/params_validated_ae.yaml

Actions:
    1. Load the best row from the validation results CSV (produced by Stage 1)
    2. Load the saved raw errors + labels from the .npz file for the best run
    3. Call find_optimal_threshold() to find the F1-maximizing threshold
    4. Check if threshold is near distribution extremes (warn if so)
    5. Write the threshold into params_validated_ae.yaml (model.threshold)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.utils.config import get_param, load_config
from src.utils.metrics import find_optimal_threshold

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
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
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


def check_threshold_extremes(errors: np.ndarray, threshold: float) -> bool:
    """Check if the threshold falls near the extremes of the error distribution.

    A threshold in the lowest 2% or highest 2% of observed errors may indicate
    poor calibration — the optimal decision boundary is near the edge of the
    validation data's support.

    Parameters
    ----------
    errors : np.ndarray
        All reconstruction errors (val_normal + val_anomaly).
    threshold : float
        The optimal threshold found by find_optimal_threshold.

    Returns
    -------
    bool
        True if threshold is near extremes (warning issued).
    """
    lower = np.percentile(errors, 2)
    upper = np.percentile(errors, 98)
    if threshold < lower:
        logger.warning(
            "Threshold %.6f is below the 2nd percentile of errors (%.6f). "
            "This may indicate poor calibration -- the validation set may not "
            "capture the true decision boundary well.", threshold, lower
        )
        return True
    elif threshold > upper:
        logger.warning(
            "Threshold %.6f is above the 98th percentile of errors (%.6f). "
            "This may indicate poor calibration -- the validation set may not "
            "capture the true decision boundary well.", threshold, upper
        )
        return True
    return False


def write_threshold_to_params(params_path: Path, threshold: float, best_row: dict) -> None:
    """Write the calibrated threshold into params_validated_ae.yaml.

    Updates the existing params_validated_ae.yaml by setting
    ``model.threshold`` to the calibrated value, along with metadata about
    the source run.

    Parameters
    ----------
    params_path : Path
        Path to params_validated_ae.yaml.
    threshold : float
        The calibrated threshold value.
    best_row : dict
        The best row from validation CSV (for metadata).
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if params is None:
        params = {}
    if "model" not in params:
        params["model"] = {}

    params["model"]["threshold"] = round(float(threshold), 6)
    # Store metadata linking threshold to source run
    params.setdefault("validation", {})
    params["validation"]["threshold_run_id"] = int(best_row["run_id"])
    params["validation"]["threshold_calibrated_at"] = datetime.now(timezone.utc).isoformat()
    params["validation"]["val_roc_auc"] = float(best_row["val_roc_auc"])
    params["validation"]["val_pr_auc"] = float(best_row["val_pr_auc"])
    params["validation"]["source_csv"] = str(get_default_csv_path())

    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote threshold=%.6f into %s", threshold, params_path)


def run_threshold_selection(
    csv_path: Path | None = None,
    params_path: Path | None = None,
) -> dict:
    """Run the full threshold calibration pipeline (Stage 2).

    Parameters
    ----------
    csv_path : Path or None
        Path to validation_results_ae.csv. Defaults to reports/tables/.
    params_path : Path or None
        Path to params_validated_ae.yaml. Defaults to config/.

    Returns
    -------
    dict
        Summary with threshold, f1, best_row info.
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

    if not params_path.exists():
        logger.error("Validated params not found: %s", params_path)
        logger.error("Run Stage 1b first: python -m src.validation.analyze_results")
        raise FileNotFoundError(f"Validated params not found: {params_path}")

    # --- Step 1: Load best row from CSV ---
    best_row = load_best_row(csv_path)
    logger.info("Selected best config: W=%s, latent_dim=%s (run_id=%s)",
                best_row["W"], best_row["latent_dim"], best_row["run_id"])

    # --- Step 2: Load raw errors + labels ---
    errors_dir = Path("reports/errors")
    errors_path = find_errors_file(best_row, errors_dir)
    data = np.load(errors_path)
    errors = data["errors"]
    labels = data["labels"]
    logger.info("Loaded %d validation samples (errors) from %s", len(errors), errors_path)

    # --- Step 3: Find optimal threshold via find_optimal_threshold ---
    threshold, best_metrics = find_optimal_threshold(labels, errors)
    logger.info("Optimal threshold: %.6f (F1=%.4f, precision=%.4f, recall=%.4f)",
                threshold, best_metrics["f1"], best_metrics["precision"], best_metrics["recall"])

    # --- Step 4: Check threshold extremes ---
    near_extremes = check_threshold_extremes(errors, threshold)

    # --- Step 5: Write threshold to params_validated_ae.yaml ---
    write_threshold_to_params(params_path, threshold, best_row)

    return {
        "threshold": threshold,
        "f1": best_metrics["f1"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "near_extremes": near_extremes,
        "run_id": best_row["run_id"],
        "W": best_row["W"],
        "latent_dim": best_row["latent_dim"],
        "params_path": str(params_path),
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
        help="Path to params_validated_ae.yaml (default: config/params_validated_ae.yaml)",
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
    logger.info("Stage 2: Threshold Calibration")
    logger.info("=" * 60)

    result = run_threshold_selection(
        csv_path=args.input,
        params_path=args.params,
    )

    logger.info("=" * 60)
    logger.info("Threshold calibration complete:")
    logger.info("  threshold: %.6f", result["threshold"])
    logger.info("  F1: %.4f (precision=%.4f, recall=%.4f)",
                result["f1"], result["precision"], result["recall"])
    if result["near_extremes"]:
        logger.warning("  WARNING: threshold near distribution extremes -- see warnings above")
    logger.info("  Calibrated on: W=%s, latent_dim=%s (run_id=%s)",
                result["W"], result["latent_dim"], result["run_id"])
    logger.info("  Written to: %s", result["params_path"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
