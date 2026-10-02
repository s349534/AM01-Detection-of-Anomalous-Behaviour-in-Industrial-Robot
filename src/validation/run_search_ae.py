"""CLI for random hyperparameter search over the vanilla AE (§4.8.6, §4.8.5).

Usage:
    python -m src.validation.run_search_ae --n-iter 20 --seed 42  # random subset
    python -m src.validation.run_search_ae --max-val-epochs 50    # exhaustive (all combos)

Produces:
    reports/tables/validation_results_ae.csv  (append mode, N rows)

Each row follows the CSV format defined in §4.8.8 of project_plan.md.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.utils.config import load_config
from src.validation.run_experiment_ae import train_and_evaluate_ae
from src.validation.search_space import get_search_space_ae, merge_search_sample

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "run_id",
    "W",
    "latent_dim",
    "val_roc_auc",
    "val_pr_auc",
    "best_epoch",
    "train_time_sec",
    "seed",
    "errors_file",
]


def get_output_path(config: dict | None = None) -> Path:
    """Resolve the CSV output path from config or use default."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    output_path = reports_dir / "tables" / "validation_results_ae.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_result_row_csv(path: Path, row: dict, write_header: bool) -> None:
    """Append a single result dict to the CSV (append mode)."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        # Only write the columns defined in CSV_COLUMNS (drop _test_metrics)
        csv_row = {col: row.get(col, "") for col in CSV_COLUMNS}
        writer.writerow(csv_row)


def run_search(
    n_iter: int | None = None,
    seed: int = 42,
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
    max_val_epochs: int | None = None,
    resume: bool = False,
) -> Path:
    """Run the random search over AE hyperparameters.

    Parameters
    ----------
    n_iter : int or None
        Number of random configurations to sample. If None (default), runs
        ALL combinations exhaustively using ParameterGrid.
    seed : int
        Random seed for ParameterSampler (only used when n_iter is set).
    config_path : str | Path | None
        Path to config.yaml. Defaults to ``config/config.yaml``.
    params_path : str | Path | None
        Path to params.yaml. Defaults to ``config/params.yaml``.
    max_val_epochs : int or None
        Override max epochs per run (useful for quick testing).
    resume : bool
        If True and the CSV already exists, skip run_ids already present.

    Returns
    -------
    Path
        Path to the output CSV file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # --- Load base config ---
    base_config = load_config(config_path, params_path)

    output_csv = get_output_path(base_config)

    # --- Determine which run_ids already exist (for resume) ---
    existing_run_ids: set[int] = set()
    write_header = True
    if not resume and output_csv.exists():
        # --no-resume: start fresh, remove existing CSV
        output_csv.unlink()
        logger.info("Removed existing CSV (no-resume mode)")
    if resume and output_csv.exists() and output_csv.stat().st_size > 0:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_run_ids.add(int(row["run_id"]))
                except (ValueError, KeyError):
                    pass
        write_header = False  # header already exists
        logger.info("Resuming: %d run(s) already in CSV", len(existing_run_ids))

    # --- Build search space and sample ---
    search_space = get_search_space_ae()

    if n_iter is None:
        # Exhaustive mode: run all combinations deterministically
        grid = ParameterGrid(search_space)
        samples = list(grid)
        mode_str = "exhaustive"
    else:
        # Random sampling mode
        sampler = ParameterSampler(search_space, n_iter=n_iter, random_state=seed)
        samples = list(sampler)
        mode_str = f"random (n_iter={n_iter})"

    logger.info(
        "Starting AE validation: %d iterations (%s, seed=%d), %d unique configs",
        len(samples), mode_str, seed, len(samples),
    )

    # --- Run experiments ---
    # Pre-load validation data shape for window_size validation
    processed_dir = Path(base_config.get("paths", {}).get("data_processed", "data/processed/"))
    val_data_path = processed_dir / "val_normal.npy"
    val_samples = None
    if val_data_path.exists():
        val_samples = np.load(val_data_path, mmap_mode="r").shape[0]
        logger.info("Validation set size: %d samples", val_samples)
    else:
        logger.warning("Validation data not found at %s -- cannot validate window_size", val_data_path)

    for i, sample in enumerate(samples):
        run_id = i + 1
        if resume and run_id in existing_run_ids:
            logger.info("Skipping run_id=%d (already completed)", run_id)
            continue

        # Merge sampled HP into config
        run_config = merge_search_sample(base_config, sample)

        # --- Validate window_size against validation set size ---
        window_size = int(run_config.get("model", {}).get("window_size", 16))
        if val_samples is not None and window_size > val_samples:
            logger.warning(
                "Skipping run_id=%d: window_size=%d > val_samples=%d would produce empty validation set",
                run_id, window_size, val_samples
            )
            # Write a failure row to maintain CSV continuity
            fail_row = {
                "run_id": run_id,
                "W": window_size,
                "latent_dim": run_config.get("model", {}).get("latent_dim", ""),
                "val_roc_auc": -1.0,
                "val_pr_auc": -1.0,
                "best_epoch": 0,
                "train_time_sec": 0.0,
                "seed": seed + run_id,
                "errors_file": "",
            }
            write_result_row_csv(output_csv, fail_row, write_header=write_header)
            write_header = False
            continue

        logger.info(
            "[Run %d/%d] Sample: %s",
            run_id, len(samples), sample,
        )

        try:
            result = train_and_evaluate_ae(
                config=run_config,
                run_id=run_id,
                seed=seed + run_id,  # varied seed per run
                max_val_epochs=max_val_epochs,
            )

            write_result_row_csv(output_csv, result, write_header=write_header)
            write_header = False  # header written after first row
            logger.info(
                "  -> W=%d latent=%d roc_auc=%.4f pr_auc=%.4f time=%.1fs",
                result["W"], result["latent_dim"],
                result["val_roc_auc"], result["val_pr_auc"],
                result["train_time_sec"],
            )

        except Exception as e:
            logger.error("Run %d failed: %s", run_id, e, exc_info=True)
            # Write a failure row so the CSV maintains run_id continuity
            fail_row = {
                "run_id": run_id,
                "W": sample.get("model.window_size", ""),
                "latent_dim": sample.get("model.latent_dim", ""),
                "val_roc_auc": -1.0,  # sentinel for failure
                "val_pr_auc": -1.0,
                "best_epoch": 0,
                "train_time_sec": 0.0,
                "seed": seed + run_id,
                "errors_file": "",
            }
            write_result_row_csv(output_csv, fail_row, write_header=write_header)
            write_header = False

    logger.info("Search complete. Results: %s", output_csv)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random search for AE hyperparameter validation (§4.8.2)"
    )
    parser.add_argument(
        "--n-iter", type=int, default=None,
        help="Number of random configurations to sample. If not set, runs ALL combinations exhaustively (default: exhaustive)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for ParameterSampler (default: 42)",
    )
    parser.add_argument(
        "--max-val-epochs", type=int, default=None,
        help="Override max epochs per run (default: from params.yaml)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from existing CSV (start fresh)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing CSV (skip run_ids already present). "
             "Default is no-resume (fresh start each time).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    output_csv = run_search(
        n_iter=args.n_iter,
        seed=args.seed,
        max_val_epochs=args.max_val_epochs,
        resume=args.resume,
    )

    elapsed = time.time() - start
    logger.info("Total search time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
