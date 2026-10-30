"""CLI for random hyperparameter search over the vanilla AE (§4.8.6, §4.8.5).

Usage:
    python -m src.validation.run_search_ae --n-iter 20 --seed 42  # random subset
    python -m src.validation.run_search_ae --max-val-epochs 50    # exhaustive (all combos)

Produces:
    reports/tables/validation_results_ae.csv  (append mode, N rows)

For each configuration, **3 models** are trained using the fixed validation seeds
VALIDATION_SEEDS = [7, 42, 123] (§4.8.6). The CSV contains 3 rows per config,
distinguished by the ``config_id`` column and the ``seed`` column.

Each row follows the CSV format defined in §4.8.8 of project_plan.md.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path

from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.utils.config import load_config
from src.utils.device import check_cuda_available
from src.validation.run_experiment_ae import train_and_evaluate_ae
from src.validation.search_space import get_search_space_ae, merge_search_sample

logger = logging.getLogger(__name__)

# Fixed validation seeds for robust hyperparameter selection (§4.8.6).
# Each configuration is trained with these 3 seeds; results are aggregated
# (mean ± std) in analyze_results.py before best-config selection.
VALIDATION_SEEDS = [7, 42, 123]

CSV_COLUMNS = [
    "run_id",
    "config_id",   # groups the 3 seed-runs for the same configuration
    "W",
    "latent_dim",
    "val_roc_auc",
    "val_pr_auc",
    "best_epoch",
    "train_time_sec",
    "seed",
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
    """Append a single result dict to the CSV (append mode).

    Validates that numeric fields are finite and in valid range before writing.
    If val_roc_auc or val_pr_auc are NaN/Inf or out of [0,1] range, writes a failure row instead.
    """
    # Validate numeric fields before writing
    csv_row = {col: row.get(col, "") for col in CSV_COLUMNS}
    
    # Check critical numeric fields for NaN/Inf and range
    try:
        val_roc_auc = float(csv_row.get("val_roc_auc", -1.0))
        val_pr_auc = float(csv_row.get("val_pr_auc", -1.0))
        best_epoch = int(csv_row.get("best_epoch", 0))
        train_time_sec = float(csv_row.get("train_time_sec", 0.0))
        
        # If AUC values are NaN/Inf or out of valid range [0, 1], convert to failure row
        if not (math.isfinite(val_roc_auc) and math.isfinite(val_pr_auc)) or \
           not (0.0 <= val_roc_auc <= 1.0) or not (0.0 <= val_pr_auc <= 1.0):
            logger.warning("Invalid AUC values (roc_auc=%s, pr_auc=%s) for run_id=%s - writing failure row",
                          csv_row.get("val_roc_auc"), csv_row.get("val_pr_auc"), csv_row.get("run_id"))
            csv_row["val_roc_auc"] = -1.0
            csv_row["val_pr_auc"] = -1.0
            csv_row["best_epoch"] = 0
            csv_row["train_time_sec"] = 0.0
        
        # Validate train_time_sec
        if not math.isfinite(train_time_sec):
            logger.warning("Invalid train_time_sec=%s for run_id=%s - setting to 0.0",
                          csv_row.get("train_time_sec"), csv_row.get("run_id"))
            csv_row["train_time_sec"] = 0.0
            
    except (ValueError, TypeError) as e:
        logger.warning("Error validating row for run_id=%s: %s - writing failure row", csv_row.get("run_id"), e)
        csv_row["val_roc_auc"] = -1.0
        csv_row["val_pr_auc"] = -1.0
        csv_row["best_epoch"] = 0
        csv_row["train_time_sec"] = 0.0
    
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
        f.flush()  # Ensure data is written to disk immediately


def run_search(
    n_iter: int | None = None,
    seed: int = 42,
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
    max_val_epochs: int | None = None,
    resume: bool = False,
    val_seeds: list[int] | None = None,
) -> Path:
    """Run the random search over AE hyperparameters.

    For each sampled configuration, ``len(val_seeds)`` models are trained
    (3 by default: 7, 42, 123). Results are written as one CSV row per seed-run,
    grouped by ``config_id``.

    Parameters
    ----------
    n_iter : int or None
        Number of random configurations to sample. If None (default), runs
        ALL combinations exhaustively using ParameterGrid.
    seed : int
        Random seed for ParameterSampler (only used when n_iter is set,
        for reproducible sampling of configurations).
    config_path : str | Path | None
        Path to config.yaml. Defaults to ``config/config.yaml``.
    params_path : str | Path | None
        Path to params.yaml. Defaults to ``config/params.yaml``.
    max_val_epochs : int or None
        Override max epochs per run (useful for quick testing).
    resume : bool
        If True and the CSV already exists, skip run_ids already present.
    val_seeds : list[int] or None
        Fixed seeds for validation runs per configuration.
        Defaults to ``VALIDATION_SEEDS`` = [7, 42, 123].

    Returns
    -------
    Path
        Path to the output CSV file.
    """
    if val_seeds is None:
        val_seeds = VALIDATION_SEEDS
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # --- Load base config ---
    base_config = load_config(
        Path(config_path) if config_path else None,
        Path(params_path) if params_path else None,
    )

    # --- Early GPU check: fail fast if CUDA not actually usable ---
    # This catches "CUDA busy/unavailable" before wasting time on search
    try:
        check_cuda_available()
    except RuntimeError as e:
        logger.error("GPU check failed: %s", e)
        logger.error("Interrompo search: GPU richiesta ma non utilizzabile.")
        logger.error("Verificare: #SBATCH --gres=gpu:1, nvidia-smi, CUDA_VISIBLE_DEVICES")
        sys.exit(1)

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
        "Starting AE validation: %d configs (%s, sample_seed=%d), "
        "%d seeds/config %s, total runs=%d",
        len(samples), mode_str, seed,
        len(val_seeds), val_seeds, len(samples) * len(val_seeds),
    )

    global_run_id = 0

    for config_idx, sample in enumerate(samples):
        config_id = config_idx + 1

        # Merge sampled HP into config (shared across all seeds for this config)
        run_config = merge_search_sample(base_config, sample)

        logger.info(
            "[Config %d/%d] Sample: %s",
            config_id, len(samples), sample,
        )

        # --- Run 3 seeds for this configuration ---
        for seed_idx, val_seed in enumerate(val_seeds):
            global_run_id += 1
            run_id = global_run_id

            if resume and run_id in existing_run_ids:
                logger.info("Skipping run_id=%d (already completed)", run_id)
                continue

            try:
                result = train_and_evaluate_ae(
                    config=run_config,
                    run_id=run_id,
                    seed=val_seed,  # fixed seed per config
                    max_val_epochs=max_val_epochs,
                )
                # Inject config_id into result row
                result["config_id"] = config_id
                write_result_row_csv(output_csv, result, write_header=write_header)
                write_header = False  # header written after first row
                logger.info(
                    "  -> seed=%d | W=%d latent=%d roc_auc=%.4f pr_auc=%.4f time=%.1fs",
                    val_seed,
                    result["W"], result["latent_dim"],
                    result["val_roc_auc"], result["val_pr_auc"],
                    result["train_time_sec"],
                )

            except Exception:
                logger.exception("Run %d (config_id=%d, seed=%d) failed",
                                 run_id, config_id, val_seed)
                # Write a failure row so the CSV maintains run_id continuity
                fail_row = {
                    "run_id": run_id,
                    "config_id": config_id,
                    "W": sample.get("data.window_size", ""),
                    "latent_dim": sample.get("model.latent_dim", ""),
                    "val_roc_auc": -1.0,  # sentinel for failure
                    "val_pr_auc": -1.0,
                    "best_epoch": 0,
                    "train_time_sec": 0.0,
                    "seed": val_seed,
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
        help="Random seed for ParameterSampler sampling only (default: 42). "
             "Training seeds are fixed: [7, 42, 123] per config (§4.8.6).",
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
    parser.add_argument(
        "--val-seeds", type=int, nargs="+", default=None,
        help="Fixed seeds for validation runs per configuration (default: 7 42 123, §4.8.6)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity (suppress sync/dep logs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    run_search(
        n_iter=args.n_iter,
        seed=args.seed,
        max_val_epochs=args.max_val_epochs,
        resume=args.resume,
        val_seeds=args.val_seeds,
    )

    elapsed = time.time() - start
    logger.info("Total search time: %.1fs", elapsed)


if __name__ == "__main__":
    main()