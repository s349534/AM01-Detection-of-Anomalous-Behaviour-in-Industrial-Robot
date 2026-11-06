"""CLI for hyperparameter search over the adversarial AE (§4.8.5-§4.8.6).

Usage:
    python -m src.validation.run_search_aae                    # exhaustive (all combos)
    python -m src.validation.run_search_aae --n-iter 25 --seed 42  # random subset

Produces:
    reports/tables/validation_results_aae.csv  (append mode, N rows)

Each row follows the CSV format defined in §4.8.8 of project_plan.md.
Discriminator hidden layers are fixed to [16, 8] (validated best) and not searched.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch

from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.utils.config import load_config
from src.validation.run_experiment_aae import train_and_evaluate_aae
from src.validation.search_space import get_search_space_aae, merge_search_sample

logger = logging.getLogger(__name__)

# Fixed validation seeds for robust hyperparameter selection (§4.8.6).
# Each configuration is trained with these 3 seeds; results are aggregated
# (mean +/- std) in analyze_results.py before best-config selection.
VALIDATION_SEEDS = [7, 42, 123]

CSV_COLUMNS = [
    "run_id",
    "config_id",
    "reconstruction_weight",
    "adversarial_weight",
    "discriminator_hidden_layers",
    "best_val_pr_auc",
    "best_val_f1",
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
    output_path = reports_dir / "tables" / "validation_results_aae.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_result_row_csv(path: Path, row: dict, write_header: bool) -> None:
    """Append a single result dict to the CSV (append mode)."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        csv_row = {col: row.get(col, "") for col in CSV_COLUMNS}
        writer.writerow(csv_row)


def run_search(
    n_iter: int | None = None,
    seed: int = 42,
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
    max_val_epochs: int | None = None,
    resume: bool = False,
    val_seeds: list[int] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run the search over AAE hyperparameters across multiple seeds per config.

    For each configuration, ``len(val_seeds)`` models are trained
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
        Path to params.yaml. Defaults to ``config/params_validated.yaml`` if exists.
    max_val_epochs : int or None
        Override max epochs per run (useful for quick testing).
    resume : bool
        If True and the CSV already exists, skip run_ids already present.
        Default is False (fresh start each time, like AE).
    val_seeds : list[int] or None
        Fixed seeds for validation runs per configuration.
        Defaults to ``VALIDATION_SEEDS`` = [7, 42, 123].
    dry_run : bool
        If True, only list the configurations that would be run without training.

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
        force=True,
    )

    # --- Load base config (default to unified validated params to inherit HP_AE_best) ---
    if params_path is None:
        validated_path = Path("config/params_validated.yaml")
        if validated_path.exists():
            params_path = validated_path
            logger.info("Inheriting HP_AE_best from %s", validated_path)

    base_config = load_config(config_path, params_path)
    inherited_w = base_config.get("data", {}).get("window_size", "unset")
    inherited_latent = base_config.get("model", {}).get("latent_dim", "unset")
    logger.info("AAE base configuration: W=%s, latent_dim=%s", inherited_w, inherited_latent)

    output_csv = get_output_path(base_config)

    # --- Build search space and sample ---
    search_space = get_search_space_aae()
    if n_iter is None:
        # Exhaustive mode: run all combinations deterministically (like AE)
        grid = ParameterGrid(search_space)
        samples = list(grid)
        mode_str = "exhaustive"
    else:
        # Random sampling mode
        sampler = ParameterSampler(search_space, n_iter=n_iter, random_state=seed)
        samples = list(sampler)
        mode_str = f"random (n_iter={n_iter})"

    total_configs = len(samples)
    total_runs = total_configs * len(val_seeds)
    logger.info(
        "Starting AAE validation: %d configs (%s, sample_seed=%d), "
        "%d seeds/config %s, total runs=%d",
        total_configs, mode_str, seed,
        len(val_seeds), val_seeds, total_runs,
    )

    # Log all configurations for visibility
    for config_idx, sample in enumerate(samples):
        config_id = config_idx + 1
        logger.info(
            "[Config %d/%d] recon_w=%s, adv_w=%s",
            config_id, total_configs,
            sample.get("training.reconstruction_weight", ""),
            sample.get("training.adversarial_weight", ""),
        )

    if dry_run:
        logger.info("DRY RUN: listed %d configurations without training", total_configs)
        return output_csv

    # --- Determine which run_ids already exist (for resume) ---
    existing_run_ids: set[int] = set()
    write_header = True

    # If not resuming and file exists, start fresh
    if not resume and output_csv.exists():
        output_csv.unlink()
        logger.info("Cleared existing CSV (no-resume mode): %s", output_csv)

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

    global_run_id = 0

    # --- Run experiments ---
    for config_idx, sample in enumerate(samples):
        config_id = config_idx + 1

        # Merge sampled HP into config (shared across all seeds for this config)
        run_config = merge_search_sample(base_config, sample)

        logger.info(
            "[Config %d/%d] Sample: recon_w=%s, adv_w=%s",
            config_id, len(samples),
            sample.get("training.reconstruction_weight", ""),
            sample.get("training.adversarial_weight", ""),
        )

        for seed_idx, val_seed in enumerate(val_seeds):
            global_run_id += 1
            run_id = global_run_id

            if resume and run_id in existing_run_ids:
                logger.info("Skipping run_id=%d (already completed)", run_id)
                continue

            try:
                result = train_and_evaluate_aae(
                    config=run_config,
                    run_id=run_id,
                    seed=val_seed,  # fixed seed per configuration
                    max_val_epochs=max_val_epochs,
                )
                result["config_id"] = config_id
                write_result_row_csv(output_csv, result, write_header=write_header)
                write_header = False  # header written after first row
                logger.info(
                    "  [Config %d, Seed %d] -> recon_w=%.2f adv_w=%.3f pr_auc=%.4f f1=%.4f time=%.1fs",
                    config_id,
                    val_seed,
                    result["reconstruction_weight"],
                    result["adversarial_weight"],
                    result["best_val_pr_auc"],
                    result["best_val_f1"],
                    result["train_time_sec"],
                )

            except Exception as e:
                logger.error("Run %d (config %d, seed %d) failed: %s", run_id, config_id, val_seed, e, exc_info=True)
                fail_row = {
                    "run_id": run_id,
                    "config_id": config_id,
                    "reconstruction_weight": sample.get("training.reconstruction_weight", ""),
                    "adversarial_weight": sample.get("training.adversarial_weight", ""),
                    "discriminator_hidden_layers": "[16, 8]",  # fixed value
                    "best_val_pr_auc": -1.0,     # sentinel for failure
                    "best_val_f1": -1.0,
                    "best_epoch": 0,
                    "train_time_sec": 0.0,
                    "seed": val_seed,
                }
                write_result_row_csv(output_csv, fail_row, write_header=write_header)
                write_header = False

            finally:
                # Free GPU memory between runs to avoid OOM / busy device errors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("AAE search complete. Results: %s", output_csv)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for AAE validation (§4.8.3). "
                    "Default is exhaustive grid search; use --n-iter for random sampling."
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--params", type=Path, default=None,
        help="Path to params yaml (default: config/params_validated.yaml if exists, else config/params.yaml)",
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
        "--val-seeds", type=int, nargs="+", default=None,
        help="Validation seeds per configuration (default: 7 42 123, §4.8.6)",
    )
    parser.add_argument(
        "--max-val-epochs", type=int, default=None,
        help="Override max epochs per run (default: from params.yaml)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing CSV (skip run_ids already present). Default is no-resume (fresh start).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Explicitly do not resume (start fresh). Default behavior.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List all configurations that would be run without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    # Resume logic: --resume enables, --no-resume disables, default is no-resume (False)
    resume = args.resume and not args.no_resume

    output_csv = run_search(
        n_iter=args.n_iter,
        seed=args.seed,
        config_path=args.config,
        params_path=args.params,
        max_val_epochs=args.max_val_epochs,
        resume=resume,
        val_seeds=args.val_seeds,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - start
    logger.info("Total AAE search time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
