"""CLI for random hyperparameter search over the adversarial AE (§4.8.5-§4.8.6).

Usage:
    python -m src.validation.run_search_aae --n-iter 25 --seed 42

Produces:
    reports/tables/validation_results_aae.csv  (append mode, N rows)

Each row follows the CSV format defined in §4.8.8 of project_plan.md, extended
with ``discriminator_hidden_layers`` (validated alongside the 2 loss weights
per the updated search space §4.8.3).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from sklearn.model_selection import ParameterSampler

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
    n_iter: int = 15,
    seed: int = 42,
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
    max_val_epochs: int | None = None,
    resume: bool = True,
    val_seeds: list[int] | None = None,
) -> Path:
    """Run the random search over AAE hyperparameters across multiple seeds per config.

    Parameters
    ----------
    n_iter : int
        Number of random configurations to sample.
    seed : int
        Random seed for ParameterSampler.
    config_path : str | Path | None
        Path to config.yaml. Defaults to ``config/config.yaml``.
    params_path : str | Path | None
        Path to params.yaml. Defaults to ``config/params.yaml``.
    max_val_epochs : int or None
        Override max epochs per run (useful for quick testing).
    resume : bool
        If True and the CSV already exists, skip run_ids already present.
    val_seeds : list[int] or None
        Fixed validation seeds per config (default: [7, 42, 123]).

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

    # --- Load base config (default to validated AE params to inherit HP_AE_best) ---
    if params_path is None:
        validated_ae_path = Path("config/params_validated_ae.yaml")
        if validated_ae_path.exists():
            params_path = validated_ae_path
            logger.info("Inheriting HP_AE_best from %s", validated_ae_path)

    base_config = load_config(config_path, params_path)
    inherited_w = base_config.get("data", {}).get("window_size", "unset")
    inherited_latent = base_config.get("model", {}).get("latent_dim", "unset")
    logger.info("AAE base configuration: W=%s, latent_dim=%s", inherited_w, inherited_latent)

    output_csv = get_output_path(base_config)

    # --- Determine which run_ids already exist (for resume) ---
    existing_run_ids: set[int] = set()
    write_header = True

    # If not resuming and file exists, start fresh
    if not resume and output_csv.exists():
        output_csv.unlink()
        logger.info("Cleared existing CSV (no-resume mode)")

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
    search_space = get_search_space_aae()
    sampler = ParameterSampler(search_space, n_iter=n_iter, random_state=seed)
    samples = list(sampler)

    logger.info(
        "Starting AAE validation: %d configs (seed=%d), %d seeds/config %s, total runs=%d",
        len(samples), seed, len(val_seeds), val_seeds, len(samples) * len(val_seeds),
    )

    global_run_id = 0

    # --- Run experiments ---
    for config_idx, sample in enumerate(samples):
        config_id = config_idx + 1

        # Merge sampled HP into config (shared across all seeds for this config)
        run_config = merge_search_sample(base_config, sample)

        logger.info(
            "[Config %d/%d] Sample: recon_w=%s, adv_w=%s, disc_hidden=%s",
            config_id, len(samples),
            sample.get("training.reconstruction_weight", ""),
            sample.get("training.adversarial_weight", ""),
            sample.get("model.discriminator.hidden_layers", ""),
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
                    "  [Config %d, Seed %d] -> recon_w=%.2f adv_w=%.3f disc=%s pr_auc=%.4f f1=%.4f time=%.1fs",
                    config_id,
                    val_seed,
                    result["reconstruction_weight"],
                    result["adversarial_weight"],
                    result["discriminator_hidden_layers"],
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
                    "discriminator_hidden_layers": str(
                        sample.get("model.discriminator.hidden_layers", "")
                    ),
                    "best_val_pr_auc": -1.0,     # sentinel for failure
                    "best_val_f1": -1.0,
                    "best_epoch": 0,
                    "train_time_sec": 0.0,
                    "seed": val_seed,
                }
                write_result_row_csv(output_csv, fail_row, write_header=write_header)
                write_header = False

    logger.info("AAE search complete. Results: %s", output_csv)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random search for AAE hyperparameter validation (§4.8.3)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--params", type=Path, default=None,
        help="Path to params yaml (default: config/params_validated_ae.yaml if exists, else config/params.yaml)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=15,
        help="Number of random configurations to sample (default: 15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for ParameterSampler (default: 42)",
    )
    parser.add_argument(
        "--val-seeds", type=int, nargs="+", default=None,
        help="Validation seeds per configuration (default: 7 42 123)",
    )
    parser.add_argument(
        "--max-val-epochs", type=int, default=None,
        help="Override max epochs per run (default: from params.yaml)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from existing CSV (start fresh)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    output_csv = run_search(
        n_iter=args.n_iter,
        seed=args.seed,
        config_path=args.config,
        params_path=args.params,
        max_val_epochs=args.max_val_epochs,
        resume=not args.no_resume,
        val_seeds=args.val_seeds,
    )

    elapsed = time.time() - start
    logger.info("Total AAE search time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
