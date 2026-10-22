"""Post-search analysis: select best config + generate sensitivity plots.

Usage:
    python -m src.validation.analyze_results
    python -m src.validation.analyze_results --input reports/tables/validation_results_ae.csv

Actions:
    1. Read the validation results CSV
    2. **Aggregate** per-config rows by mean ± std (val_pr_auc, val_roc_auc, best_epoch)
    3. Select HP_AE_best (max mean val_pr_auc, tie-break min std val_pr_auc,
       then max mean val_roc_auc, then min mean best_epoch)
    4. Generate sensitivity plots using aggregated metrics
    5. Write config/params_validated_ae.yaml with the best parameters
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Sensitivity plot specs: (hparam column, plot filename suffix, title, xlabel)
SENSITIVITY_SPECS = [
    ("W", "ae_W", "Sensitivity to Window Size (W)", "Window Size (timesteps)"),
    ("latent_dim", "ae_latent_dim", "Sensitivity to Latent Dimension", "Latent Dimension"),
]


def get_default_results_path(config: dict | None = None) -> Path:
    """Resolve default CSV path."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    return reports_dir / "tables" / "validation_results_ae.csv"


def get_default_output_dir(config: dict | None = None) -> Path:
    """Resolve default figures directory."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    return reports_dir / "figures"


def get_default_params_path() -> Path:
    """Resolve default params_validated_ae.yaml path."""
    return Path("config/params_validated_ae.yaml")


def load_results_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load validation results from CSV.

    Returns a list of dicts, skipping failure rows (val_pr_auc == -1).
    Handles both the new CSV format (with ``config_id``) and legacy
    format (without ``config_id`` — will be aggregated by W/latent_dim).
    """
    rows: list[dict[str, Any]] = []
    n_failed = 0
    n_malformed = 0
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_data: dict[str, Any] = dict(row)
            try:
                row_data["run_id"] = int(row_data["run_id"])
                # config_id is optional (legacy CSVs don't have it)
                if row_data.get("config_id"):
                    row_data["config_id"] = int(row_data["config_id"])
                row_data["W"] = int(row_data["W"])
                row_data["latent_dim"] = int(row_data["latent_dim"])
                row_data["val_roc_auc"] = float(row_data["val_roc_auc"])
                row_data["val_pr_auc"] = float(row_data["val_pr_auc"])
                row_data["best_epoch"] = int(row_data["best_epoch"])
                row_data["train_time_sec"] = float(row_data["train_time_sec"])
                row_data["seed"] = int(row_data["seed"]) if row_data.get("seed") else 0
                # Skip failed runs (sentinel val_roc_auc/val_pr_auc == -1)
                if row_data["val_pr_auc"] < 0 or row_data["val_roc_auc"] < 0:
                    n_failed += 1
                    continue
                rows.append(row_data)
            except (ValueError, KeyError):
                n_malformed += 1
                continue

    if n_failed:
        logger.warning("Skipped %d failed run(s) (val_roc_auc/val_pr_auc == -1)", n_failed)
    if n_malformed:
        logger.warning("Skipped %d malformed row(s)", n_malformed)

    if not rows:
        logger.error("No valid rows found in %s", csv_path)
    return rows


@dataclass
class AggregatedConfig:
    """Aggregated metrics for one hyperparameter configuration across multiple seeds.

    Produced by :func:`aggregate_by_config`, one instance per unique
    ``(config_id, W, latent_dim)`` triple.
    """
    config_id: int
    W: int
    latent_dim: int
    mean_val_pr_auc: float
    std_val_pr_auc: float
    mean_val_roc_auc: float
    std_val_roc_auc: float
    mean_best_epoch: float
    std_best_epoch: float
    mean_train_time_sec: float
    seeds: list[int] = field(default_factory=list)
    n_seeds: int = 0
    individual_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)


def aggregate_by_config(rows: list[dict[str, Any]]) -> list[AggregatedConfig]:
    """Aggregate individual seed-runs into per-configuration summaries.

    Groups rows by ``config_id`` if present, otherwise by ``(W, latent_dim)``.
    Computes mean ± std for val_pr_auc, val_roc_auc, best_epoch, and
    train_time_sec.

    Parameters
    ----------
    rows : list[dict]
        Individual result rows loaded from CSV (one per seed-run).

    Returns
    -------
    list[AggregatedConfig]
        One entry per configuration, sorted by ``config_id``.
    """
    # Group rows: use config_id if available, else fall back to (W, latent_dim)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    has_config_id = any("config_id" in r for r in rows)

    if not has_config_id:
        logger.warning(
            "No config_id column found in CSV — aggregating by (W, latent_dim). "
            "This indicates a legacy CSV format."
        )

    for row in rows:
        if has_config_id and "config_id" in row:
            key: tuple = (row["config_id"],)
        else:
            key = (row["W"], row["latent_dim"])
        groups[key].append(row)

    aggregated: list[AggregatedConfig] = []
    for key, group_rows in groups.items():
        pr_aucs = [r["val_pr_auc"] for r in group_rows]
        roc_aucs = [r["val_roc_auc"] for r in group_rows]
        best_epochs = [r["best_epoch"] for r in group_rows]
        train_times = [r["train_time_sec"] for r in group_rows]
        seeds = [r["seed"] for r in group_rows if r.get("seed")]

        config_id = key[0] if has_config_id else len(aggregated) + 1

        agg = AggregatedConfig(
            config_id=config_id,
            W=group_rows[0]["W"],
            latent_dim=group_rows[0]["latent_dim"],
            mean_val_pr_auc=round(float(np.mean(pr_aucs)), 6),
            std_val_pr_auc=round(float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0, 6),
            mean_val_roc_auc=round(float(np.mean(roc_aucs)), 6),
            std_val_roc_auc=round(float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0, 6),
            mean_best_epoch=round(float(np.mean(best_epochs)), 2),
            std_best_epoch=round(float(np.std(best_epochs, ddof=1)) if len(best_epochs) > 1 else 0.0, 2),
            mean_train_time_sec=round(float(np.mean(train_times)), 2),
            seeds=seeds,
            n_seeds=len(group_rows),
            individual_rows=group_rows,
        )
        aggregated.append(agg)

    # Sort by config_id for deterministic ordering
    aggregated.sort(key=lambda a: a.config_id)
    return aggregated


def select_best_config(aggregated: list[AggregatedConfig]) -> AggregatedConfig:
    """Select HP_AE_best per §4.8.6 (robust selection with 3 fixed seeds).

    Selection criteria (in order):
        1. mean val_pr_auc (max) — primary metric, threshold-free
        2. std val_pr_auc (min) — tie-break: prefers stable configs
        3. mean val_roc_auc (max) — secondary metric
        4. mean best_epoch (min) — prefers faster convergence

    Returns the best AggregatedConfig.
    """
    best = max(aggregated, key=lambda a: (
        a.mean_val_pr_auc,        # primary: mean PR-AUC (threshold-free)
        -a.std_val_pr_auc,        # tie-break 1: lower std (more stable)
        a.mean_val_roc_auc,       # tie-break 2: mean ROC-AUC
        -a.mean_best_epoch,       # tie-break 3: fewer epochs (negated for max)
    ))
    return best


def generate_sensitivity_plots(
    aggregated: list[AggregatedConfig],
    output_dir: Path,
    best: AggregatedConfig | None = None,
) -> list[Path]:
    """Generate sensitivity plots from aggregated per-config metrics (§4.8.8).

    Groups aggregated configs by the hparam value (e.g. all configs sharing the
    same W), then computes mean ± std of ``mean_val_pr_auc`` across those configs.

    Parameters
    ----------
    aggregated : list[AggregatedConfig]
        Per-configuration aggregated results.
    output_dir : Path
        Directory for output PNG files.
    best : AggregatedConfig or None
        The best config (from :func:`select_best_config`) to mark on plots.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for hparam_col, suffix, title, xlabel in SENSITIVITY_SPECS:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group aggregated configs by hparam value
        # (e.g. for W: group all configs with W=16, W=32, etc.)
        groups: dict[Any, list[float]] = defaultdict(list)
        for agg in aggregated:
            hparam_val = getattr(agg, hparam_col)
            groups[hparam_val].append(agg.mean_val_pr_auc)

        if not groups:
            logger.warning("No data for %s, skipping plot", hparam_col)
            continue

        sorted_keys = sorted(groups.keys(), key=lambda v: (isinstance(v, str), v))
        means = [float(np.mean(groups[k])) for k in sorted_keys]
        stds = np.array(
            [float(np.std(groups[k], ddof=1)) if len(groups[k]) > 1 else 0.0
             for k in sorted_keys],
            dtype=float,
        )

        x_positions = list(range(len(sorted_keys)))
        ax.errorbar(
            x_positions, means, yerr=stds,
            fmt="o-", color="#2563eb", markersize=8, linewidth=2,
            capsize=5, capthick=1.5,
            ecolor="#ef4444",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(k) for k in sorted_keys], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Mean PR-AUC (validation, mean over seeds)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark best config
        if best is not None:
            best_key = getattr(best, hparam_col)
            if best_key in sorted_keys:
                best_idx = sorted_keys.index(best_key)
                ax.scatter([best_idx], [means[best_idx]], color="#f59e0b", s=150,
                           zorder=5, label="Best config", edgecolors="black", linewidths=1.5)
                ax.legend(fontsize=11)

        fig.tight_layout()
        save_path = output_dir / f"sensitivity_{suffix}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Generated %s", save_path)
        generated.append(save_path)

    return generated


def write_validated_params(
    best: AggregatedConfig,
    base_config: dict,
    output_path: Path,
) -> None:
    """Write params_validated_ae.yaml with the best hyperparameters.

    Merges the best config into the base params.yaml structure, overriding
    only the validated HPs: W (window_size), latent_dim.

    The threshold is NO LONGER written here — it is calibrated in Stage 3
    (train_ae.py) on the final trained model's validation errors.

    Parameters
    ----------
    best : AggregatedConfig
        Best aggregated config from validation search.
    base_config : dict
        Base config (from load_config) to start from.
    output_path : Path
        Output YAML file path.
    """
    validated = copy.deepcopy(base_config)

    # Override validated hyperparameters (Stage 1: model selection)
    w_val = best.W
    # Single source of truth: data.window_size
    validated["data"]["window_size"] = w_val
    validated["model"]["latent_dim"] = best.latent_dim

    # Threshold is calibrated per-seed in Stage 3 (train_ae.py),
    # NOT in Stage 2 (select_threshold.py). No placeholder here.

    # Keep training settings from base config
    # (epochs, batch_size, learning_rate, etc. stay as defaults for final training)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated params to %s", output_path)
    logger.info(
        "Best config: W=%d, latent_dim=%d, "
        "val_roc_auc=%.4f +/- %.4f, val_pr_auc=%.4f +/- %.4f (n_seeds=%d)",
        best.W, best.latent_dim,
        best.mean_val_roc_auc, best.std_val_roc_auc,
        best.mean_val_pr_auc, best.std_val_pr_auc,
        best.n_seeds,
    )


def run_analysis(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    params_path: Path | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run the full analysis pipeline.

    Loads individual seed-run rows from CSV, aggregates per configuration
    (mean ± std over 3 fixed seeds), selects the best config, and writes
    validated params + sensitivity plots.

    Parameters
    ----------
    csv_path : Path or None
        Path to validation_results_ae.csv. Defaults to reports/tables/.
    output_dir : Path or None
        Directory for sensitivity plots. Defaults to reports/figures/.
    params_path : Path or None
        Output path for params_validated_ae.yaml. Defaults to config/.
    generate_plots : bool
        Whether to generate sensitivity plots.

    Returns
    -------
    dict
        Summary with ``best_config``, ``aggregated``, ``n_runs``, ``n_configs``,
        ``plot_paths``, ``params_path``.
    """
    config = load_config()

    if csv_path is None:
        csv_path = get_default_results_path(config)
    if output_dir is None:
        output_dir = get_default_output_dir(config)
    if params_path is None:
        params_path = get_default_params_path()

    if not csv_path.exists():
        logger.error("Results CSV not found: %s", csv_path)
        logger.error("Run: python -m src.validation.run_search_ae --n-iter 20 --seed 42 "
                 "(or --max-val-epochs 50 for exhaustive)")
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    # --- Load individual seed-run rows ---
    rows = load_results_csv(csv_path)
    if not rows:
        raise ValueError(f"No valid rows in {csv_path}")

    # --- Aggregate per configuration (mean ± std over 3 seeds) ---
    aggregated = aggregate_by_config(rows)

    # --- Select best config from aggregated metrics ---
    best = select_best_config(aggregated)

    logger.info("=" * 70)
    logger.info("VALIDATION RESULTS SUMMARY: %d runs, %d configs (seeds=%s)",
                len(rows), len(aggregated),
                aggregated[0].seeds if aggregated else "[]")
    logger.info("=" * 70)
    for agg in aggregated:
        flag = " <-- BEST" if agg == best else ""
        logger.info(
            "  Config %2d | W=%-2d latent=%-2d | "
            "pr_auc=%.4f +/- %.4f | roc_auc=%.4f +/- %.4f | "
            "epoch=%.1f | n_seeds=%d%s",
            agg.config_id, agg.W, agg.latent_dim,
            agg.mean_val_pr_auc, agg.std_val_pr_auc,
            agg.mean_val_roc_auc, agg.std_val_roc_auc,
            agg.mean_best_epoch, agg.n_seeds, flag,
        )
    logger.info("=" * 70)
    logger.info("HP_AE_best: W=%d, latent_dim=%d", best.W, best.latent_dim)
    logger.info("  val_pr_auc  = %.4f +/- %.4f", best.mean_val_pr_auc, best.std_val_pr_auc)
    logger.info("  val_roc_auc = %.4f +/- %.4f", best.mean_val_roc_auc, best.std_val_roc_auc)
    logger.info("  best_epoch  = %.1f +/- %.1f", best.mean_best_epoch, best.std_best_epoch)
    logger.info("  seeds       = %s", best.seeds)
    logger.info("=" * 70)

    # --- Generate plots ---
    plot_paths: list[Path] = []
    if generate_plots:
        plot_paths = generate_sensitivity_plots(aggregated, output_dir, best=best)

    # --- Write validated params ---
    write_validated_params(best, config, params_path)

    return {
        "best_config": best,
        "aggregated": aggregated,
        "n_runs": len(rows),
        "n_configs": len(aggregated),
        "csv_path": str(csv_path),
        "plot_paths": [str(p) for p in plot_paths],
        "params_path": str(params_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AE validation results and select best config"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to validation_results_ae.csv (default: reports/tables/validation_results_ae.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for sensitivity plots (default: reports/figures/)",
    )
    parser.add_argument(
        "--params-out", type=Path, default=None,
        help="Output path for params_validated_ae.yaml (default: config/params_validated_ae.yaml)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating sensitivity plots",
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

    run_analysis(
        csv_path=args.input,
        output_dir=args.output_dir,
        params_path=args.params_out,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
