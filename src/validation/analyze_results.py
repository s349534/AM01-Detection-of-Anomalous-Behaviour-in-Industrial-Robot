"""Post-search analysis: select best config + generate sensitivity plots.

Usage:
    python -m src.validation.analyze_results
    python -m src.validation.analyze_results --input reports/tables/validation_results_ae.csv

Actions:
    1. Read the validation results CSV
    2. Select HP_AE_best (max best_val_pr_auc, tie-break best_val_f1, then best_epoch)
    3. Generate 3 sensitivity plots
    4. Write config/params_validated_ae.yaml with the best parameters
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.config import load_config, get_param

logger = logging.getLogger(__name__)

# Sensitivity plot specs: (hparam column, plot filename suffix, title, xlabel)
SENSITIVITY_SPECS = [
    ("W", "ae_W", "Sensitivity to Window Size (W)", "Window Size (timesteps)"),
    ("latent_dim", "ae_latent_dim", "Sensitivity to Latent Dimension", "Latent Dimension"),
    ("encoder_channels", "ae_encoder_channels", "Sensitivity to Encoder Channels", "Encoder Channels"),
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

    Returns a list of dicts, skipping failure rows (best_val_pr_auc == -1).
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["run_id"] = int(row["run_id"])
                row["W"] = int(row["W"])
                row["latent_dim"] = int(row["latent_dim"])
                row["best_val_pr_auc"] = float(row["best_val_pr_auc"])
                row["best_val_f1"] = float(row["best_val_f1"])
                row["best_epoch"] = int(row["best_epoch"])
                row["train_time_sec"] = float(row["train_time_sec"])
                row["seed"] = int(row["seed"])
                # Skip failed runs
                if row["best_val_pr_auc"] < 0 or row["best_val_f1"] < 0:
                    logger.warning("Skipping failed run_id=%d", row["run_id"])
                    continue
                rows.append(row)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping malformed row: %s", e)
                continue

    if not rows:
        logger.error("No valid rows found in %s", csv_path)
    return rows


def select_best_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select HP_AE_best per §4.8.8.

    Selection criteria (in order):
        1. best_val_pr_auc (max)
        2. best_val_f1 (max, tie-break)
        3. best_epoch (min, prefers faster convergence)

    Returns the best row dict.
    """
    best = max(rows, key=lambda r: (
        r["best_val_pr_auc"],      # primary: PR-AUC
        r["best_val_f1"],          # tie-break 1: F1
        -r["best_epoch"],          # tie-break 2: fewer epochs (negated for max)
    ))
    return best


def generate_sensitivity_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate 3 sensitivity plots (§4.8.8 output spec).

    Parameters
    ----------
    rows : list[dict]
        Validation result rows.
    output_dir : Path
        Directory for output PNG files.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for hparam_col, suffix, title, xlabel in SENSITIVITY_SPECS:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by hparam value
        from collections import defaultdict
        groups: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            key = row[hparam_col]
            groups[key].append(row["best_val_pr_auc"])

        if not groups:
            logger.warning("No data for %s, skipping plot", hparam_col)
            continue

        sorted_keys = sorted(groups.keys(), key=lambda v: (isinstance(v, str), v))
        means = [np.mean(groups[k]) for k in sorted_keys]
        stds = [np.std(groups[k]) if len(groups[k]) > 1 else 0.0 for k in sorted_keys]

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
        ax.set_ylabel("PR-AUC (validation)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Mark best config
        best_row = select_best_config(rows)
        best_key = best_row[hparam_col]
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


def write_validated_params(best_row: dict[str, Any], base_config: dict, output_path: Path) -> None:
    """Write params_validated_ae.yaml with the best hyperparameters.

    Merges the best config into the base params.yaml structure, overriding
    only the validated HPs.

    Parameters
    ----------
    best_row : dict
        Best result row from validation CSV.
    base_config : dict
        Base config (from load_config) to start from.
    output_path : Path
        Output YAML file path.
    """
    import copy
    validated = copy.deepcopy(base_config)

    # Override validated hyperparameters
    w_val = best_row["W"]
    validated["model"]["window_size"] = w_val
    validated["model"]["latent_dim"] = best_row["latent_dim"]
    validated["model"]["encoder"]["conv_channels"] = eval(best_row["encoder_channels"])  # noqa: S306

    # Sincronizza anche data.window_size con model.window_size (§4.8.2)
    if "data" in validated and "window_size" in validated["data"]:
        validated["data"]["window_size"] = w_val

    # Keep training settings from base config
    # (epochs, batch_size, learning_rate, etc. stay as defaults for final training)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated params to %s", output_path)
    # Usa .get() con fallback per gestire best_row parziale (es. test programmatici)
    logger.info(
        "Best config: W=%s, latent_dim=%s, channels=%s, pr_auc=%.4f, f1=%.4f",
        best_row.get("W", "?"), best_row.get("latent_dim", "?"),
        best_row.get("encoder_channels", "?"),
        float(best_row.get("best_val_pr_auc", 0.0)),
        float(best_row.get("best_val_f1", 0.0)),
    )


def run_analysis(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    params_path: Path | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run the full analysis pipeline.

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
        Summary with best_row, n_runs, output_paths.
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
        logger.error("Run: python -m src.validation.run_search_ae --n-iter 20 --seed 42")
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    # --- Load and analyze ---
    rows = load_results_csv(csv_path)
    if not rows:
        raise ValueError(f"No valid rows in {csv_path}")

    best_row = select_best_config(rows)

    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS SUMMARY (%d runs)", len(rows))
    logger.info("=" * 60)
    for row in rows:
        flag = " <-- BEST" if row == best_row else ""
        logger.info(
            "  Run %2d | W=%-2d latent=%-2d channels=%-15s | "
            "pr_auc=%.4f f1=%.4f epoch=%-3d%s",
            row["run_id"], row["W"], row["latent_dim"], row["encoder_channels"],
            row["best_val_pr_auc"], row["best_val_f1"], row["best_epoch"], flag,
        )
    logger.info("=" * 60)
    logger.info(
        "HP_AE_best: W=%d, latent_dim=%d, encoder_channels=%s",
        best_row["W"], best_row["latent_dim"], best_row["encoder_channels"],
    )
    logger.info("  best_val_pr_auc = %.4f", best_row["best_val_pr_auc"])
    logger.info("  best_val_f1    = %.4f", best_row["best_val_f1"])
    logger.info("=" * 60)

    # --- Generate plots ---
    plot_paths: list[Path] = []
    if generate_plots:
        plot_paths = generate_sensitivity_plots(rows, output_dir)

    # --- Write validated params ---
    write_validated_params(best_row, config, params_path)

    return {
        "best_row": best_row,
        "n_runs": len(rows),
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
