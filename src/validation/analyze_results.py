"""Post-search analysis: select best config + generate sensitivity plots.

Supports both AE and AAE validation result CSVs (§4.8.5 workflow).

Usage:
    python -m src.validation.analyze_results
    python -m src.validation.analyze_results --input reports/tables/validation_results_ae.csv
    python -m src.validation.analyze_results --input reports/tables/validation_results_aae.csv
    python -m src.validation.analyze_results --aae

Actions:
    1. Read the validation results CSV
    2. Select best config (max best_val_pr_auc, tie-break best_val_f1, then best_epoch)
    3. Generate sensitivity plots (3 for AE, 3 for AAE)
    4. Write params_validated_{ae|aae}.yaml with the best parameters
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

# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------
AE_SPECS = {
    "csv_columns": ["run_id", "W", "latent_dim", "encoder_channels",
                    "best_val_pr_auc", "best_val_f1", "best_epoch",
                    "train_time_sec", "seed"],
    # (hparam column, plot filename suffix, title, xlabel)
    "sensitivity": [
        ("W", "ae_W", "Sensitivity to Window Size (W)", "Window Size (timesteps)"),
        ("latent_dim", "ae_latent_dim", "Sensitivity to Latent Dimension", "Latent Dimension"),
        ("encoder_channels", "ae_encoder_channels", "Sensitivity to Encoder Channels", "Encoder Channels"),
    ],
    "default_input": "validation_results_ae.csv",
    "default_params_out": "params_validated_ae.yaml",
}

AAE_SPECS = {
    "csv_columns": ["run_id", "reconstruction_weight", "adversarial_weight",
                    "discriminator_hidden_layers", "best_val_pr_auc", "best_val_f1",
                    "best_epoch", "train_time_sec", "seed"],
    "sensitivity": [
        ("reconstruction_weight", "aae_reconstruction_weight",
         "Sensitivity to Reconstruction Weight", "Reconstruction Weight (λ_rec)"),
        ("adversarial_weight", "aae_adversarial_weight",
         "Sensitivity to Adversarial Weight", "Adversarial Weight (λ_adv)"),
        ("discriminator_hidden_layers", "aae_discriminator_hidden_layers",
         "Sensitivity to Discriminator Architecture", "Discriminator Hidden Layers"),
    ],
    "default_input": "validation_results_aae.csv",
    "default_params_out": "params_validated_aae.yaml",
}


def _get_specs(is_aae: bool) -> dict:
    """Return the configuration dict for AE or AAE mode."""
    return AAE_SPECS if is_aae else AE_SPECS


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def get_default_results_path(config: dict | None = None, is_aae: bool = False) -> Path:
    """Resolve default CSV path based on mode."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    csv_name = _get_specs(is_aae)["default_input"]
    return reports_dir / "tables" / csv_name


def get_default_output_dir(config: dict | None = None) -> Path:
    """Resolve default figures directory."""
    if config:
        reports_dir = Path(config.get("paths", {}).get("reports", "reports/"))
    else:
        reports_dir = Path("reports/")
    return reports_dir / "figures"


def get_default_params_path(is_aae: bool = False) -> Path:
    """Resolve default params_validated_{ae|aae}.yaml path."""
    fname = _get_specs(is_aae)["default_params_out"]
    return Path("config") / fname


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_results_csv(csv_path: Path, is_aae: bool = False) -> list[dict[str, Any]]:
    """Load validation results from CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the validation results CSV.
    is_aae : bool
        If True, parse AAE columns (reconstruction_weight, adversarial_weight,
        discriminator_hidden_layers).  Otherwise parse AE columns.

    Returns
    -------
    list[dict]
        List of result rows, skipping failure rows (best_val_pr_auc == -1).
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["run_id"] = int(row["run_id"])
                row["best_val_pr_auc"] = float(row["best_val_pr_auc"])
                row["best_val_f1"] = float(row["best_val_f1"])
                row["best_epoch"] = int(row["best_epoch"])
                row["train_time_sec"] = float(row["train_time_sec"])
                row["seed"] = int(row["seed"])

                if is_aae:
                    row["reconstruction_weight"] = float(row["reconstruction_weight"])
                    row["adversarial_weight"] = float(row["adversarial_weight"])
                    # discriminator_hidden_layers is a string like "[32, 16]"
                    disc_str = row["discriminator_hidden_layers"]
                    row["discriminator_hidden_layers"] = disc_str
                else:
                    row["W"] = int(row["W"])
                    row["latent_dim"] = int(row["latent_dim"])
                    row["encoder_channels"] = str(row["encoder_channels"])

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


# ---------------------------------------------------------------------------
# Best config selection (same logic for AE and AAE)
# ---------------------------------------------------------------------------
def select_best_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best config per §4.8.8.

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


# ---------------------------------------------------------------------------
# Sensitivity plots
# ---------------------------------------------------------------------------
def generate_sensitivity_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    is_aae: bool = False,
) -> list[Path]:
    """Generate sensitivity plots for AE or AAE validation results.

    Parameters
    ----------
    rows : list[dict]
        Validation result rows.
    output_dir : Path
        Directory for output PNG files.
    is_aae : bool
        If True, generate AAE sensitivity plots; otherwise AE plots.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    specs = _get_specs(is_aae)["sensitivity"]

    for hparam_col, suffix, title, xlabel in specs:
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


# ---------------------------------------------------------------------------
# Validated params writers
# ---------------------------------------------------------------------------
def write_validated_params_ae(best_row: dict[str, Any], base_config: dict, output_path: Path) -> None:
    """Write params_validated_ae.yaml with the best hyperparameters.

    Overrides ``model.window_size``, ``model.latent_dim``, and
    ``model.encoder.conv_channels`` from the best validation run.
    """
    import copy
    validated = copy.deepcopy(base_config)

    # Override validated hyperparameters
    w_val = int(best_row["W"])
    validated["model"]["window_size"] = w_val
    validated["model"]["latent_dim"] = int(best_row["latent_dim"])
    validated["model"]["encoder"]["conv_channels"] = eval(best_row["encoder_channels"])  # noqa: S306

    # Synchronize data.window_size with model.window_size (§4.8.2)
    if "data" in validated and "window_size" in validated["data"]:
        validated["data"]["window_size"] = w_val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated AE params to %s", output_path)
    logger.info(
        "Best config: W=%s, latent_dim=%s, channels=%s, pr_auc=%.4f, f1=%.4f",
        best_row.get("W", "?"), best_row.get("latent_dim", "?"),
        best_row.get("encoder_channels", "?"),
        float(best_row.get("best_val_pr_auc", 0.0)),
        float(best_row.get("best_val_f1", 0.0)),
    )


def write_validated_params_aae(best_row: dict[str, Any], base_config: dict, output_path: Path) -> None:
    """Write params_validated_aae.yaml with the best AAE hyperparameters.

    Overrides ``training.reconstruction_weight``,
    ``training.adversarial_weight``, and ``model.discriminator.hidden_layers``
    from the best validation run.  AE-derived HPs are already at ``HP_AE_best``.
    """
    import copy
    validated = copy.deepcopy(base_config)

    # Override AAE-specific validated hyperparameters
    validated["training"]["reconstruction_weight"] = float(best_row["reconstruction_weight"])
    validated["training"]["adversarial_weight"] = float(best_row["adversarial_weight"])

    # Parse discriminator hidden_layers from string like "[32, 16]"
    disc_str = best_row["discriminator_hidden_layers"]
    disc_layers = eval(disc_str)  # noqa: S306
    validated["model"]["discriminator"]["hidden_layers"] = disc_layers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated AAE params to %s", output_path)
    logger.info(
        "Best config: recon_w=%.2f, adv_w=%.3f, disc=%s, pr_auc=%.4f, f1=%.4f",
        float(best_row["reconstruction_weight"]),
        float(best_row["adversarial_weight"]),
        disc_str,
        float(best_row.get("best_val_pr_auc", 0.0)),
        float(best_row.get("best_val_f1", 0.0)),
    )


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    params_path: Path | None = None,
    generate_plots: bool = True,
    is_aae: bool = False,
) -> dict[str, Any]:
    """Run the full analysis pipeline for AE or AAE validation results.

    Parameters
    ----------
    csv_path : Path or None
        Path to validation results CSV. Defaults to reports/tables/.
    output_dir : Path or None
        Directory for sensitivity plots. Defaults to reports/figures/.
    params_path : Path or None
        Output path for params_validated_{ae|aae}.yaml.
    generate_plots : bool
        Whether to generate sensitivity plots.
    is_aae : bool
        If True, treat the CSV as AAE validation results.

    Returns
    -------
    dict
        Summary with best_row, n_runs, output_paths.
    """
    config = load_config()
    specs = _get_specs(is_aae)

    if csv_path is None:
        csv_path = get_default_results_path(config, is_aae=is_aae)
    if output_dir is None:
        output_dir = get_default_output_dir(config)
    if params_path is None:
        params_path = get_default_params_path(is_aae=is_aae)

    if not csv_path.exists():
        logger.error("Results CSV not found: %s", csv_path)
        if is_aae:
            logger.error("Run: python -m src.validation.run_search_aae --n-iter 25 --seed 42")
        else:
            logger.error("Run: python -m src.validation.run_search_ae --n-iter 20 --seed 42")
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    # --- Load and analyze ---
    rows = load_results_csv(csv_path, is_aae=is_aae)
    if not rows:
        raise ValueError(f"No valid rows in {csv_path}")

    best_row = select_best_config(rows)

    mode_name = "AAE" if is_aae else "AE"
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS SUMMARY (%s) — %d runs", mode_name, len(rows))
    logger.info("=" * 60)

    if is_aae:
        for row in rows:
            flag = " <-- BEST" if row == best_row else ""
            logger.info(
                "  Run %2d | recon_w=%.2f adv_w=%.3f disc=%-15s | "
                "pr_auc=%.4f f1=%.4f epoch=%-3d%s",
                row["run_id"],
                row["reconstruction_weight"],
                row["adversarial_weight"],
                row["discriminator_hidden_layers"],
                row["best_val_pr_auc"], row["best_val_f1"], row["best_epoch"], flag,
            )
    else:
        for row in rows:
            flag = " <-- BEST" if row == best_row else ""
            logger.info(
                "  Run %2d | W=%-2d latent=%-2d channels=%-15s | "
                "pr_auc=%.4f f1=%.4f epoch=%-3d%s",
                row["run_id"], row["W"], row["latent_dim"], row["encoder_channels"],
                row["best_val_pr_auc"], row["best_val_f1"], row["best_epoch"], flag,
            )

    logger.info("=" * 60)
    if is_aae:
        logger.info(
            "HP_AAE_best: recon_w=%.2f, adv_w=%.3f, disc=%s",
            float(best_row["reconstruction_weight"]),
            float(best_row["adversarial_weight"]),
            best_row["discriminator_hidden_layers"],
        )
    else:
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
        plot_paths = generate_sensitivity_plots(rows, output_dir, is_aae=is_aae)

    # --- Write validated params ---
    if is_aae:
        write_validated_params_aae(best_row, config, params_path)
    else:
        write_validated_params_ae(best_row, config, params_path)

    return {
        "best_row": best_row,
        "n_runs": len(rows),
        "csv_path": str(csv_path),
        "plot_paths": [str(p) for p in plot_paths],
        "params_path": str(params_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation results and select best config (AE or AAE)"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to validation results CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for sensitivity plots (default: reports/figures/)",
    )
    parser.add_argument(
        "--params-out", type=Path, default=None,
        help="Output path for params_validated_{ae|aae}.yaml",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating sensitivity plots",
    )
    parser.add_argument(
        "--aae", action="store_true",
        help="Analyze AAE results (default: AE)",
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
        is_aae=args.aae,
    )


if __name__ == "__main__":
    main()
