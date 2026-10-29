"""Analyze validation search results, select the best hyperparameter config,
generate sensitivity plots, and write params_validated_{ae|aae}.yaml.

Supports both:
- Baseline AE (with multi-seed aggregation over W and latent_dim)
- Adversarial AE (AAE with reconstruction_weight, adversarial_weight, discriminator_hidden_layers)

Usage:
    python -m src.validation.analyze_results [--aae] [--input PATH] [--no-plots]
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
import io
import logging
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AE Aggregated Configuration & Helpers
# ---------------------------------------------------------------------------
@dataclass
class AggregatedConfig:
    """Aggregated metrics for one hyperparameter configuration across multiple seeds."""
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
    n_seeds: int
    seeds: list[int]


def get_default_results_path(config: dict, is_aae: bool = False) -> Path:
    tables_dir = Path(config.get("paths", {}).get("reports_tables", "reports/tables/"))
    fname = "validation_results_aae.csv" if is_aae else "validation_results_ae.csv"
    return tables_dir / fname


def get_default_output_dir(config: dict) -> Path:
    return Path(config.get("paths", {}).get("reports_figures", "reports/figures/"))


def get_default_params_path(is_aae: bool = False) -> Path:
    fname = "params_validated_aae.yaml" if is_aae else "params_validated_ae.yaml"
    return Path("config") / fname


def load_results_csv(csv_path: Path, is_aae: bool = False) -> list[dict[str, Any]]:
    """Load validation results from CSV with robust error and encoding handling."""
    rows: list[dict[str, Any]] = []
    n_failed = 0
    n_malformed = 0
    n_nan_inf = 0

    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    file_content = None
    used_encoding = None

    for enc in encodings:
        try:
            with open(csv_path, "r", newline="", encoding=enc) as f:
                file_content = f.read()
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if file_content is None:
        logger.error("Failed to read CSV with any encoding: %s", encodings)
        return rows

    reader = csv.DictReader(io.StringIO(file_content))
    for row_idx, row in enumerate(reader, start=1):
        row_data: dict[str, Any] = dict(row)
        try:
            row_data["run_id"] = int(row_data["run_id"])
            if row_data.get("config_id"):
                row_data["config_id"] = int(row_data["config_id"])

            row_data["best_epoch"] = int(float(row_data["best_epoch"]))
            train_time = float(row_data.get("train_time_sec", 0.0))
            if not math.isfinite(train_time):
                n_nan_inf += 1
                continue
            row_data["train_time_sec"] = train_time
            row_data["seed"] = int(row_data.get("seed", 42))

            if is_aae:
                row_data["reconstruction_weight"] = float(row_data["reconstruction_weight"])
                row_data["adversarial_weight"] = float(row_data["adversarial_weight"])
                row_data["discriminator_hidden_layers"] = str(row_data["discriminator_hidden_layers"])
                pr_auc = float(row_data["best_val_pr_auc"])
                if not math.isfinite(pr_auc) or pr_auc < 0:
                    n_failed += 1
                    continue
                row_data["best_val_pr_auc"] = pr_auc
                row_data["best_val_f1"] = float(row_data.get("best_val_f1", 0.0))
            else:
                row_data["W"] = int(row_data["W"])
                row_data["latent_dim"] = int(row_data["latent_dim"])

                # Handle different column names between AE runs
                pr_key = "val_pr_auc" if "val_pr_auc" in row_data else "best_val_pr_auc"
                roc_key = "val_roc_auc" if "val_roc_auc" in row_data else "best_val_roc_auc"

                val_pr_auc = float(row_data[pr_key])
                val_roc_auc = float(row_data[roc_key]) if roc_key in row_data and row_data[roc_key] != "" else 0.0

                if not (math.isfinite(val_pr_auc) and math.isfinite(val_roc_auc)):
                    n_nan_inf += 1
                    continue
                if val_pr_auc < 0 or val_roc_auc < 0:
                    n_failed += 1
                    continue

                row_data["best_val_pr_auc"] = val_pr_auc
                row_data["val_pr_auc"] = val_pr_auc
                row_data["val_roc_auc"] = val_roc_auc

            rows.append(row_data)
        except (ValueError, KeyError) as e:
            n_malformed += 1
            logger.warning("Row %d: malformed - %s (raw: %s)", row_idx, e, dict(row))
            continue

    if n_failed:
        logger.warning("Skipped %d failed run(s)", n_failed)
    if n_nan_inf:
        logger.warning("Skipped %d row(s) with NaN/Inf values", n_nan_inf)
    if n_malformed:
        logger.warning("Skipped %d malformed row(s)", n_malformed)

    if not rows:
        logger.error("No valid rows found in %s", csv_path)
    else:
        logger.info("Loaded %d valid rows from %s (encoding: %s)", len(rows), csv_path, used_encoding)
    return rows


def aggregate_by_config(rows: list[dict[str, Any]]) -> list[AggregatedConfig]:
    """Group rows by (W, latent_dim) and compute mean +/- std."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["W"], r["latent_dim"])
        groups.setdefault(key, []).append(r)

    aggregated: list[AggregatedConfig] = []
    for config_id, ((w, latent), group) in enumerate(groups.items(), start=1):
        pr_aucs = [g["best_val_pr_auc"] for g in group]
        roc_aucs = [g.get("val_roc_auc", 0.0) for g in group]
        epochs = [g["best_epoch"] for g in group]
        times = [g["train_time_sec"] for g in group]
        seeds = [g["seed"] for g in group]
        n = len(group)

        ddof = 1 if n > 1 else 0
        aggregated.append(
            AggregatedConfig(
                config_id=config_id,
                W=w,
                latent_dim=latent,
                mean_val_pr_auc=float(np.mean(pr_aucs)),
                std_val_pr_auc=float(np.std(pr_aucs, ddof=ddof)),
                mean_val_roc_auc=float(np.mean(roc_aucs)),
                std_val_roc_auc=float(np.std(roc_aucs, ddof=ddof)),
                mean_best_epoch=float(np.mean(epochs)),
                std_best_epoch=float(np.std(epochs, ddof=ddof)),
                mean_train_time_sec=float(np.mean(times)),
                n_seeds=n,
                seeds=seeds,
            )
        )
    return aggregated


def select_best_config_ae(aggregated: list[AggregatedConfig]) -> AggregatedConfig:
    """Select best AE config: max mean PR-AUC, min std PR-AUC, max mean ROC-AUC, min epoch."""
    return max(
        aggregated,
        key=lambda c: (
            c.mean_val_pr_auc,
            -c.std_val_pr_auc,
            c.mean_val_roc_auc,
            -c.mean_best_epoch,
        ),
    )


def select_best_config_aae(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best AAE config: max best_val_pr_auc, max best_val_f1, min best_epoch."""
    return max(
        rows,
        key=lambda r: (
            r["best_val_pr_auc"],
            r.get("best_val_f1", 0.0),
            -r["best_epoch"],
        ),
    )


def generate_sensitivity_plots_ae(
    aggregated: list[AggregatedConfig],
    output_dir: Path,
    best: AggregatedConfig | None = None,
) -> list[Path]:
    """Generate sensitivity plots for AE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    specs = [
        ("W", "ae_W", "Sensitivity to Window Size (W)", "Window Size W (timesteps)"),
        ("latent_dim", "ae_latent_dim", "Sensitivity to Latent Dimension", "Latent Dimension"),
    ]

    for hparam_col, suffix, title, xlabel in specs:
        groups: dict[Any, list[float]] = {}
        for agg in aggregated:
            val = getattr(agg, hparam_col)
            groups.setdefault(val, []).append(agg.mean_val_pr_auc)

        sorted_keys = sorted(groups.keys())
        means = np.array([float(np.mean(groups[k])) for k in sorted_keys], dtype=float)
        stds = np.array(
            [float(np.std(groups[k], ddof=1)) if len(groups[k]) > 1 else 0.0
             for k in sorted_keys],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
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
        ax.set_ylabel("Mean PR-AUC (validation)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

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


def generate_sensitivity_plots_aae(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate sensitivity plots for AAE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    specs = [
        ("reconstruction_weight", "aae_reconstruction_weight",
         "Sensitivity to Reconstruction Weight", "Reconstruction Weight"),
        ("adversarial_weight", "aae_adversarial_weight",
         "Sensitivity to Adversarial Weight", "Adversarial Weight"),
        ("discriminator_hidden_layers", "aae_discriminator_hidden_layers",
         "Sensitivity to Discriminator Hidden Layers", "Hidden Layers"),
    ]

    for hparam_col, suffix, title, xlabel in specs:
        groups: dict[Any, list[float]] = {}
        for r in rows:
            val = r[hparam_col]
            groups.setdefault(val, []).append(r["best_val_pr_auc"])

        try:
            sorted_keys = sorted(groups.keys(), key=float)
        except (ValueError, TypeError):
            sorted_keys = sorted(groups.keys(), key=str)

        means = [np.mean(groups[k]) for k in sorted_keys]
        stds = [np.std(groups[k]) for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            [str(k) for k in sorted_keys],
            means,
            yerr=stds,
            capsize=5,
            color="#0ea5e9",
            edgecolor="#0284c7",
            alpha=0.85,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Mean PR-AUC (validation)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        save_path = output_dir / f"sensitivity_{suffix}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Generated %s", save_path)
        generated.append(save_path)

    return generated


def write_validated_params_ae(
    best: AggregatedConfig,
    base_config: dict,
    output_path: Path,
) -> None:
    """Write params_validated_ae.yaml with the best hyperparameters."""
    validated = copy.deepcopy(base_config)
    validated["data"]["window_size"] = best.W
    validated["model"]["latent_dim"] = best.latent_dim

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated AE params to %s", output_path)


def write_validated_params_aae(
    best_row: dict[str, Any],
    base_config: dict,
    output_path: Path,
) -> None:
    """Write params_validated_aae.yaml with the best AAE hyperparameters."""
    validated = copy.deepcopy(base_config)
    validated["training"]["reconstruction_weight"] = float(best_row["reconstruction_weight"])
    validated["training"]["adversarial_weight"] = float(best_row["adversarial_weight"])

    disc_str = str(best_row["discriminator_hidden_layers"])
    disc_layers = eval(disc_str)  # noqa: S306
    validated["model"]["discriminator"]["hidden_layers"] = disc_layers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(validated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Wrote validated AAE params to %s", output_path)


def run_analysis(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    params_path: Path | None = None,
    generate_plots: bool = True,
    is_aae: bool = False,
) -> dict[str, Any]:
    """Run the analysis pipeline."""
    config = load_config()

    if csv_path is None:
        csv_path = get_default_results_path(config, is_aae=is_aae)
    if output_dir is None:
        output_dir = get_default_output_dir(config)
    if params_path is None:
        params_path = get_default_params_path(is_aae=is_aae)

    if not csv_path.exists():
        logger.error("Results CSV not found: %s", csv_path)
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    rows = load_results_csv(csv_path, is_aae=is_aae)
    if not rows:
        raise ValueError(f"No valid rows in {csv_path}")

    plot_paths: list[Path] = []

    if is_aae:
        best_row = select_best_config_aae(rows)
        logger.info("=" * 70)
        logger.info("AAE VALIDATION RESULTS SUMMARY: %d runs", len(rows))
        logger.info("=" * 70)
        for r in rows:
            flag = " <-- BEST" if r == best_row else ""
            logger.info(
                "  Run %2d | recon_w=%.2f adv_w=%.3f disc=%-12s | pr_auc=%.4f f1=%.4f epoch=%-3d%s",
                r["run_id"], r["reconstruction_weight"], r["adversarial_weight"],
                r["discriminator_hidden_layers"], r["best_val_pr_auc"], r["best_val_f1"],
                r["best_epoch"], flag,
            )
        logger.info("=" * 70)
        logger.info(
            "HP_AAE_best: recon_w=%.2f, adv_w=%.3f, disc=%s | pr_auc=%.4f",
            best_row["reconstruction_weight"], best_row["adversarial_weight"],
            best_row["discriminator_hidden_layers"], best_row["best_val_pr_auc"],
        )

        if generate_plots:
            plot_paths = generate_sensitivity_plots_aae(rows, output_dir)
        write_validated_params_aae(best_row, config, params_path)

        return {
            "best_config": best_row,
            "n_runs": len(rows),
            "csv_path": str(csv_path),
            "plot_paths": [str(p) for p in plot_paths],
            "params_path": str(params_path),
        }
    else:
        aggregated = aggregate_by_config(rows)
        best = select_best_config_ae(aggregated)

        logger.info("=" * 70)
        logger.info("AE VALIDATION RESULTS SUMMARY: %d runs, %d configs", len(rows), len(aggregated))
        logger.info("=" * 70)
        for agg in aggregated:
            flag = " <-- BEST" if agg == best else ""
            logger.info(
                "  Config %2d | W=%-2d latent=%-2d | pr_auc=%.4f +/- %.4f | epoch=%.1f%s",
                agg.config_id, agg.W, agg.latent_dim,
                agg.mean_val_pr_auc, agg.std_val_pr_auc, agg.mean_best_epoch, flag,
            )
        logger.info("=" * 70)
        logger.info(
            "HP_AE_best: W=%d, latent_dim=%d | pr_auc=%.4f +/- %.4f",
            best.W, best.latent_dim, best.mean_val_pr_auc, best.std_val_pr_auc,
        )

        if generate_plots:
            plot_paths = generate_sensitivity_plots_ae(aggregated, output_dir, best=best)
        write_validated_params_ae(best, config, params_path)

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
