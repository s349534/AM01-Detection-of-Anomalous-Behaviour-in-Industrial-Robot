"""Visualization utilities for anomaly detection.

This module provides:
- plot_training_history: Loss curves during training
- plot_reconstruction: Visualize original vs reconstructed data
- plot_anomalies: Highlight detected anomalies
- plot_roc_curve: Visualize ROC curve
- plot_precision_recall_curve: Visualize PR curve
- plot_error_distribution: Compare error distributions per class
- plot_sensitivity: Bar/point plots for hyperparameter sensitivity analysis
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/HPC
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot training and validation loss curves.

    Parameters
    ----------
    history : dict
        Dictionary with ``"train_loss"`` and ``"val_loss"`` keys (lists per epoch).
    save_path : str | Path | None
        If provided, save figure to this path.
    figsize : tuple
        Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label="Train Loss", color="#2563eb", linewidth=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation Loss", color="#ef4444", linewidth=2)
        # Mark best epoch
        best_epoch = int(np.argmin(val_loss)) + 1
        best_val = val_loss[best_epoch - 1]
        ax.scatter([best_epoch], [best_val], color="#f59e0b", s=100, zorder=5,
                   label=f"Best (epoch {best_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Training History", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved training history plot to %s", save_path)

    plt.close(fig)
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: str | Path | None = None,
    label: str = "Model",
) -> plt.Figure:
    """Plot ROC curve.

    Parameters
    ----------
    fpr : array
        False Positive Rates.
    tpr : array
        True Positive Rates.
    roc_auc : float
        Area under the ROC curve.
    save_path : str | Path | None
        If provided, save figure to this path.
    label : str
        Label for the curve legend.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", color="#9ca3af", linewidth=1, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ROC curve to %s", save_path)

    plt.close(fig)
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    save_path: str | Path | None = None,
    label: str = "Model",
) -> plt.Figure:
    """Plot Precision-Recall curve.

    Parameters
    ----------
    precision : array
        Precision values.
    recall : array
        Recall values.
    pr_auc : float
        Average precision (PR-AUC).
    save_path : str | Path | None
        If provided, save figure to this path.
    label : str
        Label for the curve legend.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(recall, precision, color="#10b981", linewidth=2,
            label=f"{label} (PR-AUC = {pr_auc:.4f})")

    # F1 iso-curves
    f1_scores = np.linspace(0.1, 0.9, 9)
    for f1 in f1_scores:
        p = np.linspace(0.01, 1, 100)
        r = f1 * p / (2 * p - f1)
        r = r[r <= 1]
        p_valid = p[:len(r)]
        if len(r) > 0:
            ax.plot(r, p_valid, "--", color="#d1d5db", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved PR curve to %s", save_path)

    plt.close(fig)
    return fig


def plot_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: float | None = None,
    save_path: str | Path | None = None,
    title: str = "Reconstruction Error Distribution",
) -> plt.Figure:
    """Plot histogram of reconstruction errors per class.

    Parameters
    ----------
    normal_errors : array
        Reconstruction errors for normal samples.
    anomaly_errors : array
        Reconstruction errors for anomaly samples.
    threshold : float or None
        Decision threshold to draw as vertical line.
    save_path : str | Path | None
        If provided, save figure to this path.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(normal_errors.min(), anomaly_errors.min()),
        max(normal_errors.max(), anomaly_errors.max()),
        60,
    )

    ax.hist(normal_errors, bins=bins, alpha=0.6, color="#2563eb", label="Normal", density=True)
    ax.hist(anomaly_errors, bins=bins, alpha=0.6, color="#ef4444", label="Anomaly", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="#f59e0b", linestyle="--", linewidth=2,
                   label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved error distribution plot to %s", save_path)

    plt.close(fig)
    return fig


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list[str] | None = None,
    save_path: str | Path | None = None,
    max_samples: int = 100,
) -> plt.Figure:
    """Visualize original vs reconstructed sequences.

    Parameters
    ----------
    original : array of shape (n_samples, n_timesteps)
        Original time-series data (single feature or aggregate).
    reconstructed : array, same shape
        Reconstructed data.
    feature_names : list[str] or None
        Names of features (for title).
    save_path : str | Path | None
        If provided, save figure to this path.
    max_samples : int
        Maximum number of samples to plot.
    """
    n_samples = min(len(original), max_samples)
    fig, axes = plt.subplots(
        n_samples, 1, figsize=(12, 2 * n_samples), sharex=True,
        squeeze=False,
    )

    for i in range(n_samples):
        axes[i, 0].plot(original[i], color="#2563eb", alpha=0.7, label="Original")
        axes[i, 0].plot(reconstructed[i], color="#ef4444", alpha=0.7, label="Reconstructed")
        axes[i, 0].legend(fontsize=8)
        if feature_names:
            axes[i, 0].set_title(feature_names[i] if i < len(feature_names) else f"Sample {i}",
                               fontsize=9)

    fig.suptitle("Original vs Reconstructed", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved reconstruction plot to %s", save_path)

    plt.close(fig)
    return fig


def plot_anomalies(
    timestamps: np.ndarray,
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str | Path | None = None,
    title: str = "Anomaly Scores Over Time",
) -> plt.Figure:
    """Plot reconstruction errors over time with anomaly highlights.

    Parameters
    ----------
    timestamps : array
        Time indices for x-axis.
    errors : array
        Reconstruction errors at each timestep.
    labels : array
        Binary labels (0=normal, 1=anomaly).
    threshold : float
        Decision threshold.
    save_path : str | Path | None
        If provided, save figure to this path.
    title : str
        Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: error scores
    ax1.plot(timestamps, errors, color="#3b82f6", linewidth=0.8, alpha=0.8)
    ax1.axhline(threshold, color="#f59e0b", linestyle="--", linewidth=1.5,
                label=f"Threshold = {threshold:.4f}")

    # Highlight anomaly regions
    anomaly_mask = labels == 1
    if anomaly_mask.any():
        ax1.fill_between(timestamps, 0, errors.max() * 1.1,
                         where=anomaly_mask, alpha=0.2, color="#ef4444",
                         label="Anomaly regions")

    ax1.set_ylabel("Reconstruction Error", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: binary labels
    ax2.fill_between(timestamps, 0, labels, step="mid", alpha=0.4,
                     color="#ef4444", label="Anomaly")
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.set_ylabel("Label", fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved anomaly plot to %s", save_path)

    plt.close(fig)
    return fig


def plot_sensitivity(
    df_data: dict[str, list],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a sensitivity analysis plot (point plot with error bars).

    Parameters
    ----------
    df_data : dict
        Dictionary with keys as column names, values as lists.
    x_key : str
        Column name for x-axis.
    y_key : str
        Column name for y-axis (metric to plot).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str | Path | None
        If provided, save figure to this path.
    """
    from collections import defaultdict

    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = df_data[x_key]
    y_vals = df_data[y_key]

    # Group by x value and compute mean ± std
    groups: dict[Any, list[float]] = defaultdict(list)
    for x, y in zip(x_vals, y_vals):
        groups[x].append(y)

    sorted_x = sorted(groups.keys(), key=lambda v: (isinstance(v, str), v))
    means = [np.mean(groups[x]) for x in sorted_x]
    stds = [np.std(groups[x]) if len(groups[x]) > 1 else 0.0 for x in sorted_x]

    x_positions = range(len(sorted_x))
    ax.errorbar(
        x_positions, means, yerr=stds,
        fmt="o-", color="#2563eb", markersize=8, linewidth=2,
        capsize=5, capthick=1.5,
        ecolor="#ef4444",
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([str(x) for x in sorted_x], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved sensitivity plot to %s", save_path)

    plt.close(fig)
    return fig
