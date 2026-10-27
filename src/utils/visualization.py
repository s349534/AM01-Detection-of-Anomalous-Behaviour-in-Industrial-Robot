"""Visualization utilities for anomaly detection.

This module provides:
- plot_confusion_matrix: Visualize 2x2 confusion matrix
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for CI/HPC
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm_array: np.ndarray,
    labels: tuple[str, str] = ("normal", "anomaly"),
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (6, 5),
    normalize: bool = False,
) -> plt.Figure:
    """Plot a 2x2 confusion matrix using sklearn's ConfusionMatrixDisplay.

    Parameters
    ----------
    cm_array : np.ndarray
        2x2 confusion matrix array [[TN, FP], [FN, TP]].
    labels : tuple[str, str]
        Class labels (negative, positive).
    save_path : str | Path | None
        If provided, save figure to this path.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size (width, height).
    normalize : bool
        If True, normalize the matrix by true class (row-wise).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Ensure we have a proper 2x2 array
    cm_array = np.asarray(cm_array, dtype=float).reshape(2, 2)

    if normalize:
        cm_display = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm_array.astype(int)
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, values_format=fmt, colorbar=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to %s", save_path)

    plt.close(fig)
    return fig