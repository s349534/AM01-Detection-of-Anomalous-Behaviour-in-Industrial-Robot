"""PyTorch ``Dataset`` for Kuka robot time-series data.

:class:`KukaDataset` wraps a 2-D array and extracts windows at specified
start indices **on the fly** in ``__getitem__``.

Why on-the-fly (see §4.1): pre-computing all windows costs GB;
slicing views costs ~0.18 GB.  The start indices are precomputed by
:mod:`src.data.partitioning` to enforce:
  - no overlap between partitions (R2a)
  - guard band gap ≥ W between partitions (R2b)
  - controlled stride within each partition (R1)

The ``build_dataloaders`` function loads full scaled recordings and
recomputes partition start indices for the current window_size,
enabling hyperparameter search over different window sizes without
re-running preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.partitioning import plan_from_config

logger = logging.getLogger(__name__)


class KukaDataset(Dataset):
    """Window-index-based ``Dataset`` for Kuka robot data.

    Each ``__getitem__`` returns ``(window, label)`` where window has
    shape ``(window_size, n_features)``.  The model transposes to
    ``(B, n_features, W)`` for ``Conv1d``.  ``label`` is implicit:
    ``0 = normal``, ``1 = anomaly`` — alignment guaranteed by the partition plan.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        start_indices: np.ndarray,
        window_size: int,
        label: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if data.ndim != 2:
            raise ValueError(
                f"data must be 2-D (n_samples, n_features); "
                f"got shape {data.shape} (ndim={data.ndim})"
            )

        n_samples, n_features = data.shape

        if n_samples == 0:
            raise ValueError("data is empty — cannot create dataset")

        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        if start_indices.ndim != 1:
            raise ValueError("start_indices must be 1-D array")

        if len(start_indices) > 0:
            max_start = int(start_indices.max())
            if max_start + window_size > n_samples:
                raise ValueError(
                    f"start_indices + window_size exceeds data length: "
                    f"max start={max_start}, W={window_size}, n_samples={n_samples}"
                )

        self.data = torch.from_numpy(np.ascontiguousarray(data)).to(dtype)
        self.start_indices = torch.from_numpy(start_indices.astype(np.int64))
        self.n_features = n_features
        self.window_size = window_size
        self.label = label
        self.dtype = dtype
        self.n_windows = len(start_indices)

        if self.n_windows == 0:
            logger.warning(
                "KukaDataset created with 0 windows: data length=%d, window_size=%d. "
                "DataLoader will yield no batches.",
                n_samples, window_size
            )

    def __len__(self) -> int:
        """Number of windows in this dataset."""
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(window, label)`` — window is a view, label is a 1-element tensor."""
        if idx < 0:
            idx += self.n_windows
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(
                f"index {idx} is out of range for {self.n_windows} windows"
            )

        start = int(self.start_indices[idx])
        end = start + self.window_size
        window = self.data[start:end]  # (W, n_features) — view, no copy

        label_tensor = torch.tensor([self.label], dtype=self.dtype)

        return window, label_tensor

    def __repr__(self) -> str:
        return (
            f"KukaDataset(n_windows={self.n_windows}, "
            f"window_size={self.window_size}, "
            f"n_features={self.n_features}, label={self.label})"
        )


# ---------------------------------------------------------------------------
# Factory: build all loaders from scaled recordings + on-the-fly windowing
# ---------------------------------------------------------------------------
def build_dataloaders(
    config: dict[str, Any],
    partitions: list[str] | None = None,
    batch_size: int = 256,
    num_workers: int = 4,
    use_prebuilt_partitions: bool = True,
) -> dict[str, DataLoader]:
    """Build DataLoaders for the specified partitions.

    Loads the full scaled recordings (``normal_scaled.npy``, ``slow_scaled.npy``)
    and recomputes partition start indices for the current ``config.window_size``
    using the saved partitioning parameters. Windows are extracted on-the-fly
    by :class:`KukaDataset` during iteration.

    Args:
        config: Full config dict.
        partitions: List of partition names to build loaders for.
                    Default: all five partitions.
        batch_size: Batch size.
        num_workers: DataLoader worker processes.
        use_prebuilt_partitions: Kept for API compatibility (ignored).

    Returns:
        Dict mapping partition name -> DataLoader.

    Raises:
        FileNotFoundError: If required scaled recording files are missing.
    """
    processed_dir = Path(config.get("paths", {}).get("data_processed", "data/processed/"))
    window_size = int(config.get("data", {}).get("window_size", 256))

    if partitions is None:
        partitions = ["train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"]

    # Map partition names to source recording and label
    partition_map = {
        "train": ("normal", 0),
        "val_normal": ("normal", 0),
        "val_anomaly": ("slow", 1),
        "test_normal": ("normal", 0),
        "test_anomaly": ("slow", 1),
    }

    # Load full scaled recordings
    normal_scaled_path = processed_dir / "normal_scaled.npy"
    slow_scaled_path = processed_dir / "slow_scaled.npy"
    if not normal_scaled_path.exists() or not slow_scaled_path.exists():
        raise FileNotFoundError(
            f"Scaled recordings not found in {processed_dir}. "
            f"Run preprocessing first: python -m src.data.preprocessing"
        )
    normal_scaled = np.load(normal_scaled_path)
    slow_scaled = np.load(slow_scaled_path)

    # Recompute partition plan for current window_size
    # We need the original recording lengths from the scaled data
    normal_n = normal_scaled.shape[0]
    slow_n = slow_scaled.shape[0]
    plan = plan_from_config(config, "normal", normal_n, slow_n)
    slow_plan = plan_from_config(config, "slow", normal_n, slow_n)
    plan.update(slow_plan)

    loaders: dict[str, DataLoader] = {}

    for name in partitions:
        if name not in partition_map:
            raise ValueError(f"Partition '{name}' not recognized")
        source_name, label = partition_map[name]
        starts = plan.get(name, np.array([], dtype=np.int64))

        data = normal_scaled if source_name == "normal" else slow_scaled

        # Create KukaDataset with on-the-fly window extraction
        ds = KukaDataset(data, starts, window_size, label=label)
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=False,
        )
        logger.info("Built loader for %s: %d windows, label=%d", name, len(ds), label)

    return loaders


# ---------------------------------------------------------------------------
# Single-loader convenience (used by training/validation scripts)
# ---------------------------------------------------------------------------
def build_loader(
    data: np.ndarray | torch.Tensor,
    start_indices: np.ndarray,
    window_size: int,
    batch_size: int = 256,
    label: int = 0,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a single ``DataLoader`` from data array and window start indices.

    Wraps :class:`KukaDataset` + ``DataLoader`` in one call.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The raw data (already normalised).
    start_indices : 1-D array of int
        Window start indices (absolute in the data array).
    window_size : int
        Number of timesteps per window.
    batch_size : int, default=256
    label : int, default=0
        Label to assign to every window (0 = normal, 1 = anomaly).
    shuffle : bool, default=False
        Whether to shuffle the data each epoch.
    num_workers : int, default=0
        DataLoader worker processes (``0`` = load in main process).
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.

    Returns
    -------
    DataLoader
        A ready-to-use data loader.
    """
    ds = KukaDataset(data, start_indices, window_size, label=label)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
    )