"""PyTorch ``Dataset`` for Kuka robot time-series data.

:class:`KukaDataset` wraps a 2-D array and extracts sliding windows of
``W`` consecutive timesteps **on the fly** in ``__getitem__``.

Why on-the-fly (see §4.1): pre-computing all windows at W=16 costs ~2.9 GB;
slicing views costs ~0.18 GB.  Separate instances per split prevent windows
from straddling the normal/anomaly boundary.  The training loop must transpose
the returned ``(W, n_features)`` → ``(B, n_features, W)`` for ``Conv1d``.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class KukaDataset(Dataset):
    """Sliding-window ``Dataset`` for Kuka robot data.

    Each ``__getitem__`` returns ``(window, label)`` where window has
    shape ``(window_size, n_features)``.  The model transposes to
    ``(B, n_features, W)`` for ``Conv1d``.  ``label`` is implicit:
    ``0 = normal``, ``1 = anomaly`` — alignment guaranteed at construction.

    The number of windows is ``(n_samples - window_size) // stride + 1``.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        window_size: int = 16,
        stride: int = 1,
        drop_incomplete: bool = True,
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

        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self.data = torch.from_numpy(np.ascontiguousarray(data)).to(dtype)
        self.n_features = n_features
        self.window_size = window_size
        self.stride = stride
        self.label = label
        self.dtype = dtype

        # --- Compute number of windows ---
        if not drop_incomplete:
            warnings.warn(
                "drop_incomplete=False is not fully supported yet — "
                "partial windows would need padding. Falling back to "
                "drop_incomplete=True behaviour.",
                UserWarning,
            )

        if n_samples < window_size:
            self.n_windows = 0
        else:
            # Number of full windows that fit with the given stride.
            self.n_windows = (n_samples - window_size) // stride + 1

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

        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]  # (W, n_features) — view, no copy

        label_tensor = torch.tensor(
            [self.label], dtype=self.dtype
        )

        return window, label_tensor

    def __repr__(self) -> str:
        return (
            f"KukaDataset(n_windows={self.n_windows}, "
            f"window_size={self.window_size}, stride={self.stride}, "
            f"n_features={self.n_features}, label={self.label})"
        )


# ---------------------------------------------------------------------------
# Convenience: build all four datasets from processed .npy files
# ---------------------------------------------------------------------------
def build_datasets(
    processed_dir: str | Path,
    window_size: int = 16,
    stride: int = 1,
    batch_size: int = 256,
    num_workers: int = 2,
) -> dict[str, torch.utils.data.DataLoader]:
    """Load processed .npy files and create DataLoaders for train/val/test.

    Reads ``train.npy``, ``val.npy``, ``test_normal.npy``,
    ``test_anomaly.npy`` from ``processed_dir``.  ``train`` is shuffled;
    ``val`` and ``test`` are not.  The ``test`` loader concatenates
    ``test_normal`` (label 0) and ``test_anomaly`` (label 1) via
    ``ConcatDataset`` so evaluation runs in a single pass.
    """
    from torch.utils.data import ConcatDataset, DataLoader

    processed_dir = Path(processed_dir)

    train_data = np.load(processed_dir / "train.npy")
    val_data = np.load(processed_dir / "val.npy")
    test_normal_data = np.load(processed_dir / "test_normal.npy")
    test_anomaly_data = np.load(processed_dir / "test_anomaly.npy")

    train_ds = KukaDataset(train_data, window_size, stride, label=0)
    val_ds = KukaDataset(val_data, window_size, stride, label=0)
    test_normal_ds = KukaDataset(test_normal_data, window_size, stride, label=0)
    test_anomaly_ds = KukaDataset(test_anomaly_data, window_size, stride, label=1)

    test_ds = ConcatDataset([test_normal_ds, test_anomaly_ds])

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=False,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        ),
    }

    return loaders
