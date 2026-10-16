"""Tests for Fase 2 preprocessing + dataset pipeline (33 tests).

Run: uv run pytest tests/test_dataset.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import KukaDataset
from src.data.preprocessing import (
    load_kuka_data,
    normalize_data,
    split_temporal_data,
)
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def config() -> dict:
    """Load merged config (config.yaml + params.yaml)."""
    return load_config()


@pytest.fixture(scope="module")
def loaded(config):
    """Load raw Kuka data: (normal, slow, selected_cols)."""
    normal, slow, cols = load_kuka_data(config)
    return normal, slow, cols


@pytest.fixture(scope="module")
def splits(loaded, config):
    """Split the loaded data temporally (70/15/15)."""
    normal, slow, selected_cols = loaded
    data_splits = split_temporal_data(normal, slow, config)
    return data_splits, selected_cols


@pytest.fixture(scope="module")
def normalised(splits, config):
    """MinMaxScaler the splits and return (scaled_dict, scaler)."""
    data_splits, selected_cols = splits
    return normalize_data(data_splits, config, selected_cols)


# ---------------------------------------------------------------------------
# load_kuka_data
# ---------------------------------------------------------------------------
class TestLoadKukaData:
    def test_normal_shape(self, loaded):
        normal, _, _ = loaded
        # After load: 86 features (1 action + 85 numerical)
        assert normal.shape == (233792, 86), f"Expected (233792, 86), got {normal.shape}"

    def test_slow_shape(self, loaded):
        _, slow, _ = loaded
        assert slow.shape == (41538, 86), f"Expected (41538, 86), got {slow.shape}"

    def test_selected_cols_shape(self, loaded):
        _, _, cols = loaded
        assert cols.shape == (86,), f"Expected (86,), got {cols.shape}"

    def test_columns_aligned(self, loaded):
        """Both arrays must have the same number of features."""
        normal, slow, cols = loaded
        assert normal.shape[1] == slow.shape[1] == 86
        assert len(cols) == 86

    def test_anomaly_dropped(self, loaded):
        """The 'anomaly' column must not be among the retained columns."""
        _, _, cols = loaded
        assert "anomaly" not in cols

    def test_dtype(self, loaded):
        normal, slow, _ = loaded
        assert normal.dtype == np.float32
        assert slow.dtype == np.float32


# ---------------------------------------------------------------------------
# split_temporal_data
# ---------------------------------------------------------------------------
class TestSplitTemporal:
    def test_proportions(self, splits):
        data_splits, _ = splits
        n_total = 233792
        train = data_splits["train"]
        val_normal = data_splits["val_normal"]
        test_normal = data_splits["test_normal"]
        expected_train = int(0.70 * n_total)  # 163654
        expected_val = int(0.15 * n_total)    # 35068

        assert train.shape[0] == expected_train
        assert val_normal.shape[0] == expected_val
        assert train.shape[0] + val_normal.shape[0] + test_normal.shape[0] == n_total

    def test_no_shuffle(self, loaded, config):
        """train[0] must equal the first row of the full normal array."""
        normal, slow, _ = loaded
        data_splits = split_temporal_data(normal, slow, config)
        np.testing.assert_array_equal(
            data_splits["train"][0], normal[0],
            err_msg="First train row must match first normal row (no shuffle)",
        )

    def test_val_anomaly_is_half_slow(self, loaded, config):
        """val_anomaly must contain 50% of KukaSlow."""
        normal, slow, _ = loaded
        data_splits = split_temporal_data(normal, slow, config)
        assert data_splits["val_anomaly"].shape[0] == int(0.5 * slow.shape[0])

    def test_temporal_continuity(self, loaded, config):
        """train is the first 70%, val the next 15%, test_normal the last 15%."""
        normal, slow, _ = loaded
        n_total = normal.shape[0]
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)

        data_splits = split_temporal_data(normal, slow, config)

        # train[0] == normal[0], train[-1] == normal[n_train-1]
        np.testing.assert_array_equal(data_splits["train"][0], normal[0])
        np.testing.assert_array_equal(data_splits["train"][-1], normal[n_train - 1])
        # val[0] == normal[n_train], val[-1] == normal[n_train+n_val-1]
        np.testing.assert_array_equal(data_splits["val_normal"][0], normal[n_train])
        # test_normal[-1] == normal[-1]
        np.testing.assert_array_equal(data_splits["test_normal"][-1], normal[-1])

    def test_feature_dim_consistency(self, splits):
        data_splits, selected_cols = splits
        expected_dim = len(selected_cols)  # 86 after preprocessing
        for key in ("train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"):
            assert data_splits[key].shape[1] == expected_dim


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------
class TestNormalizeData:
    def test_train_in_range(self, normalised):
        """All train features must be in [-1, 1] after MinMaxScaler."""
        scaled, _ = normalised
        train = scaled["train"]
        assert np.all(train >= -1 - 1e-6), (
            f"Train min below -1 after MinMaxScaler; max deviation = {np.abs(train.min() + 1).max()}"
        )
        assert np.all(train <= 1 + 1e-6), (
            f"Train max above 1 after MinMaxScaler; max deviation = {np.abs(train.max() - 1).max()}"
        )

    def test_no_leakage(self, normalised):
        """Val range must differ from train range (proves scaler was fit on train only)."""
        scaled, _ = normalised
        train_range = scaled["train"].max(axis=0) - scaled["train"].min(axis=0)
        val_range = scaled["val_normal"].max(axis=0) - scaled["val_normal"].min(axis=0)
        assert not np.allclose(train_range, val_range, atol=1e-3), (
            "val feature range ~= train feature range — scaler may have been fit on val"
        )

    def test_scaler_is_fitted(self, normalised):
        _, scaler = normalised
        assert hasattr(scaler, "data_min_"), "Scaler must be fitted"
        assert hasattr(scaler, "data_max_")
        # Scaler is fitted on all 86 features
        assert scaler.data_min_.shape == (86,)
        assert scaler.data_max_.shape == (86,)

    def test_all_splits_finite(self, normalised):
        scaled, _ = normalised
        for key in ("train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"):
            assert scaled[key].dtype == np.float32
            assert np.all(np.isfinite(scaled[key])), f"{key} contains inf/nan"


# ---------------------------------------------------------------------------
# KukaDataset
# ---------------------------------------------------------------------------
class TestKukaDataset:
    @pytest.fixture
    def dummy_data(self):
        """2D array: 100 samples, 5 features."""
        rng = np.random.RandomState(42)
        return rng.randn(100, 5).astype(np.float32)

    def test_len(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, stride=1, label=0)
        expected = (100 - 16) // 1 + 1  # 85
        assert len(ds) == expected, f"Expected {expected}, got {len(ds)}"

    def test_getitem_shape(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        window, label = ds[0]
        assert window.shape == (16, 5), f"Expected (16, 5), got {window.shape}"
        assert label.shape == (1,), f"Expected (1,), got {label.shape}"

    def test_getitem_dtype(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        window, label = ds[0]
        assert window.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_label_consistency_normal(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() == 0.0

    def test_label_consistency_anomaly(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=1)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() == 1.0

    def test_temporal_order_preserved(self, dummy_data):
        """Window at idx i must start at row i*stride of the original data."""
        W = 16
        stride = 1
        ds = KukaDataset(dummy_data, window_size=W, stride=stride, label=0)
        for i in range(min(5, len(ds))):
            window, _ = ds[i]
            expected = torch.from_numpy(dummy_data[i * stride : i * stride + W])
            torch.testing.assert_close(window, expected)

    def test_strided_windows(self, dummy_data):
        stride = 3
        W = 16
        ds = KukaDataset(dummy_data, window_size=W, stride=stride, label=0)
        for i in range(len(ds)):
            window, _ = ds[i]
            start = i * stride
            expected = torch.from_numpy(dummy_data[start : start + W])
            torch.testing.assert_close(window, expected)

    def test_index_out_of_range(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_window_size_greater_than_data(self):
        data = np.random.randn(10, 5).astype(np.float32)
        ds = KukaDataset(data, window_size=16, label=0)
        assert len(ds) == 0

    def test_accepts_torch_tensor(self, dummy_data):
        data = torch.from_numpy(dummy_data)
        ds = KukaDataset(data, window_size=16, label=0)
        assert len(ds) == 85
        window, _ = ds[0]
        assert window.shape == (16, 5)

    def test_negative_index(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        last_idx = len(ds) - 1
        window_neg, _ = ds[-1]
        window_pos, _ = ds[last_idx]
        torch.testing.assert_close(window_neg, window_pos)

    def test_repr(self, dummy_data):
        ds = KukaDataset(dummy_data, window_size=16, label=0)
        repr_str = repr(ds)
        assert "KukaDataset" in repr_str
        assert "label=0" in repr_str

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            KukaDataset(np.random.randn(10, 5, 3), window_size=16, label=0)

    def test_empty_data(self):
        with pytest.raises(ValueError):
            KukaDataset(np.array([]).reshape(0, 5), window_size=16, label=0)

    def test_real_data_shapes(self):
        """Test with actual processed data files (generated by run_preprocessing)."""
        train = np.load(Path("data/processed/train.npy"))
        ds = KukaDataset(train, window_size=16, label=0)
        window, label = ds[0]
        # After preprocessing: 86 features (1 action + 85 numerical)
        assert window.shape == (16, 86), f"Expected (16, 86), got {window.shape}"
        assert label.item() == 0.0
        assert len(ds) == train.shape[0] - 16 + 1
