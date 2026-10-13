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
    CONST_STD_THRESHOLD,
    load_kuka_data,
    normalize_data,
    remove_constant_features,
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
    # Also remove constant features based on train split
    data_splits, selected_cols = remove_constant_features(data_splits, selected_cols, config)
    return data_splits, selected_cols


@pytest.fixture(scope="module")
def normalised(splits, config):
    """StandardScale the splits and return (scaled_dict, scaler)."""
    data_splits, selected_cols = splits
    return normalize_data(data_splits, config, selected_cols)


# ---------------------------------------------------------------------------
# load_kuka_data
# ---------------------------------------------------------------------------
class TestLoadKukaData:
    def test_normal_shape(self, loaded):
        normal, _, _ = loaded
        # After load: 82 numerical + 32 one-hot = 114 (after correlated removal)
        assert normal.shape == (233792, 114), f"Expected (233792, 114), got {normal.shape}"

    def test_slow_shape(self, loaded):
        _, slow, _ = loaded
        assert slow.shape == (41538, 114), f"Expected (41538, 114), got {slow.shape}"

    def test_selected_cols_shape(self, loaded):
        _, _, cols = loaded
        assert cols.shape == (114,), f"Expected (114,), got {cols.shape}"

    def test_columns_aligned(self, loaded):
        """Both arrays must have the same number of features."""
        normal, slow, cols = loaded
        assert normal.shape[1] == slow.shape[1] == 114
        assert len(cols) == 114

    def test_anomaly_dropped(self, loaded):
        """The 'anomaly' column must not be among the retained columns."""
        _, _, cols = loaded
        assert "anomaly" not in cols

    def test_highly_correlated_removed(self, loaded):
        """The 3 highly correlated features must be removed."""
        _, _, cols = loaded
        removed = {
            "machine_nameKuka Robot_apparent_power",
            "machine_nameKuka Robot_power_factor",
            "sensor_id7_q2",
        }
        assert removed.isdisjoint(set(cols.tolist())), (
            f"Found removed features in selected columns: {removed & set(cols.tolist())}"
        )

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
        expected_dim = len(selected_cols)  # 109 after constant removal
        for key in ("train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"):
            assert data_splits[key].shape[1] == expected_dim


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------
class TestNormalizeData:
    def test_train_zero_mean(self, normalised):
        scaled, _ = normalised
        # Check only numerical features (exclude one-hot action columns)
        is_oh = np.array([str(c).startswith('action_') for c in scaled["train"].dtype.names]) if scaled["train"].dtype.names else np.zeros(scaled["train"].shape[1], dtype=bool)
        # Since we don't have column names here, we know first N are numerical, last 32 are one-hot
        n_numerical = scaled["train"].shape[1] - 32
        train_num = scaled["train"][:, :n_numerical]
        train_mean = train_num.mean(axis=0)
        np.testing.assert_allclose(
            train_mean, 0, atol=1e-3,
            err_msg="train numerical mean should be ≈0 after StandardScaler",
        )

    def test_train_unit_std(self, normalised):
        """All numerical features should have std ≈1 after StandardScaler (constants removed upstream)."""
        scaled, _ = normalised
        n_numerical = scaled["train"].shape[1] - 32
        train_num = scaled["train"][:, :n_numerical]
        train_std = train_num.std(axis=0)
        np.testing.assert_allclose(
            train_std, 1, atol=0.03,
            err_msg="Train numerical std should be ≈1 after StandardScaler",
        )

    def test_no_leakage(self, normalised):
        """Val mean must NOT be ≈0 (proves scaler was fit on train only)."""
        scaled, _ = normalised
        n_numerical = scaled["val_normal"].shape[1] - 32
        val_num = scaled["val_normal"][:, :n_numerical]
        val_mean_abs = np.abs(val_num.mean(axis=0)).max()
        assert val_mean_abs > 1e-3, (
            f"val numerical mean ≈ 0 — scaler may have been fit on val (max abs={val_mean_abs})"
        )

    def test_scaler_is_fitted(self, normalised):
        _, scaler = normalised
        assert hasattr(scaler, "mean_"), "Scaler must be fitted"
        assert hasattr(scaler, "scale_")
        # Scaler is fitted on numerical features only (77)
        assert scaler.mean_.shape == (77,)
        assert scaler.scale_.shape == (77,)

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
        # After preprocessing: 77 numerical + 32 one-hot = 109
        assert window.shape == (16, 109), f"Expected (16, 109), got {window.shape}"
        assert label.item() == 0.0
        assert len(ds) == train.shape[0] - 16 + 1
