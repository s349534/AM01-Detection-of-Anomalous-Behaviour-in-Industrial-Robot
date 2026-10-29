"""Tests for Fase 2 preprocessing + dataset pipeline (interleaved partitioning).

Run: uv run pytest tests/test_dataset.py -v
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch

from src.data.dataset import KukaDataset, build_dataloaders
from src.data.partitioning import (
    PartitionSpec,
    plan_partitions,
    assign_chunks_spread,
    verify_no_leakage,
)
from src.data.preprocessing import load_kuka_data
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


# ---------------------------------------------------------------------------
# load_kuka_data
# ---------------------------------------------------------------------------
class TestLoadKukaData:
    def test_normal_shape(self, loaded):
        normal, _, _ = loaded
        assert normal.shape == (233792, 86), f"Expected (233792, 86), got {normal.shape}"

    def test_slow_shape(self, loaded):
        _, slow, _ = loaded
        assert slow.shape == (41538, 86), f"Expected (41538, 86), got {slow.shape}"

    def test_selected_cols_shape(self, loaded):
        _, _, cols = loaded
        assert cols.shape == (86,), f"Expected (86,), got {cols.shape}"

    def test_columns_aligned(self, loaded):
        normal, slow, cols = loaded
        assert normal.shape[1] == slow.shape[1] == 86
        assert len(cols) == 86

    def test_anomaly_dropped(self, loaded):
        _, _, cols = loaded
        assert "anomaly" not in cols

    def test_dtype(self, loaded):
        normal, slow, _ = loaded
        assert normal.dtype == np.float32
        assert slow.dtype == np.float32


# ---------------------------------------------------------------------------
# Partitioning: assign_chunks_spread
# ---------------------------------------------------------------------------
class TestAssignChunksSpread:
    def test_exact_counts(self):
        """Total chunks assigned equals n_chunks, counts match quotas within ±1."""
        n_chunks = 100
        quotas = [0.7, 0.15, 0.15]
        owner = assign_chunks_spread(n_chunks, quotas)
        assert len(owner) == n_chunks
        counts = np.bincount(owner, minlength=3)
        expected = np.array([70, 15, 15])
        assert np.all(np.abs(counts - expected) <= 1)

    def test_spread_distribution(self):
        """Each partition's chunks are spread across the timeline."""
        n_chunks = 100
        quotas = [0.7, 0.15, 0.15]
        owner = assign_chunks_spread(n_chunks, quotas)
        # Partition 0 (70%) should appear in first, middle, last third
        for p in range(3):
            third = n_chunks // 3
            first_third = np.any(owner[:third] == p)
            middle_third = np.any(owner[third:2*third] == p)
            last_third = np.any(owner[2*third:] == p)
            assert first_third or middle_third or last_third, f"Partition {p} not spread"

    def test_deterministic(self):
        """Same input produces same assignment."""
        o1 = assign_chunks_spread(100, [0.7, 0.15, 0.15])
        o2 = assign_chunks_spread(100, [0.7, 0.15, 0.15])
        assert np.array_equal(o1, o2)


# ---------------------------------------------------------------------------
# Partitioning: plan_partitions
# ---------------------------------------------------------------------------
class TestPlanPartitions:
    def test_basic_plan(self):
        """Basic plan generation with known specs."""
        specs = [
            PartitionSpec("train", 0.7, 0.25),
            PartitionSpec("val", 0.15, 0.5),
            PartitionSpec("test", 0.15, 0.5),
        ]
        plan = plan_partitions(
            n_samples=10000,
            window_size=100,
            chunk_size=1000,
            gap=100,
            specs=specs,
        )
        assert "train" in plan
        assert "val" in plan
        assert "test" in plan
        # Check no empty partitions
        for name in specs:
            assert len(plan[name.name]) > 0, f"Partition {name.name} has no windows"

    def test_disjointness(self):
        """Partitions have no shared samples."""
        specs = [
            PartitionSpec("train", 0.7, 0.25),
            PartitionSpec("val", 0.15, 0.5),
            PartitionSpec("test", 0.15, 0.5),
        ]
        plan = plan_partitions(10000, 100, 1000, 100, specs)
        # Verify using the verification function
        verify_no_leakage(plan, window_size=100, gap=100)

    def test_stride_values(self):
        """Windows within a partition have correct stride."""
        specs = [
            PartitionSpec("train", 0.7, 0.25),  # stride = 25 (W/4)
            PartitionSpec("val", 0.15, 0.5),    # stride = 50 (W/2)
        ]
        window_size = 100
        plan = plan_partitions(
            n_samples=10000, window_size=window_size, chunk_size=1000, gap=100, specs=specs
        )
        # Check train stride: minimum diff should equal stride (25)
        train_starts = plan["train"]
        diffs = np.diff(train_starts)
        # Within a chunk, stride = 25; between chunks, larger gaps appear
        min_diff = diffs.min()
        assert min_diff == 25, f"Train min stride mismatch: got {min_diff}, expected 25"

        # Check val stride
        val_starts = plan["val"]
        diffs = np.diff(val_starts)
        min_diff = diffs.min()
        assert min_diff == 50, f"Val min stride mismatch: got {min_diff}, expected 50"

    def test_gap_at_partition_change(self):
        """Guard band of size gap between partitions."""
        specs = [
            PartitionSpec("train", 0.5, 0.25),
            PartitionSpec("val", 0.5, 0.5),
        ]
        # chunk_size must be >= 2*W + max(stride) = 2*100 + 50 = 250
        plan = plan_partitions(1000, 100, 250, 100, specs)
        # The last train window should end 100 samples before first val window
        train_ends = plan["train"] + 100  # window_size=100
        val_starts = plan["val"]
        # Find transition
        last_train_end = train_ends[np.argmax(train_ends >= val_starts[0])]
        first_val_start = val_starts[val_starts > last_train_end][0]
        gap = first_val_start - last_train_end - 1
        assert gap >= 100, f"Gap {gap} < 100"

    def test_chunk_size_insufficient_raises(self):
        """chunk_size too small raises ValueError."""
        specs = [PartitionSpec("train", 1.0, 0.25)]
        with pytest.raises(ValueError, match="too small"):
            plan_partitions(1000, 200, 100, 100, specs)  # chunk_size=100 < 2*200 + 50


# ---------------------------------------------------------------------------
# Partitioning: verify_no_leakage
# ---------------------------------------------------------------------------
class TestVerifyNoLeakage:
    def test_passes_on_valid_plan(self):
        specs = [
            PartitionSpec("train", 0.7, 0.25),
            PartitionSpec("val", 0.15, 0.5),
            PartitionSpec("test", 0.15, 0.5),
        ]
        plan = plan_partitions(10000, 100, 1000, 100, specs)
        verify_no_leakage(plan, window_size=100, gap=100)

    def test_fails_on_overlap(self):
        plan = {"a": np.array([0, 50]), "b": np.array([25])}  # overlap
        with pytest.raises(ValueError):
            verify_no_leakage(plan, window_size=100, gap=0)

    def test_fails_on_insufficient_gap(self):
        plan = {"a": np.array([0]), "b": np.array([50])}  # gap = 49, need 100
        with pytest.raises(ValueError):
            verify_no_leakage(plan, window_size=100, gap=100)


# ---------------------------------------------------------------------------
# KukaDataset (new API with start_indices)
# ---------------------------------------------------------------------------
class TestKukaDataset:
    @pytest.fixture
    def dummy_data(self):
        rng = np.random.RandomState(42)
        return rng.randn(100, 5).astype(np.float32)

    @pytest.fixture
    def starts(self):
        return np.array([0, 16, 32, 48, 64], dtype=np.int64)

    def test_len(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        assert len(ds) == len(starts)

    def test_getitem_shape(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        window, label = ds[0]
        assert window.shape == (16, 5)
        assert label.shape == (1,)

    def test_getitem_dtype(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        window, label = ds[0]
        assert window.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_label_consistency(self, dummy_data, starts):
        ds_normal = KukaDataset(dummy_data, starts, window_size=16, label=0)
        ds_anomaly = KukaDataset(dummy_data, starts, window_size=16, label=1)
        for i in range(len(ds_normal)):
            assert ds_normal[i][1].item() == 0.0
        for i in range(len(ds_anomaly)):
            assert ds_anomaly[i][1].item() == 1.0

    def test_window_correctness(self, dummy_data):
        """Window at index i uses start_indices[i]."""
        starts = np.array([10, 25, 40], dtype=np.int64)
        ds = KukaDataset(dummy_data, starts, window_size=10, label=0)
        for i, s in enumerate(starts):
            window, _ = ds[i]
            expected = torch.from_numpy(dummy_data[s:s+10])
            torch.testing.assert_close(window, expected)

    def test_index_out_of_range(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_invalid_start_indices(self, dummy_data):
        starts = np.array([95], dtype=np.int64)  # 95 + 16 > 100
        with pytest.raises(ValueError):
            KukaDataset(dummy_data, starts, window_size=16, label=0)

    def test_accepts_torch_tensor(self, dummy_data, starts):
        data = torch.from_numpy(dummy_data)
        ds = KukaDataset(data, starts, window_size=16, label=0)
        assert len(ds) == 5
        window, _ = ds[0]
        assert window.shape == (16, 5)

    def test_negative_index(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        window_neg, _ = ds[-1]
        window_pos, _ = ds[len(starts) - 1]
        torch.testing.assert_close(window_neg, window_pos)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            KukaDataset(np.random.randn(10, 5, 3), np.array([0]), window_size=16, label=0)

    def test_empty_data(self):
        with pytest.raises(ValueError):
            KukaDataset(np.array([]).reshape(0, 5), np.array([]), window_size=16, label=0)

    def test_repr(self, dummy_data, starts):
        ds = KukaDataset(dummy_data, starts, window_size=16, label=0)
        r = repr(ds)
        assert "KukaDataset" in r
        assert "label=0" in r


# ---------------------------------------------------------------------------
# Integration: load_kuka_data + plan + dataset
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_plan_and_dataset_shapes(self, config):
        """End-to-end: load -> plan -> dataset."""
        normal, slow, cols = load_kuka_data(config)

        # Plan with default config parameters
        plan = plan_partitions(
            n_samples=normal.shape[0],
            window_size=config["data"]["window_size"],
            chunk_size=config["data"]["partitioning"]["chunk_size"],
            gap=config["data"]["window_size"],
            specs=[
                PartitionSpec("train", 0.7, 0.25),
                PartitionSpec("val", 0.15, 0.5),
                PartitionSpec("test", 0.15, 0.5),
            ],
        )
        assert "train" in plan
        assert "val" in plan
        assert "test" in plan

        # Build dataset
        ds = KukaDataset(normal, plan["train"], config["data"]["window_size"], label=0)
        assert len(ds) > 0
        window, label = ds[0]
        assert window.shape == (config["data"]["window_size"], 86)
        assert label.item() == 0.0


# ---------------------------------------------------------------------------
# run_preprocessing (if processed data exists)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not Path("data/processed/normal_scaled.npy").exists(),
    reason="Processed data not generated yet"
)
class TestPreprocessingOutput:
    def test_artifacts_exist(self):
        processed_dir = Path("data/processed")
        for name in ["normal_scaled.npy", "slow_scaled.npy",
                     "plan_train.npy", "plan_val_normal.npy",
                     "plan_val_anomaly.npy", "plan_test_normal.npy",
                     "plan_test_anomaly.npy", "scaler.pkl",
                     "selected_columns.npy", "manifest.json", "stats.json"]:
            assert (processed_dir / name).exists(), f"Missing {name}"

    def test_manifest_fields(self):
        with open("data/processed/manifest.json") as f:
            manifest = json.load(f)
        assert manifest["window_size"] > 0
        assert "partitions" in manifest
        for name, info in manifest["partitions"].items():
            assert info["n_windows"] > 0
            assert 0 < info["coverage_ratio"] <= 1

    def test_build_dataloaders(self, config):
        loaders = build_dataloaders(config, partitions=["train", "val_normal", "val_anomaly"], num_workers=0, use_prebuilt_partitions=False)
        assert "train" in loaders
        assert "val_normal" in loaders
        assert "val_anomaly" in loaders
        # Check first batch: window is (B, W, n_features), label is (B, 1)
        window, label = next(iter(loaders["train"]))
        batch_size = window.shape[0]
        assert window.shape[1:] == (config["data"]["window_size"], 86)
        assert label.shape == (batch_size, 1)
        # Labels must be the partition label (0 = normal)
        assert torch.allclose(label, torch.zeros_like(label))

    def test_stats_json(self):
        """stats.json holds per-partition descriptive statistics."""
        with open("data/processed/stats.json") as f:
            stats = json.load(f)
        assert stats["n_features"] == 86
        assert len(stats["feature_names"]) == 86
        for rec in ("normal", "slow"):
            for stat in ("min", "max", "mean", "std"):
                assert len(stats["recordings"][rec]["raw"][stat]) == 86
        # Every partition must have statistics
        for name in ("train", "val_normal", "test_normal", "val_anomaly", "test_anomaly"):
            part = stats["partitions"][name]
            assert part["n_windows"] > 0
            assert len(part["scaled"]["mean"]) == 86