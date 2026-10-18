"""Tests for the evaluation module.

Run: uv run pytest tests/test_evaluation.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    calibrate_threshold,
    evaluate_model,
    evaluate_on_splits,
    compute_reconstruction_errors,
    compute_errors_on_loader,
    compute_errors_from_arrays,
    calculate_metrics,
    calculate_auc,
    compute_anomaly_scores,
    evaluate_threshold,
)
from src.models.autoencoder import SequenceAutoencoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model():
    """Create a mock SequenceAutoencoder that returns predictable reconstructions."""
    model = MagicMock(spec=SequenceAutoencoder)
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    np.random.seed(42)
    train_data = np.random.randn(100, 86).astype(np.float32)
    val_normal_data = np.random.randn(30, 86).astype(np.float32)
    val_anomaly_data = np.random.randn(20, 86).astype(np.float32)
    test_normal_data = np.random.randn(30, 86).astype(np.float32)
    test_anomaly_data = np.random.randn(20, 86).astype(np.float32)
    return {
        "train": train_data,
        "val_normal": val_normal_data,
        "val_anomaly": val_anomaly_data,
        "test_normal": test_normal_data,
        "test_anomaly": test_anomaly_data,
    }


@pytest.fixture
def dummy_loaders():
    """Create dummy DataLoaders for testing."""
    np.random.seed(42)
    # Normal data: lower errors
    val_normal_windows = torch.randn(50, 16, 86) * 0.1
    val_normal_labels = torch.zeros(50, 1)
    val_normal_loader = DataLoader(TensorDataset(val_normal_windows, val_normal_labels), batch_size=16)

    # Anomaly data: higher errors
    val_anomaly_windows = torch.randn(30, 16, 86) * 1.0 + 0.5
    val_anomaly_labels = torch.ones(30, 1)
    val_anomaly_loader = DataLoader(TensorDataset(val_anomaly_windows, val_anomaly_labels), batch_size=16)

    test_normal_windows = torch.randn(40, 16, 86) * 0.1
    test_normal_labels = torch.zeros(40, 1)
    test_normal_loader = DataLoader(TensorDataset(test_normal_windows, test_normal_labels), batch_size=16)

    test_anomaly_windows = torch.randn(25, 16, 86) * 1.0 + 0.5
    test_anomaly_labels = torch.ones(25, 1)
    test_anomaly_loader = DataLoader(TensorDataset(test_anomaly_windows, test_anomaly_labels), batch_size=16)

    return {
        "val_normal": val_normal_loader,
        "val_anomaly": val_anomaly_loader,
        "test_normal": test_normal_loader,
        "test_anomaly": test_anomaly_loader,
    }


# ---------------------------------------------------------------------------
# Test calculate_auc (delegated to metrics tests, but smoke test here)
# ---------------------------------------------------------------------------
class TestCalculateAUC:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        auc = calculate_auc(y_true, scores)
        assert auc["roc_auc"] == pytest.approx(1.0, abs=1e-6)
        assert auc["pr_auc"] == pytest.approx(1.0, abs=1e-6)

    def test_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        auc = calculate_auc(y_true, scores)
        assert auc["roc_auc"] == 0.5
        assert auc["pr_auc"] == 0.5


# ---------------------------------------------------------------------------
# Test calculate_metrics
# ---------------------------------------------------------------------------
class TestCalculateMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test calibrate_threshold
# ---------------------------------------------------------------------------
class TestCalibrateThreshold:
    def test_f1_optimal_method(self, mock_model, dummy_loaders):
        """Test F1-optimal threshold calibration."""
        # Mock the compute_errors_on_loader to return predictable errors
        with patch("src.evaluation.threshold.compute_errors_on_loader") as mock_compute:
            # Normal errors: low values
            # Anomaly errors: high values
            mock_compute.side_effect = [
                np.array([0.1, 0.15, 0.2, 0.12, 0.18]),  # val_normal_errors
                np.array([0.8, 0.9, 1.0, 0.85, 0.95]),   # val_anomaly_errors
            ]

            threshold, metrics = calibrate_threshold(
                mock_model,
                dummy_loaders["val_normal"],
                dummy_loaders["val_anomaly"],
                torch.device("cpu"),
                metric="mae",
                method="f1_optimal",
            )

            assert isinstance(threshold, float)
            assert "f1" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert metrics["f1"] > 0.5  # Should separate well

    def test_percentile_method(self, mock_model, dummy_loaders):
        """Test percentile-based threshold calibration."""
        with patch("src.evaluation.threshold.compute_errors_on_loader") as mock_compute:
            mock_compute.side_effect = [
                np.array([0.1, 0.15, 0.2, 0.12, 0.18]),  # val_normal_errors
                np.array([0.8, 0.9, 1.0, 0.85, 0.95]),   # val_anomaly_errors
            ]

            threshold, metrics = calibrate_threshold(
                mock_model,
                dummy_loaders["val_normal"],
                dummy_loaders["val_anomaly"],
                torch.device("cpu"),
                metric="mae",
                method="percentile_99",
                percentile=99.0,
            )

            assert isinstance(threshold, float)
            assert "f1" in metrics


# ---------------------------------------------------------------------------
# Test evaluate_model
# ---------------------------------------------------------------------------
class TestEvaluateModel:
    def test_evaluate_model(self, mock_model, dummy_loaders):
        """Test evaluate_model with a fixed threshold."""
        with patch("src.evaluation.scoring.compute_errors_on_loader") as mock_compute:
            # Normal errors: low
            # Anomaly errors: high
            mock_compute.side_effect = [
                np.array([0.1, 0.15, 0.2]),  # test_normal_errors
                np.array([0.8, 0.9, 1.0]),   # test_anomaly_errors
            ]

            threshold = 0.5
            metrics = evaluate_model(
                mock_model,
                dummy_loaders["test_normal"],
                dummy_loaders["test_anomaly"],
                threshold,
                torch.device("cpu"),
                metric="mae",
            )

            assert "pr_auc" in metrics
            assert "roc_auc" in metrics
            assert "f1" in metrics
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "confusion_matrix" in metrics
            assert metrics["threshold"] == threshold


# ---------------------------------------------------------------------------
# Test compute_anomaly_scores
# ---------------------------------------------------------------------------
class TestComputeAnomalyScores:
    def test_basic_thresholding(self):
        errors = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        threshold = 1.0
        preds = compute_anomaly_scores(errors, threshold)
        expected = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(preds, expected)

    def test_all_below_threshold(self):
        errors = np.array([0.1, 0.2, 0.3])
        preds = compute_anomaly_scores(errors, 1.0)
        np.testing.assert_array_equal(preds, np.array([0, 0, 0]))

    def test_all_above_threshold(self):
        errors = np.array([1.1, 1.2, 1.3])
        preds = compute_anomaly_scores(errors, 1.0)
        np.testing.assert_array_equal(preds, np.array([1, 1, 1]))

    def test_returns_int_type(self):
        errors = np.array([0.1, 1.0, 2.0])
        preds = compute_anomaly_scores(errors, 0.5)
        assert preds.dtype in (np.int32, np.int64)


# ---------------------------------------------------------------------------
# Test evaluate_threshold
# ---------------------------------------------------------------------------
class TestEvaluateThreshold:
    def test_full_evaluation(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        threshold = 0.5

        result = evaluate_threshold(y_true, scores, threshold)

        assert result["threshold"] == threshold
        assert result["roc_auc"] == pytest.approx(1.0, abs=1e-6)
        assert result["pr_auc"] == pytest.approx(1.0, abs=1e-6)
        assert result["f1"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        assert cm["tn"] == 4
        assert cm["tp"] == 4
        assert cm["fp"] == 0
        assert cm["fn"] == 0


# ---------------------------------------------------------------------------
# Test evaluate_on_splits (integration test with temp dir)
# ---------------------------------------------------------------------------
class TestEvaluateOnSplits:
    def test_evaluate_on_splits(self, tmp_path, dummy_data):
        """Integration test for evaluate_on_splits with real data files."""
        # Save dummy data to temp directory
        for key, arr in dummy_data.items():
            np.save(tmp_path / f"{key}.npy", arr)

        # Create a real model (small for testing)
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=8)
        model.eval()

        # Mock compute_errors_on_loader to avoid actual inference
        with patch("src.evaluation.scoring.compute_errors_on_loader") as mock_compute:
            mock_compute.side_effect = [
                np.array([0.1] * 20),  # val_normal_errors
                np.array([0.8] * 15),  # val_anomaly_errors
                np.array([0.1] * 18),  # test_normal_errors
                np.array([0.8] * 12),  # test_anomaly_errors
            ]

            result = evaluate_on_splits(
                model=model,
                processed_dir=tmp_path,
                window_size=16,
                batch_size=8,
                threshold=None,  # calibrate
                device=torch.device("cpu"),
                metric="mae",
            )

            assert "threshold" in result
            assert "val_metrics" in result
            assert "test_metrics" in result
            assert result["val_metrics"] is not None
            assert result["test_metrics"]["f1"] > 0.5


# ---------------------------------------------------------------------------
# Test compute_reconstruction_errors (scoring)
# ---------------------------------------------------------------------------
class TestComputeReconstructionErrors:
    def test_compute_errors_on_loader_import(self):
        """Test that compute_errors_on_loader is importable and callable."""
        # Just verify the function exists and has correct signature
        import inspect
        sig = inspect.signature(compute_errors_on_loader)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "loader" in params
        assert "device" in params
        assert "metric" in params

    def test_compute_reconstruction_errors_import(self):
        """Test that compute_reconstruction_errors is importable and callable."""
        import inspect
        sig = inspect.signature(compute_reconstruction_errors)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "loader" in params
        assert "device" in params
        assert "metric" in params

    def test_compute_errors_from_arrays_import(self):
        """Test that compute_errors_from_arrays is importable and callable."""
        import inspect
        sig = inspect.signature(compute_errors_from_arrays)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "data" in params
        assert "window_size" in params
        assert "batch_size" in params
        assert "device" in params