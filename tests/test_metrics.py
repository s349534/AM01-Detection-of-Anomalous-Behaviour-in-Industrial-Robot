"""Tests for the metrics module.

Run: uv run pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    calculate_auc,
    calculate_metrics,
    compute_anomaly_scores,
    find_optimal_threshold,
    percentile_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def perfect_scores():
    """Scores where all anomalies have higher scores than normal."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y_true, scores


@pytest.fixture
def random_scores():
    """Scores uncorrelated with labels (AUC ≈ 0.5)."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.random.rand(10)
    return y_true, scores


@pytest.fixture
def perfect_pred():
    """Perfect binary predictions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Test calculate_auc
# ---------------------------------------------------------------------------
class TestCalculateAUC:
    def test_perfect_separation(self, perfect_scores):
        """AUC should be 1.0 when scores perfectly separate classes."""
        y_true, scores = perfect_scores
        auc = calculate_auc(y_true, scores)
        assert auc["roc_auc"] == pytest.approx(1.0, abs=1e-6)
        assert auc["pr_auc"] == pytest.approx(1.0, abs=1e-6)

    def test_random_scores(self, random_scores):
        """AUC should be around 0.5 for uncorrelated scores."""
        y_true, scores = random_scores
        auc = calculate_auc(y_true, scores)
        assert 0.3 <= auc["roc_auc"] <= 0.7, f"ROC-AUC={auc['roc_auc']}, expected ~0.5"
        assert 0.3 <= auc["pr_auc"] <= 0.7, f"PR-AUC={auc['pr_auc']}, expected ~0.5"

    def test_single_class(self):
        """AUC should be 0.5 when only one class is present."""
        y_true = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        auc = calculate_auc(y_true, scores)
        assert auc["roc_auc"] == 0.5
        assert auc["pr_auc"] == 0.5


# ---------------------------------------------------------------------------
# Test calculate_metrics
# ---------------------------------------------------------------------------
class TestCalculateMetrics:
    def test_perfect_predictions(self, perfect_pred):
        y_true, y_pred = perfect_pred
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)

    def test_half_right(self):
        # y_true: [0,0,0,0, 1,1,1,1]
        # y_pred: [0,0,0,0, 0,1,1,1]  → 1 false negative (idx 4)
        # TN=4, TP=3, FN=1, FP=0 → 7/8 correct
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.875)  # 7/8 correct
        assert metrics["precision"] == pytest.approx(1.0)   # TP/(TP+FP) = 3/3
        assert metrics["recall"] == pytest.approx(0.75)   # TP/(TP+FN) = 3/4
        assert metrics["f1"] == pytest.approx(2 * 1.0 * 0.75 / (1.0 + 0.75), abs=1e-6)

    def test_no_positives_predicted(self):
        """All predictions negative — precision should be 0 (no division by zero)."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["precision"] == 0.0  # zero_division=0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


# ---------------------------------------------------------------------------
# Test percentile_threshold
# ---------------------------------------------------------------------------
class TestPercentileThreshold:
    def test_99th_percentile(self):
        scores = np.linspace(0, 100, 10001)  # 0.0, 0.01, ..., 100.0
        threshold = percentile_threshold(scores, 99)
        assert threshold == pytest.approx(99.0, abs=0.2)

    def test_50th_percentile(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = percentile_threshold(scores, 50)
        assert threshold == pytest.approx(3.0)

    def test_single_value(self):
        scores = np.array([42.0])
        threshold = percentile_threshold(scores, 99)
        assert threshold == 42.0


# ---------------------------------------------------------------------------
# Test find_optimal_threshold
# ---------------------------------------------------------------------------
class TestFindOptimalThreshold:
    def test_finds_perfect_threshold(self, perfect_scores):
        """Optimal threshold should perfectly separate classes."""
        y_true, scores = perfect_scores
        threshold, metrics = find_optimal_threshold(y_true, scores)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_threshold_is_float(self, perfect_scores):
        y_true, scores = perfect_scores
        threshold, _ = find_optimal_threshold(y_true, scores)
        assert isinstance(threshold, float)

    def test_threshold_value_reasonable(self, perfect_scores):
        """Threshold should be between 0.4 and 0.6 for perfect scores."""
        y_true, scores = perfect_scores
        threshold, metrics = find_optimal_threshold(y_true, scores)
        assert 0.4 <= threshold <= 0.6


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
