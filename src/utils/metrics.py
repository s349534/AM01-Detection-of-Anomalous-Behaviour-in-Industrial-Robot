"""Deprecated: Metrics module moved to src.evaluation.metrics.

This file is kept for backward compatibility. Please update imports to:

    from src.evaluation.metrics import (
        calculate_metrics,
        calculate_auc,
        find_optimal_threshold,
        percentile_threshold,
        compute_anomaly_scores,
        evaluate_threshold,
    )
"""
import warnings

warnings.warn(
    "src.utils.metrics is deprecated; use src.evaluation.metrics instead",
    DeprecationWarning,
    stacklevel=2,
)

from src.evaluation.metrics import (  # noqa: F401,F403
    calculate_metrics,
    calculate_auc,
    find_optimal_threshold,
    percentile_threshold,
    compute_anomaly_scores,
    evaluate_threshold,
)