"""Data preprocessing pipeline for the AM01 anomaly-detection project.

Independent of notebooks (see project_plan.md §5.1): every function here is
importable, unit-testable, and reproducible.

Pipeline (no clipping — see §7 #9):
    load_kuka_data → split_temporal_data → normalize_data → save_processed_data
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# A feature is considered "constant" if its standard deviation across all
# rows is below this threshold (numpy default ddof=0).  The value 1e-8 is
# deliberately small so that only *truly* constant features (e.g. a sensor
# stuck at its idle reading) trigger removal.
CONST_STD_THRESHOLD = 1e-8


# ---------------------------------------------------------------------------
# Step 1 — load
# ---------------------------------------------------------------------------
def load_kuka_data(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three raw .npy files, drop `anomaly`, remove constant features.

    KukaSlow has 87 columns (feature + anomaly label); KukaNormal has 86.
    The `anomaly` column (always 1) is dropped before alignment.  A feature
    is removed only if it is constant in **both** datasets — a feature that
    is constant in one but not the other may be discriminative and is kept.

    Returns
    -------
    normal : np.ndarray, shape (233 792, 82), float32
    slow : np.ndarray, shape (41 538, 82), float32
    selected_col_names : np.ndarray, shape (82,)
    """
    data_cfg = config["data"]
    normal_path = Path(data_cfg["normal_data_path"])
    slow_path = Path(data_cfg["slow_data_path"])
    col_names_path = Path(data_cfg["column_names_path"])

    # --- Load raw .npy files ---
    normal = np.load(normal_path, allow_pickle=True)   # (233792, 86)
    slow = np.load(slow_path, allow_pickle=True)        # (41538, 87)
    col_names = np.load(col_names_path, allow_pickle=True)  # (87,)

    logger.info(
        "Loaded: normal=%s, slow=%s, col_names=%s",
        normal.shape, slow.shape, col_names.shape,
    )

    # --- Step 2: align columns (drop 'anomaly' from KukaSlow) ---
    if slow.shape[1] == normal.shape[1] + 1:
        # The 87th column is 'anomaly' — drop it.
        dropped_name = str(col_names[slow.shape[1] - 1])
        slow = slow[:, : normal.shape[1]]
        logger.info(
            "Dropped '%s' column from KukaSlow (%d → %d)",
            dropped_name, normal.shape[1] + 1, slow.shape[1],
        )

    assert slow.shape[1] == normal.shape[1], (
        f"Column mismatch after alignment: "
        f"normal={normal.shape[1]}, slow={slow.shape[1]}"
    )

    # --- Step 3: identify zero-variance features ---
    std_normal = normal.std(axis=0)   # (86,)
    std_slow = slow.std(axis=0)        # (86,)

    # Only remove if constant in BOTH datasets.
    constant_mask = (std_normal < CONST_STD_THRESHOLD) & (
        std_slow < CONST_STD_THRESHOLD
    )
    constant_indices = np.where(constant_mask)[0]

    if constant_indices.size > 0:
        constant_names = [str(col_names[i]) for i in constant_indices]
        logger.info(
            "Removing %d zero-variance feature(s) from both datasets: %s",
            constant_indices.size, constant_names,
        )
    else:
        logger.info("No zero-variance features found.")

    # Warn if a feature is constant in one dataset but not the other.
    one_only_mask = (std_normal < CONST_STD_THRESHOLD) & (
        std_slow >= CONST_STD_THRESHOLD
    )
    if one_only_mask.any():
        indices = np.where(one_only_mask)[0]
        names = [str(col_names[i]) for i in indices]
        warnings.warn(
            f"Feature(s) constant in KukaNormal but NOT in KukaSlow — kept: {names}. "
            "They may be discriminative between normal and anomalous behaviour.",
            UserWarning,
        )

    one_only_mask_2 = (std_normal >= CONST_STD_THRESHOLD) & (
        std_slow < CONST_STD_THRESHOLD
    )
    if one_only_mask_2.any():
        indices = np.where(one_only_mask_2)[0]
        names = [str(col_names[i]) for i in indices]
        warnings.warn(
            f"Feature(s) constant in KukaSlow but NOT in KukaNormal — kept: {names}.",
            UserWarning,
        )

    # --- Step 4: select columns ---
    keep_mask = ~constant_mask
    keep_indices = np.where(keep_mask)[0]

    normal = normal[:, keep_mask].astype(np.float32)
    slow = slow[:, keep_mask].astype(np.float32)
    selected_col_names = col_names[keep_indices]

    logger.info(
        "After removing %d constant features → normal=%s, slow=%s",
        constant_indices.size, normal.shape, slow.shape,
    )

    return normal, slow, selected_col_names


# ---------------------------------------------------------------------------
# Step 2 — split
# ---------------------------------------------------------------------------
def split_temporal_data(
    normal: np.ndarray,
    slow: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Temporal split (no shuffle) of KukaNormal plus all of KukaSlow.

    Normal data is split 60/20/20 along the time axis.  All of KukaSlow
    (the anomaly class) goes to ``test_anomaly``.  No shuffle because
    lag-1 autocorrelation ≈ 0.99+: shuffling would leak future into past.

    Returns dict with keys: train, val, test_normal, test_anomaly.
    """
    train_split = float(config["training"]["train_split"])  # 0.6
    val_split = float(config["training"]["val_split"])      # 0.2

    n_total = normal.shape[0]
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    # n_test = n_total - n_train - n_val  (the remainder)

    train = normal[:n_train]
    val = normal[n_train : n_train + n_val]
    test_normal = normal[n_train + n_val :]
    test_anomaly = slow  # entire KukaSlow → test_anomaly

    logger.info(
        "Temporal split (no shuffle): "
        "train=%d (%.0f%%), val=%d (%.0f%%), "
        "test_normal=%d (%.0f%%), test_anomaly=%d (100%% of KukaSlow)",
        train.shape[0], train_split * 100,
        val.shape[0], val_split * 100,
        test_normal.shape[0], (1 - train_split - val_split) * 100,
        test_anomaly.shape[0],
    )

    # --- Sanity assertions ---
    assert train.shape[0] + val.shape[0] + test_normal.shape[0] == n_total, (
        "Split mismatch — train + val + test_normal ≠ normal"
    )
    assert test_normal.shape[1] == train.shape[1] == val.shape[1] == slow.shape[1], (
        "Feature dimension mismatch across splits"
    )

    return {
        "train": train,
        "val": val,
        "test_normal": test_normal,
        "test_anomaly": test_anomaly,
    }


# ---------------------------------------------------------------------------
# Step 3 — normalize
# ---------------------------------------------------------------------------
def normalize_data(
    data: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    """StandardScaler (z-score) fitted on ``train`` only.

    Features have wildly different scales (std 0.01–51.8); without
    normalisation the latent space is dominated by high-variance features.
    Fit on train only — any other split would leak statistics and bias
    evaluation.  No clipping: sensor saturation is absorbed by the scaler
    and the MSE loss (§7 #9).

    Returns
    -------
    scaled : dict[str, np.ndarray] — standardised float32 arrays
    scaler : StandardScaler — the fitted scaler (save as scaler.pkl)
    """
    norm_method = config.get("data", {}).get("normalization", "standard")
    if norm_method != "standard":
        raise ValueError(
            f"Only 'standard' normalization is supported, got '{norm_method}'"
        )

    scaler = StandardScaler()
    scaler.fit(data["train"])  # fit on train ONLY

    # --- Detect features that are constant within the TRAIN split ---
    # load_kuka_data checks constants on the FULL dataset, but a feature may
    # be non-constant globally yet constant within the first 60 % (train).
    # StandardScaler sets scale_=1 for zero-variance features, so the transform
    # is safe — we just log a warning so the analyst is aware.
    near_zero_var = scaler.var_ < CONST_STD_THRESHOLD
    if near_zero_var.any():
        nz_cols = np.where(near_zero_var)[0]
        logger.warning(
            "Features with near-zero variance in TRAIN split (indices %s): "
            "these are constant within the first %.0f%% of KukaNormal. "
            "StandardScaler handles them safely (scale set to 1).",
            nz_cols.tolist(),
            config["training"]["train_split"] * 100,
        )

    scaled: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        scaled[key] = scaler.transform(arr).astype(np.float32)

    # --- Log statistics (train should be ≈0 mean, ≈1 std for non-constant feats) ---
    for key in ("train", "val", "test_normal", "test_anomaly"):
        arr = scaled[key]
        logger.info(
            "%-14s: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  shape=%s",
            key, arr.mean(), arr.std(), arr.min(), arr.max(), arr.shape,
        )

    # --- Assertions: train is properly standardised ---
    # Check mean ≈ 0 across all features (this is always exact with StandardScaler).
    train_mean = scaled["train"].mean(axis=0)
    assert np.allclose(train_mean, 0, atol=1e-3), (
        f"Train mean should be ≈0 after StandardScaler; "
        f"max abs deviation = {np.abs(train_mean).max()}"
    )

    # Check std ≈ 1 for non-constant features only (constant features in train
    # will have std = 0 after scaling, which is expected and harmless).
    non_const = ~near_zero_var
    if non_const.any():
        train_std_nonconst = scaled["train"].std(axis=0)[non_const]
        assert np.all((train_std_nonconst > 0.97) & (train_std_nonconst < 1.03)), (
            f"Non-constant train std should be ≈1; "
            f"min={train_std_nonconst.min():.4f}, max={train_std_nonconst.max():.4f}"
        )

    # --- Assertions: val/test are NOT zero-mean (proves no leakage) ---
    val_mean_abs = np.abs(scaled["val"].mean(axis=0)).max()
    assert val_mean_abs > 1e-3, (
        f"Val mean ≈ 0 — possible scaler leakage (max abs mean={val_mean_abs})"
    )

    return scaled, scaler


# ---------------------------------------------------------------------------
# Step 4 — save
# ---------------------------------------------------------------------------
def save_processed_data(
    data: dict[str, np.ndarray],
    scaler: StandardScaler,
    selected_col_names: np.ndarray,
    config: dict[str, Any],
) -> None:
    """Save .npy arrays, scaler.pkl, and selected_columns.npy to data/processed/."""
    output_dir = Path(config["paths"]["processed_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save .npy arrays ---
    for key, arr in data.items():
        path = output_dir / f"{key}.npy"
        np.save(path, arr)
        logger.info("Saved %s: shape=%s, dtype=%s", path, arr.shape, arr.dtype)

    # --- Save scaler ---
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved %s", scaler_path)

    # --- Save selected column names ---
    cols_path = output_dir / "selected_columns.npy"
    np.save(cols_path, selected_col_names)
    logger.info("Saved %s: %d columns", cols_path, selected_col_names.shape[0])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_preprocessing(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run the full preprocessing pipeline: load → split → normalize → save."""
    logger.info("=" * 60)
    logger.info("AM01 — Fase 2 Preprocessing Pipeline")
    logger.info("NO clipping applied (by project design — see §7 #9)")
    logger.info("=" * 60)

    # --- Step 1: Load ---
    logger.info("[1/4] Loading Kuka data...")
    normal, slow, selected_cols = load_kuka_data(config)

    # --- Step 2: Split ---
    logger.info("[2/4] Splitting data (temporal, 60/20/20)...")
    data_splits = split_temporal_data(normal, slow, config)

    # --- Step 3: Normalize ---
    logger.info("[3/4] Normalizing (StandardScaler, fit on train only)...")
    scaled_splits, scaler = normalize_data(data_splits, config)

    # --- Step 4: Save ---
    logger.info("[4/4] Saving processed data...")
    save_processed_data(scaled_splits, scaler, selected_cols, config)

    logger.info("Preprocessing completed successfully.")
    return scaled_splits


# ---------------------------------------------------------------------------
# CLI entry-point  (allows ``python -m src.data.preprocessing``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    run_preprocessing(cfg)
