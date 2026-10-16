"""Data preprocessing pipeline for the AM01 anomaly-detection project.

Independent of notebooks (see project_plan.md §5.1): every function here is
importable, unit-testable, and reproducible.

Pipeline:
    load_kuka_data → split_temporal_data → normalize_data → save_processed_data

Key EDA-driven decisions:
- action (index 0) is kept as a single numeric channel, scaled together with
  the other features by MinMaxScaler
- Clipping: NOT implemented (0% hardware saturation)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Column index of the action channel
ACTION_COL_IDX = 0


# ---------------------------------------------------------------------------
# Step 1 — load
# ---------------------------------------------------------------------------
def load_kuka_data(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three raw .npy files and drop `anomaly`.

    KukaSlow has 87 columns (feature + anomaly label); KukaNormal has 86.
    The `anomaly` column (always 1) is dropped before alignment.
    The `action` column (index 0) is kept as a single numeric channel.

    Returns
    -------
    normal : np.ndarray, shape (233 792, 86), float32
    slow : np.ndarray, shape (41 538, 86), float32
    selected_col_names : np.ndarray, shape (86,)  (action + 85 numerical features)
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

    # --- Step 1: align columns (drop 'anomaly' from KukaSlow) ---
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

    # After dropping anomaly, both have 86 columns:
    # action (0) + 85 numerical features (1-85)
    # col_names has 87 names: action (0) + 85 numerical (1-85) + anomaly (86)
    selected_col_names = col_names[: normal.shape[1]]

    logger.info(
        "Final features: %d total (1 action + %d numerical)",
        normal.shape[1], normal.shape[1] - 1,
    )

    return (
        normal.astype(np.float32),
        slow.astype(np.float32),
        selected_col_names,
    )


# ---------------------------------------------------------------------------
# Step 2 — split
# ---------------------------------------------------------------------------
def split_temporal_data(
    normal: np.ndarray,
    slow: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Temporal split (no shuffle) of KukaNormal and KukaSlow.

    Normal data is split 70/15/15 along the time axis.
    KukaSlow is split 50/50 along the time axis.
    No shuffle because lag-1 autocorrelation ≈ 0.99+: shuffling would leak future into past.

    Returns dict with keys: train, val_normal, val_anomaly, test_normal, test_anomaly.
    """
    train_split = float(config["training"]["train_split"])           # 0.70
    val_split = float(config["training"]["val_split"])               # 0.15
    slow_val_split = float(config["training"].get("slow_val_split", 0.50))  # 0.50

    n_total = normal.shape[0]
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    # n_test = n_total - n_train - n_val  (remainder = 15%)

    train = normal[:n_train]
    val_normal = normal[n_train : n_train + n_val]
    test_normal = normal[n_train + n_val :]

    s_total = slow.shape[0]
    s_val = int(slow_val_split * s_total)
    val_anomaly = slow[:s_val]
    test_anomaly = slow[s_val:]

    logger.info(
        "Temporal split (no shuffle): "
        "train=%d (%.0f%%), val_normal=%d (%.0f%%), test_normal=%d (%.0f%%), "
        "val_anomaly=%d (%.0f%%), test_anomaly=%d (%.0f%%)",
        train.shape[0], train_split * 100,
        val_normal.shape[0], val_split * 100,
        test_normal.shape[0], (1 - train_split - val_split) * 100,
        val_anomaly.shape[0], slow_val_split * 100,
        test_anomaly.shape[0], (1 - slow_val_split) * 100,
    )

    # --- Sanity assertions ---
    assert train.shape[0] + val_normal.shape[0] + test_normal.shape[0] == n_total, (
        "Split mismatch — train + val_normal + test_normal ≠ normal"
    )
    assert val_anomaly.shape[0] + test_anomaly.shape[0] == s_total, (
        "Split mismatch — val_anomaly + test_anomaly ≠ slow"
    )
    assert test_normal.shape[1] == train.shape[1] == val_normal.shape[1] == slow.shape[1], (
        "Feature dimension mismatch across splits"
    )

    return {
        "train": train,
        "val_normal": val_normal,
        "val_anomaly": val_anomaly,
        "test_normal": test_normal,
        "test_anomaly": test_anomaly,
    }


# ---------------------------------------------------------------------------
# Step 3 — normalize
# ---------------------------------------------------------------------------
def normalize_data(
    data: dict[str, np.ndarray],
    config: dict[str, Any],
    selected_col_names: np.ndarray,
) -> tuple[dict[str, np.ndarray], MinMaxScaler]:
    """MinMaxScaler(feature_range=(-1,1)) fitted on ``train``.

    All features — including the action channel at index 0 — are scaled
    together.  Fit on train only — any other split would leak statistics and
    bias evaluation.  No clipping: sensor saturation is absorbed by the
    scaler and the MAE loss (§7 #9).

    Returns
    -------
    scaled : dict[str, np.ndarray] — minimax-scaled float32 arrays
    scaler : MinMaxScaler — the fitted scaler (save as scaler.pkl)
    """
    norm_method = config.get("data", {}).get("normalization", "minmax")
    if norm_method != "minmax":
        raise ValueError(
            f"Only 'minmax' normalization is supported, got '{norm_method}'"
        )

    n_features = data["train"].shape[1]
    logger.info(
        "Normalization: MinMaxScaler(feature_range=(-1,1)) on all %d features, fit on train",
        n_features,
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Fit on train only
    scaler.fit(data["train"])

    scaled: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        scaled[key] = scaler.transform(arr).astype(np.float32)

    # --- Log statistics ---
    for key in ("train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"):
        arr = scaled[key]
        logger.info(
            "%-14s: [min=%.4f max=%.4f]  shape=%s",
            key, arr.min(), arr.max(), arr.shape,
        )

    # --- Assertions: train features are in [-1, 1] ---
    train_min = scaled["train"].min(axis=0)
    train_max = scaled["train"].max(axis=0)
    assert np.all(train_min >= -1 - 1e-6), (
        f"Train min below -1 after MinMaxScaler; max deviation = {np.abs(train_min + 1).max()}"
    )
    assert np.all(train_max <= 1 + 1e-6), (
        f"Train max above 1 after MinMaxScaler; max deviation = {np.abs(train_max - 1).max()}"
    )

    # --- Assertions: val/test range differs from train (proves no leakage) ---
    # MinMaxScaler maps train to [-1,1]; val/test may legitimately fall outside
    # that range.  What must NOT happen is that val/test share train's exact
    # range, which would mean the scaler was fit on them.
    train_range = train_max - train_min
    val_range = scaled["val_normal"].max(axis=0) - scaled["val_normal"].min(axis=0)
    assert not np.allclose(train_range, val_range, atol=1e-3), (
        "val feature range ~= train feature range — scaler may have been fit on val"
    )

    # --- Verify slow data is NOT compressed toward zero ---
    slow_arr = scaled["val_anomaly"]
    slow_range = slow_arr.max(axis=0) - slow_arr.min(axis=0)
    n_crushed = int(np.sum(slow_range < 0.05))
    if n_crushed > 0:
        logger.warning(
            "%d/%d slow features have range < 0.05 — possible compression toward zero",
            n_crushed, slow_arr.shape[1],
        )
    else:
        logger.info(
            "Slow data range check OK: min feature range = %.4f, max = %.4f",
            slow_range.min(), slow_range.max(),
        )

    return scaled, scaler


# ---------------------------------------------------------------------------
# Step 4 — save
# ---------------------------------------------------------------------------
def save_processed_data(
    data: dict[str, np.ndarray],
    scaler: MinMaxScaler,
    selected_col_names: np.ndarray,
    config: dict[str, Any],
) -> None:
    """Save .npy arrays, scaler.pkl, and selected_columns.npy to data/processed/."""
    output_dir = Path(config["paths"]["data_processed"])
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

    # --- Update params.yaml with discovered input_dim ---
    _update_params_input_dim(config, int(selected_col_names.shape[0]))


def _update_params_input_dim(config: dict[str, Any], input_dim: int) -> None:
    """Update params.yaml with the discovered input_dim from preprocessing.

    The params.yaml file contains a placeholder input_dim (82) that gets
    overridden at runtime by reading selected_columns.npy. This function
    writes the actual discovered value back to params.yaml so the config
    reflects the true data dimensionality.

    Parameters
    ----------
    config : dict
        Full config dict (must contain paths).
    input_dim : int
        Actual number of features after preprocessing.
    """
    # Project root: this file is src/data/preprocessing.py → parents[2] = project_root
    project_root = Path(__file__).resolve().parents[2]
    params_path = project_root / "config" / "params.yaml"

    if not params_path.exists():
        logger.warning("params.yaml not found at %s, skipping input_dim update", params_path)
        return

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}

        # Update model.input_dim
        if "model" not in params:
            params["model"] = {}
        old_dim = params["model"].get("input_dim", "unset")
        params["model"]["input_dim"] = input_dim

        with open(params_path, "w", encoding="utf-8") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Updated params.yaml: model.input_dim %s → %d", old_dim, input_dim)

    except Exception as e:
        logger.warning("Failed to update params.yaml with input_dim: %s", e)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_preprocessing(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run the full preprocessing pipeline: load → split → normalize → save."""
    logger.info("=" * 60)
    logger.info("AM01 — Fase 2 Preprocessing Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Load ---
    logger.info("[1/4] Loading Kuka data...")
    normal, slow, selected_cols = load_kuka_data(config)

    # --- Step 2: Split ---
    logger.info("[2/4] Splitting data (temporal, 70/15/15)...")
    data_splits = split_temporal_data(normal, slow, config)

    # --- Step 3: Normalize ---
    logger.info("[3/4] Normalizing (MinMaxScaler, fit on train)...")
    scaled_splits, scaler = normalize_data(data_splits, config, selected_cols)

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
