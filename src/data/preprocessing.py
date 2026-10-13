"""Data preprocessing pipeline for the AM01 anomaly-detection project.

Independent of notebooks (see project_plan.md §5.1): every function here is
importable, unit-testable, and reproducible.

Pipeline (no clipping — see §7 #9):
    load_kuka_data → split_temporal_data → normalize_data → save_processed_data

Key EDA-driven decisions:
- action (0-31) → one-hot encoding (32 binary cols), excluded from StandardScaler
- Clipping: NOT implemented (0% hardware saturation)
- Highly correlated features (Spearman > 0.95) removed from numerical features only
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# A feature is considered "constant" if its standard deviation across all
# rows is below this threshold (numpy default ddof=0).  The value 1e-8 is
# deliberately small so that only *truly* constant features (e.g. a sensor
# stuck at its idle reading) trigger removal.
CONST_STD_THRESHOLD = 1e-8

# Column indices and constants
ACTION_COL_IDX = 0
N_ACTION_CLASSES = 32
CORR_THRESHOLD = 0.95  # Spearman correlation threshold for removing redundant numerical features


def _is_onehot_name(name: str) -> bool:
    """Check if column name is a one-hot action column (prefix 'action_')."""
    return str(name).startswith("action_")


# ---------------------------------------------------------------------------
# Step 1 — load
# ---------------------------------------------------------------------------
def load_kuka_data(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three raw .npy files, drop `anomaly`,
    remove highly correlated numerical features (Spearman > 0.95),
    and one-hot encode `action` column.

    KukaSlow has 87 columns (feature + anomaly label); KukaNormal has 86.
    The `anomaly` column (always 1) is dropped before alignment.
    The `action` column (index 0) is one-hot encoded to 32 binary columns.
    Highly correlated numerical features (Spearman > 0.95) are removed using
    the train split (first 70%) to avoid leakage — the feature with lower
    variance in each correlated pair is dropped.
    Constant features (near-zero variance in TRAIN split) are removed
    in a separate step after splitting (see `remove_constant_features`).

    Returns
    -------
    normal : np.ndarray, shape (233 792, ~107), float32  (~75 numerical + 32 one-hot)
    slow : np.ndarray, shape (41 538, ~107), float32
    selected_col_names : np.ndarray, shape (~107,)  (numerical names + 'action_0'...'action_31')
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

    # --- Step 2: extract action column (index 0) for one-hot encoding ---
    action_normal = normal[:, ACTION_COL_IDX].astype(int)   # (N,)
    action_slow = slow[:, ACTION_COL_IDX].astype(int)        # (M,)

    # --- Step 3: work with numerical features only (exclude action column 0) ---
    # After dropping anomaly, both have 86 columns: action (0) + 85 numerical features
    # col_names has 87 names: action (0) + 85 numerical (1-85) + anomaly (86)
    numerical_normal = normal[:, 1:]   # (N, 85) - exclude action
    numerical_slow = slow[:, 1:]       # (M, 85) - exclude action
    numerical_names = col_names[1:86]  # 85 names - exclude action (0) and anomaly (86)

    # --- Step 4: remove highly correlated numerical features (Spearman > 0.95) ---
    # Use train split (first 70%) to avoid data leakage
    train_split = float(config["training"]["train_split"])
    n_train = int(train_split * numerical_normal.shape[0])
    train_numerical = numerical_normal[:n_train]

    logger.info("Computing Spearman correlation on train split (%d samples)...", n_train)
    corr_matrix, _ = spearmanr(train_numerical)
    np.fill_diagonal(corr_matrix, 0.0)  # ignore diagonal

    # Find features to remove: for each correlated pair, drop the one with lower variance
    to_remove = set()
    n_feat = corr_matrix.shape[0]
    for i in range(n_feat):
        if i in to_remove:
            continue
        for j in range(i + 1, n_feat):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > CORR_THRESHOLD:
                var_i = train_numerical[:, i].var()
                var_j = train_numerical[:, j].var()
                to_remove.add(i if var_i < var_j else j)

    if to_remove:
        # Get removed names before filtering
        removed_names = [str(numerical_names[i]) for i in sorted(to_remove)]
        remove_mask = np.ones(numerical_normal.shape[1], dtype=bool)
        remove_mask[list(to_remove)] = False
        numerical_normal = numerical_normal[:, remove_mask]
        numerical_slow = numerical_slow[:, remove_mask]
        numerical_names = numerical_names[remove_mask]
        logger.info(
            "Removed %d highly correlated numerical features (Spearman > %.2f): %s",
            len(to_remove), CORR_THRESHOLD, removed_names,
        )
    else:
        logger.info("No highly correlated numerical features found (Spearman > %.2f).", CORR_THRESHOLD)

    # --- Step 7: one-hot encode action ---
    action_onehot_normal = np.eye(N_ACTION_CLASSES, dtype=np.float32)[action_normal]
    action_onehot_slow = np.eye(N_ACTION_CLASSES, dtype=np.float32)[action_slow]
    action_names = np.array([f"action_{i}" for i in range(N_ACTION_CLASSES)])

    # --- Step 8: concatenate numerical + one-hot ---
    normal = np.hstack([numerical_normal, action_onehot_normal]).astype(np.float32)
    slow = np.hstack([numerical_slow, action_onehot_slow]).astype(np.float32)
    selected_col_names = np.concatenate([numerical_names, action_names])

    n_numerical = numerical_normal.shape[1]
    logger.info(
        "Final features: %d numerical + %d one-hot (action) = %d total",
        n_numerical, N_ACTION_CLASSES, len(selected_col_names),
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
# Step 2b — remove constant features (based on TRAIN split only)
# ---------------------------------------------------------------------------
def remove_constant_features(
    data: dict[str, np.ndarray],
    selected_col_names: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Remove numerical features with near-zero variance in the TRAIN split only.

    This prevents StandardScaler from encountering zero-variance features during
    fit, which would set scale_=1.0 and cause only mean-centering (no z-scoring).
    Such features maintain their original magnitude while others are in [-3, 3],
    creating artificial drift in val/test splits and distorting anomaly detection.

    The threshold CONST_STD_THRESHOLD (1e-8) is deliberately small to catch only
    truly constant features (e.g., a sensor stuck at its idle reading).

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Dictionary with keys: train, val_normal, val_anomaly, test_normal, test_anomaly.
    selected_col_names : np.ndarray
        Array of column names corresponding to data columns.
    config : dict[str, Any]
        Configuration dict (used for train_split logging).

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray]
        Updated data dict and selected_col_names with constant features removed.
    """
    # Derive numerical vs one-hot indices from column names
    is_onehot = np.array([_is_onehot_name(n) for n in selected_col_names])
    numerical_indices = np.where(~is_onehot)[0]

    # Compute variance on train numerical features ONLY
    train_numerical = data["train"][:, numerical_indices]
    variances = train_numerical.var(axis=0)  # ddof=0 (population variance)

    # Identify constant features in train split
    constant_mask = variances < CONST_STD_THRESHOLD

    if not constant_mask.any():
        logger.info("No constant numerical features found in TRAIN split.")
        return data, selected_col_names

    # Get names of features to remove
    removed_indices = np.where(constant_mask)[0]
    removed_names = [str(selected_col_names[numerical_indices[i]]) for i in removed_indices]

    logger.info(
        "Removing %d constant numerical feature(s) from TRAIN split (var < %.0e): %s",
        len(removed_indices), CONST_STD_THRESHOLD, removed_names,
    )

    # Build keep mask for ALL features (numerical + one-hot)
    # One-hot features are never removed (they have variance p(1-p) > 0)
    keep_mask = np.ones(len(selected_col_names), dtype=bool)
    keep_mask[numerical_indices[constant_mask]] = False

    # Apply to all splits
    updated_data = {}
    for key, arr in data.items():
        updated_data[key] = arr[:, keep_mask]

    updated_names = selected_col_names[keep_mask]

    n_numerical = np.sum(~np.array([_is_onehot_name(n) for n in updated_names]))
    n_onehot = np.sum(np.array([_is_onehot_name(n) for n in updated_names]))
    logger.info(
        "After constant removal: %d numerical + %d one-hot (action) = %d total",
        n_numerical, n_onehot, len(updated_names),
    )

    return updated_data, updated_names


# ---------------------------------------------------------------------------
# Step 3 — normalize
# ---------------------------------------------------------------------------
def normalize_data(
    data: dict[str, np.ndarray],
    config: dict[str, Any],
    selected_col_names: np.ndarray,
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    """StandardScaler (z-score) fitted on ``train`` numerical features only.

    One-hot encoded action columns (prefix 'action_') are excluded from scaling:
    they are already in [0, 1] range and StandardScaler would distort them
    (e.g., mean ≈ 0.03 for rare actions → 1 maps to +30, 0 maps to -0.03).

    Features have wildly different scales (std 0.01–51.8); without
    normalisation the latent space is dominated by high-variance features.
    Fit on train only — any other split would leak statistics and bias
    evaluation.  No clipping: sensor saturation is absorbed by the scaler
    and the MAE loss (§7 #9).

    Returns
    -------
    scaled : dict[str, np.ndarray] — standardised float32 arrays
    scaler : StandardScaler — the fitted scaler (save as scaler.pkl), fitted on numerical features only
    """
    norm_method = config.get("data", {}).get("normalization", "standard")
    if norm_method != "standard":
        raise ValueError(
            f"Only 'standard' normalization is supported, got '{norm_method}'"
        )

    # Derive numerical vs one-hot indices from column names
    is_onehot = np.array([_is_onehot_name(n) for n in selected_col_names])
    numerical_indices = np.where(~is_onehot)[0]
    onehot_indices = np.where(is_onehot)[0]

    n_numerical = len(numerical_indices)
    n_onehot = len(onehot_indices)
    logger.info("Normalization: %d numerical features (StandardScaler), %d one-hot action features (kept as 0/1)",
                n_numerical, n_onehot)

    scaler = StandardScaler()
    # Fit on train numerical features ONLY
    train_numerical = data["train"][:, numerical_indices]
    scaler.fit(train_numerical)

    scaled: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        arr_scaled = arr.copy()
        # Scale only numerical features
        arr_scaled[:, numerical_indices] = scaler.transform(arr[:, numerical_indices])
        # One-hot features remain unchanged (already 0/1)
        scaled[key] = arr_scaled.astype(np.float32)

    # --- Log statistics (separate numerical vs one-hot) ---
    for key in ("train", "val_normal", "val_anomaly", "test_normal", "test_anomaly"):
        arr = scaled[key]
        num_arr = arr[:, numerical_indices]
        oh_arr = arr[:, onehot_indices] if n_onehot > 0 else None
        logger.info(
            "%-14s: num[mean=%.4f std=%.4f]  oh[mean=%.4f std=%.4f]  shape=%s",
            key,
            num_arr.mean(), num_arr.std(),
            oh_arr.mean() if oh_arr is not None else 0.0,
            oh_arr.std() if oh_arr is not None else 0.0,
            arr.shape,
        )

    # --- Assertions: train numerical features are properly standardised ---
    train_num = scaled["train"][:, numerical_indices]
    train_mean = train_num.mean(axis=0)
    assert np.allclose(train_mean, 0, atol=1e-3), (
        f"Train numerical mean should be ≈0 after StandardScaler; "
        f"max abs deviation = {np.abs(train_mean).max()}"
    )

    # All features should have std ≈ 1 (constant features removed upstream)
    train_std = train_num.std(axis=0)
    assert np.all((train_std > 0.97) & (train_std < 1.03)), (
        f"Train numerical std should be ≈1 for all features; "
        f"min={train_std.min():.4f}, max={train_std.max():.4f}"
    )

    # --- Assertions: val/test numerical features are NOT zero-mean (proves no leakage) ---
    val_num_mean_abs = np.abs(scaled["val_normal"][:, numerical_indices].mean(axis=0)).max()
    assert val_num_mean_abs > 1e-3, (
        f"Val numerical mean ~= 0 -- possible scaler leakage (max abs mean={val_num_mean_abs})"
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
        Actual number of features after preprocessing (numerical + one-hot).
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
    """Run the full preprocessing pipeline: load → split → remove constants → normalize → save."""
    logger.info("=" * 60)
    logger.info("AM01 — Fase 2 Preprocessing Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Load ---
    logger.info("[1/5] Loading Kuka data...")
    normal, slow, selected_cols = load_kuka_data(config)

    # --- Step 2: Split ---
    logger.info("[2/5] Splitting data (temporal, 70/15/15)...")
    data_splits = split_temporal_data(normal, slow, config)

    # --- Step 3: Remove constant features (based on TRAIN split only) ---
    logger.info("[3/5] Removing constant features (train split variance < %.0e)...", CONST_STD_THRESHOLD)
    data_splits, selected_cols = remove_constant_features(data_splits, selected_cols, config)

    # --- Step 4: Normalize ---
    logger.info("[4/5] Normalizing (StandardScaler on numerical only, fit on train)...")
    scaled_splits, scaler = normalize_data(data_splits, config, selected_cols)

    # --- Step 5: Save ---
    logger.info("[5/5] Saving processed data...")
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
