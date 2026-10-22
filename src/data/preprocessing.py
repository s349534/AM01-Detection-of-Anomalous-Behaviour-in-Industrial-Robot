"""Data preprocessing pipeline for the AM01 anomaly-detection project.

Independent of notebooks (see project_plan.md §5.1): every function here is
importable, unit-testable, and reproducible.

Pipeline:
    load_kuka_data → plan_partitions (interleaved) → scale (fit on train chunks) → save

Artifacts:
    normal_scaled.npy, slow_scaled.npy, plan_*.npy, scaler.pkl,
    selected_columns.npy, manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.data.partitioning import plan_from_config, verify_no_leakage
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Column index of the action channel
# ACTION_COL_IDX = 0


# ---------------------------------------------------------------------------
# Step 1 — load (unchanged)
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
# Step 2 — partition (NEW: interleaved chunks with guard bands)
# ---------------------------------------------------------------------------
def plan_partitions_for_recordings(
    normal: np.ndarray,
    slow: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compute window start indices for all partitions from both recordings.

    Args:
        normal: Normal recording array (n_samples, n_features).
        slow: Slow recording array (n_samples, n_features).
        config: Full config dict.

    Returns:
        Dict mapping partition name -> window start indices.
    """
    normal_plan = plan_from_config(config, "normal", normal.shape[0], slow.shape[0])
    slow_plan = plan_from_config(config, "slow", normal.shape[0], slow.shape[0])

    # Merge plans
    plan = {}
    plan.update(normal_plan)
    plan.update(slow_plan)
    return plan


# ---------------------------------------------------------------------------
# Step 3 — normalize (fit on train rows only — no leakage)
# ---------------------------------------------------------------------------
def fit_scaler_on_train(
    normal: np.ndarray,
    slow: np.ndarray,
    plan: dict[str, np.ndarray],
    config: dict[str, Any],
    window_size: int,
) -> tuple[MinMaxScaler, np.ndarray]:
    """Fit MinMaxScaler on TRAIN rows only (no leakage).

    The scaler is fitted only on the rows covered by train windows.
    Val/test may fall outside [-1,1] after transform — this is expected
    due to distribution drift and is reported as a warning.
    """
    # --- Build train mask from plan ---
    train_mask = np.zeros(normal.shape[0], dtype=bool)
    for s in plan.get("train", np.array([], dtype=np.int64)):
        train_mask[s:s + window_size] = True

    train_rows = normal[train_mask]

    norm_method = config.get("data", {}).get("normalization", "minmax")
    if norm_method != "minmax":
        raise ValueError(f"Only 'minmax' normalization is supported, got '{norm_method}'")

    logger.info(
        "Normalization: MinMaxScaler(feature_range=(-1,1)) on %d features, fit on %d train rows",
        normal.shape[1], len(train_rows),
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_rows)

    # --- Transform all data ---
    normal_scaled = scaler.transform(normal).astype(np.float32)
    slow_scaled = scaler.transform(slow).astype(np.float32)

    # --- Check for out-of-range values on val/test (warning, not error) ---
    clip_out_of_range = config.get("data", {}).get("clip_out_of_range", False)

    # Check val/test partitions for out-of-range values
    for name, starts in plan.items():
        if name == "train":
            continue
        if "anomaly" in name:
            arr = slow_scaled
            n_samples = slow.shape[0]
        else:
            arr = normal_scaled
            n_samples = normal.shape[0]

        covered = np.zeros(n_samples, dtype=bool)
        for s in starts:
            covered[s:s + window_size] = True
        if not np.any(covered):
            continue

        vals = arr[covered]
        out_of_range = (vals < -1.0) | (vals > 1.0)
        n_out = int(out_of_range.sum())
        if n_out > 0:
            n_feats = int((out_of_range.sum(axis=0) > 0).sum())
            logger.warning(
                "Partition %s: %d samples (%d features) out of [-1,1] range after scaling. "
                "This is expected due to distribution drift. Consider setting data.clip_out_of_range=true.",
                name, n_out, n_feats
            )

    if clip_out_of_range:
        logger.info("Clipping scaled values to [-1, 1] range")
        normal_scaled = np.clip(normal_scaled, -1.0, 1.0)
        slow_scaled = np.clip(slow_scaled, -1.0, 1.0)

    # Build train_mask from plan (for manifest)
    train_mask = np.zeros(normal.shape[0], dtype=bool)
    for s in plan.get("train", np.array([], dtype=np.int64)):
        train_mask[s:s + window_size] = True

    return scaler, train_mask

def scale_recordings(    normal: np.ndarray,
    slow: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both recordings with the fitted scaler."""
    normal_scaled = scaler.transform(normal).astype(np.float32)
    slow_scaled = scaler.transform(slow).astype(np.float32)
    return normal_scaled, slow_scaled


def _checksum(arr: np.ndarray) -> str:
    """Compute SHA256 checksum of array data."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Step 4 — save (new artifacts + manifest)
# ---------------------------------------------------------------------------


def compute_statistics(
    normal: np.ndarray,
    slow: np.ndarray,
    normal_scaled: np.ndarray,
    slow_scaled: np.ndarray,
    plan: dict[str, np.ndarray],
    window_size: int,
    feature_names: np.ndarray,
) -> dict[str, Any]:
    """Compute descriptive statistics for recordings and partitions.

    Args:
        normal: Raw normal recording (n_samples, n_features).
        slow: Raw slow recording (n_samples, n_features).
        normal_scaled: Scaled normal recording.
        slow_scaled: Scaled slow recording.
        plan: Partition plan with start indices per partition.
        window_size: Window length W.
        feature_names: Array of feature names.

    Returns:
        Dictionary with statistics for recordings and partitions.
    """
    def compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
        """Compute min, max, mean, std per feature."""
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    def partition_stats(scaled_arr: np.ndarray, starts: np.ndarray, n_samples: int) -> dict[str, Any]:
        if len(starts) == 0:
            return {"n_windows": 0, "n_covered_samples": 0, "coverage_ratio": 0.0,
                    "raw": None, "scaled": None}
        covered = np.zeros(n_samples, dtype=bool)
        for s in starts:
            covered[s:s + window_size] = True
        n_covered = int(covered.sum())
        return {
            "n_windows": len(starts),
            "n_covered_samples": n_covered,
            "coverage_ratio": float(n_covered) / n_samples,
            "raw": None,  # raw stats would require accessing original data
            "scaled": compute_stats(scaled_arr[covered]),
        }

    # Recording-level stats
    normal_raw_stats = compute_stats(normal)
    slow_raw_stats = compute_stats(slow)
    normal_scaled_stats = compute_stats(normal_scaled)
    slow_scaled_stats = compute_stats(slow_scaled)

    # Partition-level stats (using scaled data)
    partitions = {}
    for name, starts in plan.items():
        if "anomaly" in name:
            arr_scaled = slow_scaled
            n_total = slow.shape[0]
        else:
            arr_scaled = normal_scaled
            n_total = normal.shape[0]
        partitions[name] = partition_stats(arr_scaled, starts, n_total)

    return {
        "window_size": window_size,
        "n_features": int(normal.shape[1]),
        "feature_names": feature_names.tolist(),
        "recordings": {
            "normal": {
                "n_samples": int(normal.shape[0]),
                "raw": normal_raw_stats,
                "scaled": normal_scaled_stats,
            },
            "slow": {
                "n_samples": int(slow.shape[0]),
                "raw": slow_raw_stats,
                "scaled": slow_scaled_stats,
            },
        },
        "partitions": partitions,
    }


def save_processed_data(
    normal: np.ndarray,
    slow: np.ndarray,
    normal_scaled: np.ndarray,
    slow_scaled: np.ndarray,
    plan: dict[str, np.ndarray],
    scaler: MinMaxScaler,
    selected_col_names: np.ndarray,
    config: dict[str, Any],
    window_size: int,
) -> None:
    """Save all artifacts to data/processed/ including manifest.json."""
    output_dir = Path(config["paths"]["data_processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build train_mask from plan (for manifest)
    window_size = int(config.get("data", {}).get("window_size", 256))
    train_mask = np.zeros(normal.shape[0], dtype=bool)
    for s in plan.get("train", np.array([], dtype=np.int64)):
        train_mask[s:s + window_size] = True

    # Remove legacy 5-file artifacts if present
    legacy_files = [
        "train.npy", "val_normal.npy", "val_anomaly.npy",
        "test_normal.npy", "test_anomaly.npy"
    ]
    for fname in legacy_files:
        fpath = output_dir / fname
        if fpath.exists():
            fpath.unlink()
            logger.info("Removed legacy artifact: %s", fname)

    # --- Save scaled recordings ---
    np.save(output_dir / "normal_scaled.npy", normal_scaled)
    logger.info("Saved normal_scaled.npy: shape=%s, dtype=%s", normal_scaled.shape, normal_scaled.dtype)

    np.save(output_dir / "slow_scaled.npy", slow_scaled)
    logger.info("Saved slow_scaled.npy: shape=%s, dtype=%s", slow_scaled.shape, slow_scaled.dtype)

    # --- Save partition plans ---
    for name, starts in plan.items():
        path = output_dir / f"plan_{name}.npy"
        np.save(path, starts)
        logger.info("Saved %s: %d windows", path, len(starts))

    # --- Save scaler ---
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved %s", scaler_path)

    # --- Save selected column names ---
    cols_path = output_dir / "selected_columns.npy"
    np.save(cols_path, selected_col_names)
    logger.info("Saved %s: %d columns", cols_path, selected_col_names.shape[0])

    # --- Build and save manifest ---
    part_cfg = config.get("data", {}).get("partitioning", {})
    train_mask = np.zeros(normal.shape[0], dtype=bool)
    for s in plan.get("train", np.array([], dtype=np.int64)):
        train_mask[s:s + window_size] = True
    
    part_cfg = config.get("data", {}).get("partitioning", {})
    manifest = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "window_size": window_size,
        "chunk_size": int(part_cfg.get("chunk_size", 1024)),
        "gap": int(part_cfg.get("gap", "window")) if part_cfg.get("gap", "window") != "window" else "window",
        "mode": part_cfg.get("mode", "interleaved"),
        "stride_multipliers": part_cfg.get("stride_multipliers", {}),
        "quotas": part_cfg.get("quotas", {}),
        "normal_n_samples": normal_scaled.shape[0],
        "slow_n_samples": slow_scaled.shape[0],
        "n_features": normal_scaled.shape[1],
        "partitions": {},
        "scaler_checksum": _checksum(np.concatenate([scaler.data_min_, scaler.data_max_])),
        "normal_checksum": _checksum(normal_scaled),
        "slow_checksum": _checksum(slow_scaled),
        "scaler_fit_rows": int(train_mask.sum()),
        "scaler_fit_min": scaler.data_min_.tolist(),
        "scaler_fit_max": scaler.data_max_.tolist(),
    }

    for name, starts in plan.items():
        n_samples = normal_scaled.shape[0] if "anomaly" not in name else slow_scaled.shape[0]
        covered = np.zeros(n_samples, dtype=bool)
        for s in starts:
            covered[s:s + window_size] = True
        guard_samples = n_samples - covered.sum()
        
        # Count out-of-range features for this partition
        if "anomaly" in name:
            arr = slow_scaled
            n_samples = slow_scaled.shape[0]
        else:
            arr = normal_scaled
            n_samples = normal_scaled.shape[0]
        
        covered = np.zeros(n_samples, dtype=bool)
        for s in starts:
            covered[s:s + window_size] = True
        vals = arr[covered] if np.any(covered) else np.array([])
        n_out = 0
        if len(vals) > 0:
            out_of_range = (vals < -1.0) | (vals > 1.0)
            n_out = int(out_of_range.sum())
            n_feats = int((out_of_range.sum(axis=0) > 0).sum()) if n_out > 0 else 0
        
        manifest["partitions"][name] = {
            "n_windows": len(starts),
            "n_covered_samples": int(covered.sum()),
            "guard_samples": int(guard_samples),
            "coverage_ratio": float(covered.sum()) / n_samples,
            "n_features_out_of_range": n_feats if 'n_feats' in locals() else 0,
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("Saved %s", manifest_path)

    # --- Save statistics ---
    stats = compute_statistics(normal, slow, normal_scaled, slow_scaled, plan, window_size, selected_col_names)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    logger.info("Saved %s", stats_path)
    manifest["stats_file"] = "stats.json"

    # --- Update params.yaml with discovered input_dim ---
    _update_params_input_dim(config, int(selected_col_names.shape[0]))

    # Rewrite manifest with stats_file reference
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _update_params_input_dim(config: dict[str, Any], input_dim: int) -> None:
    """Update params.yaml with the discovered input_dim from preprocessing."""
    project_root = Path(__file__).resolve().parents[2]
    params_path = project_root / "config" / "params.yaml"

    if not params_path.exists():
        logger.warning("params.yaml not found at %s, skipping input_dim update", params_path)
        return

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}

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
    """Run the full preprocessing pipeline: load → plan → scale → save."""
    logger.info("=" * 60)
    logger.info("AM01 — Fase 2 Preprocessing Pipeline (Interleaved)")
    logger.info("=" * 60)

    # --- Step 1: Load ---
    logger.info("[1/5] Loading Kuka data...")
    normal, slow, selected_cols = load_kuka_data(config)

    # --- Step 2: Plan partitions ---
    logger.info("[2/5] Planning interleaved partitions...")
    plan = plan_partitions_for_recordings(normal, slow, config)

    # --- Step 2b: Verify no leakage between partitions (R2a disgiunzione + R2b buffer) ---
    # NB: normal and slow are two different recordings with independent index
    # spaces — they must be verified separately, otherwise windows from
    # different recordings would appear to overlap.
    window_size = int(config.get("data", {}).get("window_size", 256))
    gap_cfg = config.get("data", {}).get("partitioning", {}).get("gap", "window")
    gap = window_size if gap_cfg == "window" else int(gap_cfg)
    normal_parts = {k: v for k, v in plan.items() if "anomaly" not in k}
    slow_parts = {k: v for k, v in plan.items() if "anomaly" in k}
    verify_no_leakage(normal_parts, window_size=window_size, gap=gap)
    verify_no_leakage(slow_parts, window_size=window_size, gap=gap)
    logger.info(
        "Partition plan verified: no shared samples, gap >= %d between partitions.", gap
    )

    # --- Step 3: Scale ---
    logger.info("[3/5] Fitting scaler on normal data...")
    scaler, train_mask = fit_scaler_on_train(normal, slow, plan, config, window_size)

    logger.info("[4/5] Scaling recordings...")
    normal_scaled, slow_scaled = scale_recordings(normal, slow, scaler)

    # --- Step 4: Save ---
    logger.info("[5/5] Saving processed data...")
    save_processed_data(
        normal, slow, normal_scaled, slow_scaled, plan, scaler, selected_cols,
        config, window_size
    )

    logger.info("Preprocessing completed successfully.")
    return plan


# ---------------------------------------------------------------------------
# CLI entry-point  (allows ``python -m src.data.preprocessing``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Config file")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"), help="Params file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config, args.params)
    run_preprocessing(cfg)