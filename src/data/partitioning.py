"""Partitioning module for AM01 anomaly detection.

Implements interleaved chunk-based partitioning with stride-controlled windows
and guard bands to prevent leakage between partitions.

Public API:
    PartitionSpec        — dataclass describing one partition's quota and stride
    plan_partitions      — compute window start indices for all partitions
    verify_no_leakage    — verify R2a (disjointness) and R2b (temporal gap)
    plan_from_config     — build plan for normal and slow recordings from config
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PartitionSpec:
    """Specification for one partition.

    Attributes:
        name: Partition identifier (e.g., "train", "val_normal").
        quota: Fraction of chunks assigned to this partition.
        stride_multiplier: Window stride as a multiple of W (e.g., 0.25 = W/4).
    """
    name: str
    quota: float
    stride_multiplier: float


def _exact_counts(n_chunks: int, quotas: np.ndarray) -> np.ndarray:
    """Compute exact integer counts per partition using largest remainder method.

    Args:
        n_chunks: Total number of chunks.
        quotas: Array of quotas (sum = 1).

    Returns:
        Integer counts summing to n_chunks, each ≈ quota * n_chunks.
    """
    quotas = np.asarray(quotas, dtype=float)
    counts = np.floor(quotas * n_chunks).astype(int)
    rem = n_chunks - counts.sum()
    if rem > 0:
        frac = quotas * n_chunks - counts
        idx = np.argsort(-frac)[:rem]
        counts[idx] += 1
    return counts


def assign_chunks_spread(n_chunks: int, quotas: list[float]) -> np.ndarray:
    """Assign chunks to partitions in an interleaved (spread) manner.

    Uses a largest-deficit (Bresenham-like) algorithm to distribute each
    partition's chunks evenly across the timeline while respecting exact counts.

    Args:
        n_chunks: Number of chunks.
        quotas: List of quotas for each partition (sum = 1).

    Returns:
        Array of shape (n_chunks,) with partition index (0..P-1) for each chunk.
    """
    quotas = np.asarray(quotas, dtype=float)
    counts = _exact_counts(n_chunks, quotas)
    owner = np.empty(n_chunks, dtype=np.int64)
    assigned = np.zeros(len(quotas), dtype=int)
    remaining = counts.copy()

    for c in range(n_chunks):
        deficit = (c + 1) * quotas - assigned
        deficit[remaining == 0] = -np.inf
        p = int(np.argmax(deficit))
        owner[c] = p
        assigned[p] += 1
        remaining[p] -= 1

    return owner


def assign_chunks_chronological(n_chunks: int, quotas: list[float]) -> np.ndarray:
    """Assign chunks to partitions in contiguous blocks (chronological mode).

    Useful as an ablation baseline (equivalent to the old temporal split).

    Args:
        n_chunks: Number of chunks.
        quotas: List of quotas for each partition (sum = 1).

    Returns:
        Array of shape (n_chunks,) with partition index for each chunk.
    """
    counts = _exact_counts(n_chunks, np.asarray(quotas, dtype=float))
    owner = np.empty(n_chunks, dtype=np.int64)
    start = 0
    for p, c in enumerate(counts):
        owner[start:start + c] = p
        start += c
    return owner


def plan_partitions(
    n_samples: int,
    window_size: int,
    chunk_size: int,
    gap: int,
    specs: list[PartitionSpec],
    mode: str = "interleaved",
) -> dict[str, np.ndarray]:
    """Compute window start indices for all partitions.

    Args:
        n_samples: Total number of samples in the recording.
        window_size: Window length W.
        chunk_size: Chunk length L (samples). Must satisfy L >= 2*W + max(stride).
        gap: Guard band size (samples) applied at partition changes.
        specs: List of PartitionSpec for each partition.
        mode: "interleaved" (default) or "chronological".

    Returns:
        Dict mapping partition name -> sorted array of window start indices (int64).

    Raises:
        ValueError: If chunk_size is too small for the given window_size and strides.
        ValueError: If mode is not "interleaved" or "chronological".
    """
    if mode not in ("interleaved", "chronological"):
        raise ValueError(f"mode must be 'interleaved' or 'chronological', got '{mode}'")

    if chunk_size < 2 * window_size + max(int(s.stride_multiplier * window_size) for s in specs):
        raise ValueError(
            f"chunk_size={chunk_size} too small for W={window_size} and strides. "
            f"Need chunk_size >= 2*W + max(stride) = {2 * window_size + max(int(s.stride_multiplier * window_size) for s in specs)}"
        )

    n_chunks = -(-n_samples // chunk_size)
    bounds = np.linspace(0, n_samples, n_chunks + 1, dtype=np.int64)
    chunk_starts = bounds[:-1]
    chunk_ends = bounds[1:]
    chunk_lengths = chunk_ends - chunk_starts

    quotas = [s.quota for s in specs]

    if mode == "interleaved":
        owner = assign_chunks_spread(n_chunks, quotas)
    else:  # chronological
        owner = assign_chunks_chronological(n_chunks, quotas)

    strides = [max(1, round(s.stride_multiplier * window_size)) for s in specs]

    starts_by_partition: dict[str, np.ndarray] = {}

    for p_idx, (spec, stride) in enumerate(zip(specs, strides)):
        mask = (owner == p_idx)
        chunks_idx = np.flatnonzero(mask)
        if len(chunks_idx) == 0:
            starts_by_partition[spec.name] = np.array([], dtype=np.int64)
            continue

        # Guard band at the end of this chunk if the next chunk has a different owner
        next_idx = np.minimum(chunks_idx + 1, n_chunks - 1)
        next_owner = owner[next_idx]
        guard = np.where(next_owner != owner[chunks_idx], gap, 0)

        usable = chunk_lengths[chunks_idx] - guard
        last_local = usable - window_size

        # Shared local grid (same stride for all chunks of this partition)
        max_last = int(last_local.max())
        if max_last < 0:
            starts_by_partition[spec.name] = np.array([], dtype=np.int64)
            continue

        local = np.arange(0, max_last + 1, stride)
        mask_local = local[None, :] <= last_local[:, None]
        base = chunk_starts[chunks_idx][:, None] + local[None, :]
        starts = base[mask_local]
        starts_by_partition[spec.name] = np.sort(starts).astype(np.int64)

    return starts_by_partition


def verify_no_leakage(
    starts_by_partition: dict[str, np.ndarray],
    window_size: int,
    gap: int,
) -> None:
    """Verify partitioning satisfies R2a (disjointness) and R2b (temporal gap).

    Args:
        starts_by_partition: Dict from partition name to window start indices.
        window_size: Window length W.
        gap: Required minimum gap between windows of different partitions.

    Raises:
        ValueError: If any invariant is violated.
    """
    all_starts = []
    all_ends = []
    all_labels = []

    for label, starts in starts_by_partition.items():
        if len(starts) == 0:
            continue
        all_starts.append(starts)
        all_ends.append(starts + window_size)
        all_labels.extend([label] * len(starts))

    if not all_starts:
        return

    all_starts = np.concatenate(all_starts)
    all_ends = np.concatenate(all_ends)
    all_labels = np.array(all_labels)

    # Sort by start time
    order = np.argsort(all_starts)
    all_starts = all_starts[order]
    all_ends = all_ends[order]
    all_labels = all_labels[order]

    # R2a: no sample shared between different partitions
    for i in range(len(all_starts) - 1):
        if all_labels[i] != all_labels[i + 1]:
            # Different partitions must not overlap
            if not (all_ends[i] <= all_starts[i + 1]):
                raise ValueError(
                    f"Overlap between {all_labels[i]} (end={all_ends[i]}) "
                    f"and {all_labels[i + 1]} (start={all_starts[i + 1]})"
                )
            # R2b: gap >= gap between different partitions
            if not (all_starts[i + 1] - all_ends[i] >= gap):
                raise ValueError(
                    f"Insufficient gap between {all_labels[i]} (end={all_ends[i]}) "
                    f"and {all_labels[i + 1]} (start={all_starts[i + 1]}): "
                    f"got {all_starts[i + 1] - all_ends[i]}, need >= {gap}"
                )
            # R2b: gap >= gap between different partitions
            # all_ends is start + window_size (first sample after window)
            # so unused samples = all_starts[i+1] - all_ends[i]
            assert all_starts[i + 1] - all_ends[i] >= gap, (
                f"Insufficient gap between {all_labels[i]} (end={all_ends[i]}) "
                f"and {all_labels[i + 1]} (start={all_starts[i + 1]}): "
                f"got {all_starts[i + 1] - all_ends[i]}, need >= {gap}"
            )


def _specs_from_config(config: dict[str, Any], source: str) -> list[PartitionSpec]:
    """Build PartitionSpec list from config for a given source (normal or slow)."""
    part_cfg = config.get("data", {}).get("partitioning", {})
    quotas = part_cfg.get("quotas", {}).get(source, {})
    stride_mult = part_cfg.get("stride_multipliers", {})

    if source == "normal":
        return [
            PartitionSpec("train", quotas.get("train", 0.70), stride_mult.get("train", 0.25)),
            PartitionSpec("val_normal", quotas.get("val_normal", 0.15), stride_mult.get("val_normal", 0.50)),
            PartitionSpec("test_normal", quotas.get("test_normal", 0.15), stride_mult.get("test_normal", 0.50)),
        ]
    elif source == "slow":
        return [
            PartitionSpec("val_anomaly", quotas.get("val_anomaly", 0.50), stride_mult.get("val_anomaly", 0.50)),
            PartitionSpec("test_anomaly", quotas.get("test_anomaly", 0.50), stride_mult.get("test_anomaly", 0.50)),
        ]
    else:
        raise ValueError(f"Unknown source: {source}")


def plan_from_config(
    config: dict[str, Any],
    source: str,
    normal_n: int,
    slow_n: int,
) -> dict[str, np.ndarray]:
    """Build partition plan for a recording from config.

    Args:
        config: Full config dict (merged config.yaml + params.yaml).
        source: "normal" or "slow".
        normal_n: Number of samples in normal recording.
        slow_n: Number of samples in slow recording.

    Returns:
        Dict mapping partition name -> window start indices.
    """
    part_cfg = config.get("data", {}).get("partitioning", {})
    window_size = int(config.get("data", {}).get("window_size", 256))
    chunk_size = int(part_cfg.get("chunk_size", 1024))
    gap_cfg = part_cfg.get("gap", "window")
    gap = window_size if gap_cfg == "window" else int(gap_cfg)
    mode = part_cfg.get("mode", "interleaved")

    specs = _specs_from_config(config, source)
    n_samples = normal_n if source == "normal" else slow_n

    return plan_partitions(n_samples, window_size, chunk_size, gap, specs, mode)