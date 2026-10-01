"""Search spaces for hyperparameter validation (project_plan.md §4.8).

Each ``get_search_space_*`` function returns a dict in the format expected by
``sklearn.model_selection.ParameterSampler``.  Keys use dot-notation matching
the nested structure of ``config/params.yaml``.
"""
from __future__ import annotations

from typing import Any


def get_search_space_ae() -> dict[str, Any]:
    """AE hyperparameter search space (§4.8.2).

    Returns
    -------
    dict
        Dict with dot-notation keys compatible with ``ParameterSampler``
        and the config structure in ``params.yaml``::

            {
                "model.window_size": [16, 32, 64, 128],
                "model.latent_dim": [8, 12, 16, 24, 32],
            }

    Notes
    -----
    - Encoder architecture is now FIXED to 3 layers with channels (64, 32, 16).
      The ``encoder_channels`` hyperparameter has been removed from the search space.
    - Decoder is auto-derived from encoder channels (reversed), not configured separately.
    - Window size must be a multiple of 4 (due to 2 pooling stages = factor 4 reduction).
      All values in the search space (16, 32, 64, 128) satisfy this constraint.
    - Total grid: 4 × 5 = 20 combinations. All can be sampled exhaustively.
    - W range: powers of 2 from 16 to 128, default 16. Covers short windows (local patterns)
      to long windows (slow trends). Default 16 aligned with params.yaml.
    """
    return {
        "model.window_size": [16, 32, 64, 128],  # all multiples of 4
        "model.latent_dim": [8, 12, 16, 24, 32],
        # encoder_channels REMOVED - fixed architecture (64, 32, 16)
    }


def merge_search_sample(
    base_config: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge a ParameterSampler sample into a base config dict.

    Handles dot-notation keys (e.g. ``model.window_size``) by traversing
    and creating nested dicts as needed.

    Parameters
    ----------
    base_config : dict
        The original config (from ``load_config``).  Deep-copied, not mutated.
    sample : dict
        A single sample from ``ParameterSampler`` (dot-notation keys).

    Returns
    -------
    dict
        A new config dict with the sampled HP overlaid onto the base.
    """
    import copy

    config = copy.deepcopy(base_config)

    for dot_key, value in sample.items():
        parts = dot_key.split(".")
        target: dict[str, Any] = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    return config


def get_search_space_aae() -> dict[str, Any]:
    """AAE-specific hyperparameter search space (§4.8.3).

    Only the 2 AAE-specific HPs are searched; all AE-derived HPs are fixed
    to ``HP_AE_best``.  These are produced after Fase 3.1 completes.

    Returns
    -------
    dict
        Search space for ``reconstruction_weight`` and ``adversarial_weight``.
    """
    return {
        "training.reconstruction_weight": [0.5, 0.7, 1.0, 1.4, 2.0],
        "training.adversarial_weight": [0.01, 0.05, 0.1, 0.15, 0.3, 0.5],
    }