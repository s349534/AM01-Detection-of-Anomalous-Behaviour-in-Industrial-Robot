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
                "data.window_size": [32, 64, 128],
                "model.latent_dim": [8, 16, 32],
            }

    Notes
    -----
    - Encoder architecture is now FIXED to 3 layers with channels (64, 32, 16).
      The ``encoder_channels`` hyperparameter has been removed from the search space.
    - Decoder is auto-derived from encoder channels (reversed), not configured separately.
    - Window size must be a multiple of 4 (due to 2 pooling stages = factor 4 reduction).
      All values in the search space (32, 64, 128) satisfy this constraint.
    - Total grid: 3 × 3 = 9 combinations. All can be sampled exhaustively.
    - W range: powers of 2 from 32 to 128, default 128. Covers short windows (local patterns)
      to long windows (slow trends). Default 128 aligned with params.yaml.
    """
    return {
        "data.window_size": [32, 64, 128],  # all multiples of 4
        "model.latent_dim": [8, 16, 32],
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
    """Search space for adversarial autoencoder validation (§4.8.3 + §4.1).

    AE-derived HPs (``W``, ``latent_dim``, ``encoder_channels``) are **fixed**
    to ``HP_AE_best`` and not re-validated.  Only AAE-specific HPs are searched:

    1. ``training.reconstruction_weight`` — MSE loss multiplier
    2. ``training.adversarial_weight``    — BCE loss multiplier

    Discriminator hidden layers are FIXED to [16, 8] (validated best from prior runs)
    and not included in the search space.

    Grid = 5 × 6 = 30 combos; all run exhaustively per validation run.

    Notes
    -----
    - Adversarial weight range explores below current best (0.01) to test if
      performance continues improving, and includes higher values (0.1, 0.2) to
      confirm the cliff observed at >=0.05. Log-scale spacing for lower values.
    - Reconstruction weight range is denser in the optimal zone (0.7-1.4) identified
      from prior validation results.
    """
    return {
        "training.reconstruction_weight": [0.5, 0.7, 0.9, 1.1, 1.4],
        "training.adversarial_weight": [0.001, 0.005, 0.01, 0.02, 0.1, 0.2],
    }
