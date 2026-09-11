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
                "model.window_size": [8, 12, 16, 24, 32],
                "model.latent_dim": [8, 12, 16, 24, 32],
                "model.encoder.conv_channels": [[64, 32], [128, 64]],
            }

    Notes
    -----
    - ``encoder_channels = [128, 64, 32]`` (3-layer) is **excluded** because
      ``Conv1dEncoder`` and ``Conv1dDecoder`` are hardcoded for exactly 2 conv
      layers (they do ``c1, c2 = conv_channels``).  See ``src/models/autoencoder.py``
      lines 95 and 169.
    - Total grid: 5 × 5 × 2 = 50 combinations. 20 are sampled per validation run.
    """
    return {
        "model.window_size": [8, 12, 16, 24, 32],
        "model.latent_dim": [8, 12, 16, 24, 32],
        "model.encoder.conv_channels": [[64, 32], [128, 64]],
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
