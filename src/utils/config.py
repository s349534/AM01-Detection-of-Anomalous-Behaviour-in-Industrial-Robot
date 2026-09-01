"""Unified configuration loader for the AM01 project.

Loads and merges ``config/config.yaml`` (project metadata, paths, environment)
and ``config/params.yaml`` (model/training/data hyperparameters) into a single
dictionary.  ``params.yaml`` keys take precedence in case of conflict.

Public API
----------
load_config  — merge the two YAML files (with sensible defaults).
get_param    — dot-notation accessor (``get_param(cfg, "model.input_dim")``).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Project root: src/utils/config.py → utils/ → src/ → project_root/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_DEFAULT_PARAMS_PATH = _PROJECT_ROOT / "config" / "params.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a single YAML file, returning {} if missing."""
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: Path | None = None,
    params_path: Path | None = None,
) -> dict[str, Any]:
    """Load and merge ``config.yaml`` + ``params.yaml`` into one dict.

    Parameters
    ----------
    config_path :
        Path to ``config/config.yaml`` (project metadata, paths, device).
        Defaults to ``config/config.yaml`` relative to project root.
    params_path :
        Path to ``config/params.yaml`` (model/training/data hyperparameters).
        Defaults to ``config/params.yaml`` relative to project root.

    Returns
    -------
    dict
        A deep-copied dictionary with keys from both files merged.
        Nested dicts are merged recursively; ``params.yaml`` overrides
        ``config.yaml`` on key conflicts.
    """
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    params_path = Path(params_path) if params_path else _DEFAULT_PARAMS_PATH

    config = _load_yaml(config_path)
    params = _load_yaml(params_path)

    merged: dict[str, Any] = copy.deepcopy(config)

    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge ``override`` into ``base`` (override wins)."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                _deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    _deep_merge(merged, params)
    return merged


def get_param(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Retrieve a nested config value using dot-notation.

    Examples
    --------
    >>> get_param(cfg, "model.input_dim", 82)
    >>> get_param(cfg, "training.batch_size", 256)
    >>> get_param(cfg, "data.window_size", 16)
    """
    parts = key.split(".")
    value: Any = config
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
