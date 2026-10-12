"""Reproducibility utilities for setting random seeds across the AM01 project.

Centralised here so that every training / validation script uses the **same**
seed-setting logic — no more copy-pasted ``_set_seed`` functions.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA) with the
    same value.  Call this at the very start of every experiment / training
    run that involves stochasticity.

    Parameters
    ----------
    seed : int
        The seed value to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
