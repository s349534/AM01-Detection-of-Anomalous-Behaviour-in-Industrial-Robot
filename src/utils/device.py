"""Device/GPU utilities for robust CUDA handling."""
from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


def check_cuda_available(device: str | torch.device | None = None) -> torch.device:
    """
    Check if CUDA is actually available and usable (can allocate memory).

    This goes beyond `torch.cuda.is_available()` by attempting an actual
    tensor allocation, which fails if GPU is busy/unavailable.

    Parameters
    ----------
    device : str | torch.device | None
        Requested device. If None, auto-detects CUDA if available.

    Returns
    -------
    torch.device
        Validated device (cuda or cpu).

    Raises
    ------
    RuntimeError
        If CUDA is requested/selected but not actually usable.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)

    if dev.type == "cuda":
        # Try actual allocation - this fails if GPU is busy/unavailable
        try:
            torch.zeros(1, device=dev)
        except (RuntimeError, AssertionError) as e:
            raise RuntimeError(
                f"GPU {dev} non disponibile per allocazione memoria: {e}. "
                f"Verificare: nvidia-smi, nessun altro job su GPU, "
                f"CUDA_VISIBLE_DEVICES corretto, driver CUDA compatibile, "
                f"PyTorch compilato con CUDA."
            ) from e

    return dev


def get_device(device: str | torch.device | None = None, require_cuda: bool = False) -> torch.device:
    """
    Get validated device with optional CUDA requirement.

    Parameters
    ----------
    device : str | torch.device | None
        Requested device. None → auto-detect.
    require_cuda : bool
        If True, raise error if CUDA not usable (no fallback to CPU).

    Returns
    -------
    torch.device
        Validated device.

    Raises
    ------
    RuntimeError
        If require_cuda=True and CUDA not usable.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)

    if dev.type == "cuda":
        try:
            return check_cuda_available(dev)
        except RuntimeError as e:
            if require_cuda:
                raise
            logger.warning("CUDA non utilizzabile: %s. Fallback a CPU.", e)
            return torch.device("cpu")

    return dev