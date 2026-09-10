"""ML models module for anomaly detection.

This module contains:
- autoencoder.py: Sequence-aware 1D-CNN autoencoder
- adversarial_ae.py: Adversarial autoencoder with discriminator
- compare_models.py: Model comparison utilities
"""
from src.models.autoencoder import (
    Conv1dDecoder,
    Conv1dEncoder,
    EarlyStopping,
    SequenceAutoencoder,
    fit_autoencoder,
)

__all__ = [
    "Conv1dEncoder",
    "Conv1dDecoder",
    "SequenceAutoencoder",
    "EarlyStopping",
    "fit_autoencoder",
]