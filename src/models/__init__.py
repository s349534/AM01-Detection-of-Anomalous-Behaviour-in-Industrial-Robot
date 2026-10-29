"""ML models module for anomaly detection.

This module contains:
- autoencoder.py: Core autoencoder architecture (Conv1dEncoder, Conv1dDecoder, SequenceAutoencoder)
- training.py: Training utilities (train_epoch, evaluate_epoch, fit_autoencoder, EarlyStopping)
- inference.py: Inference utilities (compute_errors_on_loader)
- adversarial_ae.py: Adversarial autoencoder with discriminator
- compare_models.py: Model comparison utilities
"""
from src.models.autoencoder import (
    Conv1dEncoder,
    Conv1dDecoder,
    SequenceAutoencoder,
)
from src.models.training import (
    EarlyStopping,
    evaluate_epoch,
    fit_autoencoder,
    train_epoch,
)
from src.models.inference import compute_errors_on_loader
from src.models.adversarial_ae import (
    AdversarialAutoencoder,
    Discriminator,
    fit_aae,
)

__all__ = [
    "Conv1dEncoder",
    "Conv1dDecoder",
    "SequenceAutoencoder",
    "EarlyStopping",
    "train_epoch",
    "evaluate_epoch",
    "fit_autoencoder",
    "compute_errors_on_loader",
    "AdversarialAutoencoder",
    "Discriminator",
    "fit_aae",
]