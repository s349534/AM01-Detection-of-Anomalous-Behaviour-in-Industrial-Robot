"""Training utilities for the Sequence-Aware 1D-CNN Autoencoder.

Extracted from ``src/models/autoencoder.py`` to separate training logic
from model architecture. Provides reusable functions for training and
evaluation loops.

Public API
----------
train_epoch  — execute one training epoch
evaluate_epoch — execute one evaluation epoch
fit_autoencoder — complete training loop with early stopping
save_checkpoint — save model checkpoint with metadata
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler to monitor validation loss during training."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_state: dict[str, Any] | None = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int = 0) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0

        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best observed model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)



def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch on normal robot time-series windows.

    Parameters
    ----------
    model : nn.Module
        Model in training mode.
    dataloader : DataLoader
        Yields batches of ``(window, label)`` or just ``window``.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : nn.Module
        Loss function (e.g. ``nn.L1Loss()`` for MAE, ``nn.MSELoss()`` for MSE).
    device : torch.device
        Device for inference.

    Returns
    -------
    float
        Mean loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Handle both batch formats: (windows, labels) or windows
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        optimizer.zero_grad()
        x_rec = model(x)
        loss = criterion(x_rec, x)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model reconstruction loss on validation data.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode.
    dataloader : DataLoader
        Yields batches of ``(window, label)`` or just ``window``.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device for inference.

    Returns
    -------
    float
        Mean loss over the epoch.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            x_rec = model(x)
            loss = criterion(x_rec, x)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(1, total_samples)


def fit_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    min_delta: float = 1e-4,
    checkpoint_path: str | Path | None = None,  # kept for backward compat, unused
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
    loss: str | None = None,
) -> dict[str, list[float]]:
    """Complete training loop for the baseline SequenceAutoencoder.

    Includes Adam optimizer, early stopping on validation loss.
    Checkpoint serialization is handled by the caller (train_single_seed).

    Parameters
    ----------
    model : nn.Module
        The autoencoder model to train.
    train_loader : DataLoader
        DataLoader providing training batches (normal movements only).
    val_loader : DataLoader
        DataLoader providing validation batches.
    epochs : int, default=100
        Maximum training epochs.
    learning_rate : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=1e-4
        L2 regularization coefficient.
    patience : int, default=10
        Number of epochs without improvement before early stopping.
    min_delta : float, default=1e-4
        Minimum required improvement in validation loss.
    checkpoint_path : str | Path | None, default=None
        Deprecated — checkpoint saving is handled by the caller.
    device : str | torch.device | None
        Hardware device to train on.
    criterion : nn.Module | None
        Loss function instance. If None, ``loss`` is used to build it.
    loss : str | None
        Loss name ("mae" or "mse"), used to build ``criterion`` when ``criterion``
        is None.  Defaults to "mae" (robust to sensor saturation, see §7 #9).

    Returns
    -------
    dict
        History with keys ``"train_loss"`` (list[float]), ``"val_loss"`` (list[float]),
        and ``"best_epoch"`` (int — the epoch with the best validation loss).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.to(dev)

    if criterion is None:
        if loss is None or loss == "mae":
            criterion = nn.L1Loss()
        elif loss == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss '{loss}', must be 'mae' or 'mse'")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Simple early stopping (no separate EarlyStopping class)
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_state = model.state_dict()  # always track best weights in memory
    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    logger.info("Starting SequenceAutoencoder training on %s (%d epochs)", dev, epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, dev)
        val_loss = evaluate_epoch(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Epoch %3d/%3d — Train Loss: %.6f — Val Loss: %.6f",
                epoch, epochs, train_loss, val_loss,
            )

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d (best val loss: %.6f)",
                    epoch, best_val_loss,
                )
                break

    # Restore best weights
    model.load_state_dict(best_state)

    if checkpoint_path is not None:
        save_checkpoint(model, history, checkpoint_path)

    history["best_epoch"] = best_epoch
    return history


def save_checkpoint(
    model: nn.Module,
    history: dict,
    path: str | Path,
    **metadata: Any,
) -> None:
    """Save model checkpoint with optional metadata.

    Args:
        model: The model to save.
        history: Training history dictionary.
        path: Path to save the checkpoint.
        **metadata: Additional metadata to include in the checkpoint
            (e.g., input_dim, window_size, latent_dim, model_config, seed).
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)