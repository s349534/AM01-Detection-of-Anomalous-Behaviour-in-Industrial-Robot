"""Sequence-Aware 1D-CNN Autoencoder for Anomaly Detection.

This module implements the baseline autoencoder architecture designed for the
Kuka industrial robot time-series dataset.

Architecture overview (see docs/project_plan.md §4.1.2):
--------------------------------------------------------
Encoder (1D-Conv):
    Input:  (B, input_dim=82, W=16)
       ↓    Conv1d(82 → 128, kernel=5, padding=2) + ReLU
            MaxPool1d(kernel_size=2) → (B, 128, W/2=8)
       ↓    Conv1d(128 → 64, kernel=3, padding=1) + ReLU
            AdaptiveAvgPool1d(1) → (B, 64, 1)
       ↓    Flatten → (B, 64)
            Linear(64 → latent_dim=16)
    Output: z ∈ ℝ^(B, 16)

Decoder (Symmetric 1D-ConvTranspose):
    Input:  z ∈ ℝ^(B, 16)
       ↓    Linear(16 → 64 * (W // 2)) + ReLU
            Reshape → (B, 64, W/2=8)
       ↓    ConvTranspose1d(64 → 128, kernel=4, stride=2, padding=1) + ReLU → (B, 128, W=16)
       ↓    Conv1d(128 → input_dim=82, kernel=3, padding=1)
    Output: x̂ ∈ ℝ^(B, input_dim=82, W=16)

Input Layout Support:
    The model accepts both (B, input_dim, W) and (B, W, input_dim) tensors.
    If the caller provides (B, W, input_dim) — as produced by standard DataLoaders
    wrapping KukaDataset — the input is transposed internally and the reconstructed
    tensor is returned in the same (B, W, input_dim) layout.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import get_param, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Encoder
# ---------------------------------------------------------------------------
class Conv1dEncoder(nn.Module):
    """1D-Convolutional Encoder for multivariate time-series windows.

    Compresses an input tensor of shape ``(B, input_dim, W)`` into a latent
    vector ``z`` of shape ``(B, latent_dim)``.

    Parameters
    ----------
    input_dim : int, default=82
        Number of sensor channels (features).
    window_size : int, default=16
        Number of timesteps per window (W).
    conv_channels : tuple[int, int], default=(128, 64)
        Number of output channels for the two Conv1d layers.
    conv_kernels : tuple[int, int], default=(5, 3)
        Kernel sizes for the two Conv1d layers.
    pool_size : int, default=2
        Kernel size and stride for the intermediate MaxPool1d.
    latent_dim : int, default=16
        Dimension of the compressed latent representation z.
    activation : str, default="relu"
        Activation function ("relu" or "leaky_relu").
    """

    def __init__(
        self,
        input_dim: int = 82,
        window_size: int = 16,
        conv_channels: tuple[int, int] = (128, 64),
        conv_kernels: tuple[int, int] = (5, 3),
        pool_size: int = 2,
        latent_dim: int = 16,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.pool_size = pool_size
        self.latent_dim = latent_dim

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ReLU

        c1, c2 = conv_channels
        k1, k2 = conv_kernels

        self.net = nn.Sequential(
            # Layer 1: preserves temporal length L=W
            nn.Conv1d(input_dim, c1, kernel_size=k1, padding=k1 // 2),
            act_cls(),
            # Pool 1: reduces temporal length to W // pool_size
            nn.MaxPool1d(kernel_size=pool_size),
            # Layer 2: preserves reduced length
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            act_cls(),
            # Adaptive pooling: guarantees temporal dimension of 1 regardless of W
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # Dense projection to latent bottleneck
            nn.Linear(c2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence tensor to latent vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim, W)``.

        Returns
        -------
        torch.Tensor
            Latent representation z of shape ``(B, latent_dim)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Decoder
# ---------------------------------------------------------------------------
class Conv1dDecoder(nn.Module):
    """Symmetric 1D-Transposed Convolutional Decoder.

    Reconstructs the original sequence window ``(B, input_dim, W)`` from a
    latent vector ``z`` of shape ``(B, latent_dim)``.

    Parameters
    ----------
    latent_dim : int, default=16
        Dimension of the input latent representation z.
    input_dim : int, default=82
        Number of output sensor channels.
    window_size : int, default=16
        Target temporal length (W) to reconstruct.
    conv_channels : tuple[int, int], default=(64, 128)
        Channel dimensions for transposed and regular convolutions.
    activation : str, default="relu"
        Activation function ("relu" or "leaky_relu").
    """

    def __init__(
        self,
        latent_dim: int = 16,
        input_dim: int = 82,
        window_size: int = 16,
        conv_channels: tuple[int, int] = (64, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.window_size = window_size
        self.conv_channels = conv_channels

        # The intermediate temporal length before upsampling is W // 2
        self.half_w = max(1, window_size // 2)
        c1, c2 = conv_channels

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ReLU

        # Project latent vector to flat intermediate feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, c1 * self.half_w),
            act_cls(),
        )

        # Upsampling via ConvTranspose1d: (B, c1, half_w) → (B, c2, W)
        # For L_in = W // 2, kernel=4, stride=2, padding=1 gives:
        # L_out = (L_in - 1)*2 - 2*1 + 4 = 2 * L_in = W (for even W).
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(c1, c2, kernel_size=4, stride=2, padding=1),
            act_cls(),
            # Final 1x1-equivalent smoothing/projection to original sensor count
            nn.Conv1d(c2, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to sequence tensor.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed sequence tensor of shape ``(B, input_dim, W)``.
        """
        batch_size = z.size(0)
        h = self.fc(z)
        h = h.view(batch_size, self.conv_channels[0], self.half_w)
        x_rec = self.deconv(h)

        # Safety adjustment if window_size was odd
        if x_rec.size(-1) != self.window_size:
            x_rec = nn.functional.interpolate(
                x_rec, size=self.window_size, mode="linear", align_corners=False
            )

        return x_rec


# ---------------------------------------------------------------------------
# 3. Sequence Autoencoder (Encoder + Decoder orchestrator)
# ---------------------------------------------------------------------------
class SequenceAutoencoder(nn.Module):
    """Sequence-Aware 1D-CNN Autoencoder for time-series anomaly detection.

    Composes :class:`Conv1dEncoder` and :class:`Conv1dDecoder`.  Provides
    methods for end-to-end forward pass, standalone encoding/decoding, and
    sample-wise anomaly score calculation.

    Parameters
    ----------
    input_dim : int, default=82
        Number of input channels (sensor features).
    window_size : int, default=16
        Number of consecutive timesteps per window (W).
    latent_dim : int, default=16
        Dimension of latent bottleneck vector z.
    encoder_channels : tuple[int, int], default=(128, 64)
        Conv channels in encoder.
    encoder_kernels : tuple[int, int], default=(5, 3)
        Conv kernel sizes in encoder.
    pool_size : int, default=2
        Pooling kernel size.
    decoder_channels : tuple[int, int], default=(64, 128)
        Conv channels in decoder.
    activation : str, default="relu"
        Activation function.
    """

    def __init__(
        self,
        input_dim: int = 82,
        window_size: int = 16,
        latent_dim: int = 16,
        encoder_channels: tuple[int, int] = (128, 64),
        encoder_kernels: tuple[int, int] = (5, 3),
        pool_size: int = 2,
        decoder_channels: tuple[int, int] = (64, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.encoder = Conv1dEncoder(
            input_dim=input_dim,
            window_size=window_size,
            conv_channels=encoder_channels,
            conv_kernels=encoder_kernels,
            pool_size=pool_size,
            latent_dim=latent_dim,
            activation=activation,
        )

        self.decoder = Conv1dDecoder(
            latent_dim=latent_dim,
            input_dim=input_dim,
            window_size=window_size,
            conv_channels=decoder_channels,
            activation=activation,
        )

    def _format_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Verify and standardise input shape to ``(B, input_dim, W)``.

        Accepts either:
        - ``(B, input_dim, W)``: standard PyTorch Conv1d format
        - ``(B, W, input_dim)``: standard DataLoader / sequence format

        Returns
        -------
        x_formatted : torch.Tensor
            Tensor with shape ``(B, input_dim, W)``.
        was_transposed : bool
            True if the input was in ``(B, W, input_dim)`` format.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3-D tensor (B, W, D) or (B, D, W), got ndim={x.ndim} "
                f"with shape {x.shape}"
            )

        b, d1, d2 = x.shape

        # Case 1: already (B, input_dim, W)
        if d1 == self.input_dim and d2 == self.window_size:
            return x, False

        # Case 2: sequence layout (B, W, input_dim) from DataLoader
        if d1 == self.window_size and d2 == self.input_dim:
            return x.transpose(1, 2), True

        # Case 3: flexible W if d1 == input_dim
        if d1 == self.input_dim:
            return x, False

        # Case 4: flexible W if d2 == input_dim
        if d2 == self.input_dim:
            return x.transpose(1, 2), True

        raise ValueError(
            f"Input shape {x.shape} is incompatible with expected "
            f"input_dim={self.input_dim} and window_size={self.window_size}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent vector z.

        Accepts both ``(B, input_dim, W)`` and ``(B, W, input_dim)``.
        """
        x_conv, _ = self._format_input(x)
        return self.encoder(x_conv)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to reconstructed sequence ``(B, input_dim, W)``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end autoencoder forward pass.

        Reconstructs the input sequence. Preserves caller layout: if called
        with ``(B, W, input_dim)``, returns ``(B, W, input_dim)``. If called
        with ``(B, input_dim, W)``, returns ``(B, input_dim, W)``.
        """
        x_conv, was_transposed = self._format_input(x)
        z = self.encoder(x_conv)
        x_hat = self.decoder(z)

        if was_transposed:
            return x_hat.transpose(1, 2)
        return x_hat

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: Literal["none", "mean", "sample"] = "sample",
        metric: Literal["mse", "mae"] = "mse",
    ) -> torch.Tensor:
        """Compute reconstruction error between input and its autoencoder reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor (either layout).
        reduction : {"none", "mean", "sample"}, default="sample"
            - "none": element-wise error tensor matching x shape.
            - "mean": scalar average error over the entire batch.
            - "sample": 1D tensor of shape ``(B,)`` containing the mean error
              per window. This is the standard anomaly score per sample.
        metric : {"mse", "mae"}, default="mse"
            Distance metric used (MSE: squared difference, MAE: absolute difference).

        Returns
        -------
        torch.Tensor
            Reconstruction error tensor according to requested reduction.
        """
        with torch.no_grad():
            x_hat = self.forward(x)

        if metric == "mse":
            diff = (x - x_hat) ** 2
        elif metric == "mae":
            diff = torch.abs(x - x_hat)
        else:
            raise ValueError(f"Unsupported metric: '{metric}', must be 'mse' or 'mae'")

        if reduction == "none":
            return diff
        if reduction == "mean":
            return diff.mean()
        if reduction == "sample":
            # Average across all feature and timestep dimensions for each sample
            dims = tuple(range(1, diff.ndim))
            return diff.mean(dim=dims)

        raise ValueError(
            f"Unsupported reduction: '{reduction}', must be 'none', 'mean', or 'sample'"
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> SequenceAutoencoder:
        """Instantiate a SequenceAutoencoder from a configuration dictionary.

        If config is None, loads default config via ``src.utils.config.load_config()``.
        """
        if config is None:
            config = load_config()

        input_dim = int(get_param(config, "model.input_dim", 82))
        window_size = int(get_param(config, "model.window_size", 16))
        latent_dim = int(get_param(config, "model.latent_dim", 16))

        enc_channels = tuple(
            get_param(config, "model.encoder.conv_channels", [128, 64])
        )
        enc_kernels = tuple(
            get_param(config, "model.encoder.conv_kernels", [5, 3])
        )
        pool_size = int(get_param(config, "model.encoder.pool_size", 2))
        dec_channels = tuple(
            get_param(config, "model.decoder.conv_channels", [64, 128])
        )
        activation = str(get_param(config, "model.encoder.activation", "relu"))

        return cls(
            input_dim=input_dim,
            window_size=window_size,
            latent_dim=latent_dim,
            encoder_channels=enc_channels,  # type: ignore[arg-type]
            encoder_kernels=enc_kernels,    # type: ignore[arg-type]
            pool_size=pool_size,
            decoder_channels=dec_channels,  # type: ignore[arg-type]
            activation=activation,
        )


# ---------------------------------------------------------------------------
# 4. Early Stopping Helper
# ---------------------------------------------------------------------------
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

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0

        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best observed model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# 5. Modular Training & Evaluation Functions
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch on normal robot time-series windows."""
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
    """Evaluate model reconstruction loss on validation data."""
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
    model: SequenceAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    min_delta: float = 1e-4,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    criterion: nn.Module | None = None,
) -> dict[str, list[float]]:
    """Complete training loop for the baseline SequenceAutoencoder.

    Includes Adam optimizer, early stopping on validation loss, and optional
    checkpoint serialization.

    Parameters
    ----------
    model : SequenceAutoencoder
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
        Path to save the best model weights.
    device : str | torch.device
        Hardware device to train on.
    criterion : nn.Module | None
        Loss function, defaults to nn.MSELoss().

    Returns
    -------
    dict[str, list[float]]
        History dictionary containing "train_loss" and "val_loss" per epoch.
    """
    dev = torch.device(device)
    model.to(dev)

    if criterion is None:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    logger.info("Starting SequenceAutoencoder training on %s (%d epochs)", dev, epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, dev)
        val_loss = evaluate_epoch(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%3d — Train Loss: %.6f — Val Loss: %.6f",
                epoch, epochs, train_loss, val_loss,
            )

        if early_stopping(val_loss, model):
            logger.info(
                "Early stopping triggered at epoch %d (best val loss: %.6f)",
                epoch, -early_stopping.best_score if early_stopping.best_score else 0.0,
            )
            break

    # Restore best checkpoint weights
    early_stopping.restore_best_weights(model)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "input_dim": model.input_dim,
                "window_size": model.window_size,
                "latent_dim": model.latent_dim,
            },
            ckpt,
        )
        logger.info("Saved best checkpoint to %s", ckpt)

    return history