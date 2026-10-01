"""Sequence-Aware 1D-CNN Autoencoder for Anomaly Detection.

This module implements the baseline autoencoder architecture designed for the
Kuka industrial robot time-series dataset.

Architecture overview (see docs/project_plan.md §4.1.2):
---------------------------------------------------------
Encoder (1D-Conv, 3 layers, 2 pooling stages):
    Input:  (B, input_dim=82, W)  with W multiple of 4
       ↓    Conv1d(82 → 64, kernel=5, padding=2) + ReLU
            MaxPool1d(kernel_size=2) → (B, 64, W/2)
       ↓    Conv1d(64 → 32, kernel=3, padding=1) + ReLU
            MaxPool1d(kernel_size=2) → (B, 32, W/4)
       ↓    Conv1d(32 → 16, kernel=3, padding=1) + ReLU
            → (B, 16, W/4)
       ↓    Flatten → (B, 16 * W/4) = (B, 4W)
            Linear(4W → latent_dim)
    Output: z ∈ ℝ^(B, latent_dim)

Decoder (Auto-derived symmetric 1D-ConvTranspose):
    Input:  z ∈ ℝ^(B, latent_dim)
       ↓    Linear(latent_dim → 16 * (W/4)) + ReLU
            Reshape → (B, 16, W/4)
       ↓    ConvTranspose1d(16 → 32, kernel=4, stride=2, padding=1) + ReLU → (B, 32, W/2)
       ↓    ConvTranspose1d(32 → 64, kernel=4, stride=2, padding=1) + ReLU → (B, 64, W)
       ↓    Conv1d(64 → input_dim=82, kernel=3, padding=1)
    Output: x̂ ∈ ℝ^(B, input_dim=82, W)

    Decoder channels are automatically derived as the reverse of encoder
    conv_channels (excluding input_dim). No separate decoder_channels config.

Input Layout Support:
    The model accepts both (B, input_dim, W) and (B, W, input_dim) tensors.
    If the caller provides (B, W, input_dim) — as produced by standard DataLoaders
    wrapping KukaDataset — the input is transposed internally and the reconstructed
    tensor is returned in the same (B, W, input_dim) layout.

Fixed Architecture Parameters:
    - Encoder conv_channels: (64, 32, 16) — funnel progression
    - Encoder conv_kernels: (5, 3, 3)
    - Pooling: 2 stages of MaxPool1d(2) after conv layers 1 and 2
    - Window size must be a multiple of 4
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

    Architecture (fixed 3-layer funnel with 2 pooling stages):
        Conv1d(input_dim → 64, k=5) + ReLU + MaxPool1d(2)  → (B, 64, W/2)
        Conv1d(64 → 32, k=3) + ReLU + MaxPool1d(2)        → (B, 32, W/4)
        Conv1d(32 → 16, k=3) + ReLU                       → (B, 16, W/4)
        Flatten                                            → (B, 16 * W/4)
        Linear(16 * W/4 → latent_dim)

    Parameters
    ----------
    input_dim : int, default=82
        Number of sensor channels (features).
    window_size : int, default=16
        Number of timesteps per window (W). Must be a multiple of 4.
    conv_channels : tuple[int, int, int], default=(64, 32, 16)
        Output channels for the three Conv1d layers (funnel: decreasing).
    conv_kernels : tuple[int, int, int], default=(5, 3, 3)
        Kernel sizes for the three Conv1d layers.
    pool_size : int, default=2
        Kernel size and stride for the two MaxPool1d layers (after conv 1 & 2).
    latent_dim : int, default=16
        Dimension of the compressed latent representation z.
    activation : str, default="relu"
        Activation function ("relu" or "leaky_relu").
    """

    def __init__(
        self,
        input_dim: int = 82,
        window_size: int = 16,
        conv_channels: tuple[int, int, int] = (64, 32, 16),
        conv_kernels: tuple[int, int, int] = (5, 3, 3),
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

        # Validate window_size compatibility with 2 pooling stages (factor 4)
        if window_size % 4 != 0:
            raise ValueError(
                f"window_size must be a multiple of 4 (got {window_size}), "
                f"because the encoder applies two MaxPool1d({pool_size}) stages "
                f"reducing temporal dimension by factor {pool_size**2}."
            )

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ReLU

        c1, c2, c3 = conv_channels
        k1, k2, k3 = conv_kernels

        # Temporal length after two pooling stages
        self.reduced_temporal_len = window_size // (pool_size * pool_size)
        # Flattened feature size before the final Linear layer
        self.flattened_size = c3 * self.reduced_temporal_len

        self.net = nn.Sequential(
            # Layer 1: preserves temporal length L=W
            nn.Conv1d(input_dim, c1, kernel_size=k1, padding=k1 // 2),
            act_cls(),
            # Pool 1: reduces temporal length to W // pool_size
            nn.MaxPool1d(kernel_size=pool_size),
            # Layer 2: preserves reduced length
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            act_cls(),
            # Pool 2: reduces temporal length to W // (pool_size^2)
            nn.MaxPool1d(kernel_size=pool_size),
            # Layer 3: preserves final reduced length (no pooling after)
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2),
            act_cls(),
            # Flatten entire sequence (no AdaptiveAvgPool1d)
            nn.Flatten(),
            # Dense projection to latent bottleneck
            nn.Linear(self.flattened_size, latent_dim),
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
    """Auto-derived Symmetric 1D-Transposed Convolutional Decoder.

    Reconstructs the original sequence window ``(B, input_dim, W)`` from a
    latent vector ``z`` of shape ``(B, latent_dim)``.

    The decoder architecture is automatically derived from the encoder's
    ``conv_channels`` by reversing the channel progression (excluding input_dim):
    - Encoder channels: (c1, c2, c3) = (64, 32, 16)  [funnel down]
    - Decoder channels: (c3, c2, c1) = (16, 32, 64)  [funnel up]
    - Two ConvTranspose1d layers for the two pooling stages
    - Final Conv1d projects c1 → input_dim

    Architecture (for encoder conv_channels=(64, 32, 16), 2 pooling stages):
        Linear(latent_dim → 16 * (W/4)) + ReLU
        Reshape → (B, 16, W/4)
        ConvTranspose1d(16 → 32, k=4, s=2, p=1) + ReLU → (B, 32, W/2)
        ConvTranspose1d(32 → 64, k=4, s=2, p=1) + ReLU → (B, 64, W)
        Conv1d(64 → input_dim, k=3, p=1) → (B, input_dim, W)

    Parameters
    ----------
    latent_dim : int, default=16
        Dimension of the input latent representation z.
    input_dim : int, default=82
        Number of output sensor channels.
    window_size : int, default=16
        Target temporal length (W) to reconstruct. Must be multiple of 4.
    encoder_channels : tuple[int, int, int], default=(64, 32, 16)
        Encoder's conv_channels (used to derive decoder structure).
    pool_size : int, default=2
        Pooling factor used in encoder (determines upsampling stages).
    activation : str, default="relu"
        Activation function ("relu" or "leaky_relu").
    """

    def __init__(
        self,
        latent_dim: int = 16,
        input_dim: int = 82,
        window_size: int = 16,
        encoder_channels: tuple[int, int, int] = (64, 32, 16),
        pool_size: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.window_size = window_size
        self.encoder_channels = encoder_channels
        self.pool_size = pool_size

        # Validate window_size compatibility
        if window_size % (pool_size * pool_size) != 0:
            raise ValueError(
                f"window_size must be a multiple of {pool_size * pool_size} "
                f"(got {window_size}) for {2} upsampling stages."
            )

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ReLU

        # Encoder channels: (c1, c2, c3) = e.g., (64, 32, 16)
        # Decoder works in reverse: starts from c3, goes to c2, then c1
        c1, c2, c3 = encoder_channels

        # Temporal length at the bottleneck (after encoder's 2 pooling stages)
        self.bottleneck_temporal_len = window_size // (pool_size * pool_size)
        # Flattened size at bottleneck = c3 * bottleneck_temporal_len
        self.bottleneck_flat_size = c3 * self.bottleneck_temporal_len

        # Project latent vector to flat bottleneck feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.bottleneck_flat_size),
            act_cls(),
        )

        # Two upsampling stages (matching encoder's two pooling stages)
        # Stage 1: (B, c3, W/4) → (B, c2, W/2)
        # Stage 2: (B, c2, W/2) → (B, c1, W)
        self.deconv = nn.Sequential(
            # Upsample 1: c3 → c2
            nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=pool_size, padding=1),
            act_cls(),
            # Upsample 2: c2 → c1
            nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=pool_size, padding=1),
            act_cls(),
            # Final projection to original sensor count
            nn.Conv1d(c1, input_dim, kernel_size=3, padding=1),
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
        # Reshape to (B, c3, bottleneck_temporal_len)
        h = h.view(batch_size, self.encoder_channels[2], self.bottleneck_temporal_len)
        x_rec = self.deconv(h)

        # Safety adjustment if window_size doesn't match exactly
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

    The decoder architecture is automatically derived from the encoder's
    conv_channels (reversed), ensuring perfect symmetry.

    Parameters
    ----------
    input_dim : int, default=82
        Number of input channels (sensor features).
    window_size : int, default=16
        Number of consecutive timesteps per window (W). Must be multiple of 4.
    latent_dim : int, default=16
        Dimension of latent bottleneck vector z.
    encoder_channels : tuple[int, int, int], default=(64, 32, 16)
        Conv channels in encoder (3 layers, funnel: decreasing).
    encoder_kernels : tuple[int, int, int], default=(5, 3, 3)
        Conv kernel sizes in encoder (3 layers).
    pool_size : int, default=2
        Pooling kernel size (applied after conv layers 1 and 2).
    activation : str, default="relu"
        Activation function.
    """

    def __init__(
        self,
        input_dim: int = 82,
        window_size: int = 16,
        latent_dim: int = 16,
        encoder_channels: tuple[int, int, int] = (64, 32, 16),
        encoder_kernels: tuple[int, int, int] = (5, 3, 3),
        pool_size: int = 2,
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
            encoder_channels=encoder_channels,
            pool_size=pool_size,
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
        metric: Literal["mse", "mae"] = "mae",
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
        metric : {"mse", "mae"}, default="mae"
            Distance metric used (MAE: absolute difference, MSE: squared difference).
            Default is "mae" for robustness to sensor saturation (see §7 #9).

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

        Expected config structure:
            model:
              input_dim: 82
              window_size: 32          # must be multiple of 4
              latent_dim: 16
              encoder:
                conv_channels: [64, 32, 16]  # 3 layers, funnel
                conv_kernels: [5, 3, 3]
                pool_size: 2
                activation: "relu"
              # decoder section is derived from encoder, not configured separately
        """
        if config is None:
            config = load_config()

        input_dim = int(get_param(config, "model.input_dim", 82))
        window_size = int(get_param(config, "model.window_size", 16))
        latent_dim = int(get_param(config, "model.latent_dim", 16))

        enc_channels = tuple(
            get_param(config, "model.encoder.conv_channels", [64, 32, 16])
        )
        enc_kernels = tuple(
            get_param(config, "model.encoder.conv_kernels", [5, 3, 3])
        )
        pool_size = int(get_param(config, "model.encoder.pool_size", 2))
        activation = str(get_param(config, "model.encoder.activation", "relu"))

        return cls(
            input_dim=input_dim,
            window_size=window_size,
            latent_dim=latent_dim,
            encoder_channels=enc_channels,  # type: ignore[arg-type]
            encoder_kernels=enc_kernels,    # type: ignore[arg-type]
            pool_size=pool_size,
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
        self.best_epoch: int = 0
        self.early_stop = False

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
    loss: str | None = None,
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
        Loss function instance. If None, ``loss`` is used to build it.
    loss : str | None
        Loss name ("mae" or "mse"), used to build ``criterion`` when ``criterion``
        is None.  Defaults to "mae" (robust to sensor saturation, see §7 #9).

    Returns
    -------
    dict
        History with keys "train_loss" (list[float]), "val_loss" (list[float]),
        and "best_epoch" (int — the epoch with the best validation loss).
    """
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
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "best_epoch": 0}

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

        if early_stopping(val_loss, model, epoch):
            logger.info(
                "Early stopping triggered at epoch %d (best val loss: %.6f)",
                epoch, -early_stopping.best_score if early_stopping.best_score else 0.0,
            )
            break

    # Record the actual best epoch (tracked by EarlyStopping)
    history["best_epoch"] = early_stopping.best_epoch

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