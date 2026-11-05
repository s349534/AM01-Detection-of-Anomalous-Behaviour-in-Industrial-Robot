"""Sequence-Aware 1D-CNN Autoencoder for Anomaly Detection.

This module implements the baseline autoencoder architecture designed for the
Kuka industrial robot time-series dataset.

Architecture overview (see docs/project_plan.md §4.1.2):
---------------------------------------------------------
Encoder (1D-Conv, 3 layers, 2 pooling stages):
    Input:  (B, input_dim=86, W)  with W multiple of 4
       ↓    Conv1d(input_dim → 64, kernel=5, padding=2) + ReLU
            MaxPool1d(kernel_size=2) → (B, 64, W/2)
       ↓    Conv1d(64 → 32, kernel=3, padding=1) + ReLU
            MaxPool1d(kernel_size=2) → (B, 32, W/4)
       ↓    Conv1d(32 → 16, kernel=3, padding=1) + ReLU
            → (B, 16, W/4)
       ↓    Conv1d(16 → latent_dim, kernel_size=1)
    Output: z ∈ ℝ^(B, latent_dim, W/4) — temporal structure preserved

Decoder (Auto-derived symmetric 1D-ConvTranspose):
    Input:  z ∈ ℝ^(B, latent_dim, W/4)
       ↓    Conv1d(latent_dim → 16, kernel_size=1) (channel projection)
       ↓    ConvTranspose1d(16 → 32, kernel=4, stride=2, padding=1) + ReLU → (B, 32, W/2)
       ↓    ConvTranspose1d(32 → 64, kernel=4, stride=2, padding=1) + ReLU → (B, 64, W)
       ↓    Conv1d(64 → input_dim=86, kernel=3, padding=1)
    Output: x̂ ∈ ℝ^(B, input_dim=86, W)

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

Anomaly Scoring:
    Classic mean absolute error (MAE) across all sensor channels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.config import get_param, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Encoder
# ---------------------------------------------------------------------------
class Conv1dEncoder(nn.Module):
    """1D-Convolutional Encoder for multivariate time-series windows.

    Compresses an input tensor of shape ``(B, input_dim, W)`` into a latent
    tensor ``z`` of shape ``(B, latent_dim, W/4)`` preserving the temporal
    dimension at the bottleneck.

    Architecture (fixed 3-layer funnel with 2 pooling stages):
        Conv1d(input_dim → 64, k=5) + ReLU + MaxPool1d(2)  → (B, 64, W/2)
        Conv1d(64 → 32, k=3) + ReLU + MaxPool1d(2)        → (B, 32, W/4)
        Conv1d(32 → 16, k=3) + ReLU                       → (B, 16, W/4)
        Conv1d(16 → latent_dim, kernel_size=1)            → (B, latent_dim, W/4)

    Parameters
    ----------
    input_dim : int, default=86
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
        Dimension of the latent bottleneck representation per timestep.
    activation : str, default="relu"
        Activation function ("relu" or "leaky_relu").
    """

    def __init__(
        self,
        input_dim: int = 86,
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
            # Channel projection to latent dimension (preserves temporal length)
            nn.Conv1d(c3, latent_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence tensor to latent tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim, W)``.

        Returns
        -------
        torch.Tensor
            Latent representation z of shape ``(B, latent_dim, W/4)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Decoder
# ---------------------------------------------------------------------------
class Conv1dDecoder(nn.Module):
    """Auto-derived Symmetric 1D-Transposed Convolutional Decoder.

    Reconstructs the original sequence window ``(B, input_dim, W)`` from a
    latent tensor ``z`` of shape ``(B, latent_dim, W/4)`` preserving the
    temporal dimension at the bottleneck.

    The decoder architecture is automatically derived from the encoder's
    ``conv_channels`` by reversing the channel progression (excluding input_dim):
    - Encoder channels: (c1, c2, c3) = (64, 32, 16)  [funnel down]
    - Decoder channels: (c3, c2, c1) = (16, 32, 64)  [funnel up]
    - Two ConvTranspose1d layers for the two pooling stages
    - Final Conv1d projects c1 → input_dim

    Architecture (for encoder conv_channels=(64, 32, 16), 2 pooling stages):
        Conv1d(latent_dim → 16, kernel_size=1)              → (B, 16, W/4)
        ConvTranspose1d(16 → 32, k=4, s=2, p=1) + ReLU      → (B, 32, W/2)
        ConvTranspose1d(32 → 64, k=4, s=2, p=1) + ReLU      → (B, 64, W)
        Conv1d(64 → input_dim, k=3, p=1)                    → (B, input_dim, W)

    Parameters
    ----------
    latent_dim : int, default=16
        Dimension of the input latent representation z per timestep.
    input_dim : int, default=86
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
        input_dim: int = 86,
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

        # Channel projection from latent_dim to c3 (preserves temporal length)
        self.channel_proj = (
            nn.Conv1d(latent_dim, c3, kernel_size=1)
            if latent_dim != c3
            else nn.Identity()
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
        """Decode latent tensor back to sequence tensor.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(B, latent_dim, W/4)``.

        Returns
        -------
        torch.Tensor
            Reconstructed sequence tensor of shape ``(B, input_dim, W)``.
        """
        h = self.channel_proj(z)
        x_rec = self.deconv(h)
        x_rec = torch.tanh(x_rec)

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
    input_dim : int, default=86
        Number of input channels (sensor features).
    window_size : int, default=16
        Number of consecutive timesteps per window (W). Must be multiple of 4.
    latent_dim : int, default=16
        Dimension of the latent bottleneck per timestep (z shape: (B, latent_dim, W/4)).
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
        input_dim: int = 86,
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
        self.encoder_channels = encoder_channels
        self.encoder_kernels = encoder_kernels
        self.pool_size = pool_size
        self.activation = activation

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
        """Encode sequence to latent tensor z of shape ``(B, latent_dim, W/4)``.

        Accepts both ``(B, input_dim, W)`` and ``(B, W, input_dim)``.
        """
        x_conv, _ = self._format_input(x)
        return self.encoder(x_conv)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor z of shape ``(B, latent_dim, W/4)`` to reconstructed sequence ``(B, input_dim, W)``."""
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
            diff = (x - x_hat).pow(2)
        elif metric == "mae":
            diff = (x - x_hat).abs()
        else:
            raise ValueError(f"Unsupported metric: '{metric}', must be 'mse' or 'mae'")

        if reduction == "none":
            return diff
        if reduction == "mean":
            return diff.mean()
        if reduction == "sample":
            # Average across all feature and timestep dimensions for each sample
            return diff.mean(dim=(1, 2))

        raise ValueError(
            f"Unsupported reduction: '{reduction}', must be 'none', 'mean', or 'sample'"
        )

    def architecture_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with the full architecture parameters.
        
        Used for checkpoint serialization so the model can be rebuilt exactly
        from the checkpoint without needing the original config.
        """
        return {
            "input_dim": self.input_dim,
            "window_size": self.window_size,
            "latent_dim": self.latent_dim,
            "encoder_channels": list(self.encoder_channels),
            "encoder_kernels": list(self.encoder_kernels),
            "pool_size": self.pool_size,
            "activation": self.activation,
        }

    @classmethod
    def from_architecture(cls, arch: dict[str, Any]) -> SequenceAutoencoder:
        """Instantiate a SequenceAutoencoder directly from architecture dict.
        
        Used when loading from a checkpoint that contains the full architecture
        metadata, avoiding any config dependency.
        
        Parameters
        ----------
        arch : dict
            Dictionary with keys: input_dim, window_size, latent_dim,
            encoder_channels, encoder_kernels, pool_size, activation.
        
        Returns
        -------
        SequenceAutoencoder
            Model instance with the exact architecture from the checkpoint.
        """
        return cls(
            input_dim=arch["input_dim"],
            window_size=arch["window_size"],
            latent_dim=arch["latent_dim"],
            encoder_channels=tuple(arch["encoder_channels"]),
            encoder_kernels=tuple(arch["encoder_kernels"]),
            pool_size=arch["pool_size"],
            activation=arch["activation"],
        )

    @staticmethod
    def _discover_runtime_params(config: dict[str, Any]) -> dict[str, Any]:
        """Discover model parameters that depend on preprocessed data at runtime.

        Reads ``selected_columns.npy`` from the processed data directory to determine
        the actual ``input_dim`` (number of features after preprocessing). This
        overrides any hardcoded value in the config, ensuring the model always
        matches the data.

        Parameters
        ----------
        config : dict
            Full config dict (must contain ``paths.data_processed``).

        Returns
        -------
        dict
            Dictionary with discovered runtime parameters (e.g., ``input_dim``).
            Empty if processed data not found (falls back to config values).
        """
        from pathlib import Path
        import numpy as np

        processed_dir = Path(get_param(config, "paths.data_processed", "data/processed/"))
        selected_cols_path = processed_dir / "selected_columns.npy"

        discovered = {}
        if selected_cols_path.exists():
            try:
                selected_cols = np.load(selected_cols_path, allow_pickle=True)
                actual_input_dim = int(selected_cols.shape[0])
                config_input_dim = int(get_param(config, "model.input_dim", 86))

                if actual_input_dim != config_input_dim:
                    logger.warning(
                        "input_dim mismatch: config=%d, discovered from data=%d. "
                        "Using discovered value (data is source of truth).",
                        config_input_dim, actual_input_dim,
                    )
                discovered["input_dim"] = actual_input_dim

            except Exception as e:
                logger.warning("Could not read selected_columns.npy for runtime param discovery: %s", e)
        else:
            logger.debug(
                "Processed data not found at %s — using config values for input_dim.",
                processed_dir,
            )

        return discovered

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> SequenceAutoencoder:
        """Instantiate a SequenceAutoencoder from a configuration dictionary.

        If config is None, loads default config via ``src.utils.config.load_config()``.

        Runtime parameter discovery:
            Automatically detects ``input_dim`` from
            ``data/processed/selected_columns.npy`` if available. This ensures the
            model architecture always matches the actual preprocessed data, regardless
            of what the config file says.

        Expected config structure:
            data:
              window_size: 32        # must be multiple of 4
            model:
              input_dim: 86          # placeholder; overridden at runtime from data
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

        # Discover runtime parameters from preprocessed data (overrides config)
        runtime_params = cls._discover_runtime_params(config)

        input_dim = int(runtime_params.get("input_dim", get_param(config, "model.input_dim", 86)))
        # Read window_size from data.window_size (single source of truth)
        window_size = int(get_param(config, "data.window_size", 256))
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


# Re-export EarlyStopping and compute_errors_on_loader for backward compatibility
from src.models.training import EarlyStopping  # noqa: F401
from src.models.inference import compute_errors_on_loader  # noqa: F401
