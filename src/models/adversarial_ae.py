"""Adversarial Autoencoder (AAE) for anomaly detection (project_plan.md §4.0).

Extends the vanilla ``SequenceAutoencoder`` (Fase 3) by adding a discriminator
that regularises the latent space to match a standard Gaussian prior ``N(0, I)``.

Architecture overview:
----------------------
Encoder  (Conv1d):   (B, input_dim, W) → z ∈ ℝ^(B, latent_dim, W/4)  [shared with AE]
Decoder  (Conv1d):   z                 → x̂ ∈ ℝ^(B, input_dim, W)    [shared with AE]
Discriminator (MLP): z                 → D(z) ∈ [0, 1]              [aggregates over time]

Training (alternating, per Makhzani et al. 2015):
    Phase D:  maximize  E[log D(z_prior)] + E[log(1 − D(E(x)))]
    Phase G:  minimize  λ_rec · Loss(x, x̂) + λ_adv · E[log(1 − D(E(x)))]
              ≡         λ_rec · Loss(x, x̂) − λ_adv · log D(E(x))   (non-saturating)

Input Layout Support:
    Accepts both ``(B, input_dim, W)`` and ``(B, W, input_dim)`` layouts,
    matching ``SequenceAutoencoder`` behaviour.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.autoencoder import Conv1dDecoder, Conv1dEncoder, SequenceAutoencoder
from src.utils.config import get_param, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """MLP discriminator for the AAE latent space.

    Maps latent representations ``z`` (either 2D ``(B, latent_dim)`` or
    temporal 3D ``(B, latent_dim, T)``) to a scalar probability in ``(B, 1)``
    indicating whether ``z`` came from the prior ``N(0, I)`` (real) or the
    encoder ``E(x)`` (fake).

    Parameters
    ----------
    latent_dim : int, default=16
        Dimension of the latent space (input size).
    hidden_layers : tuple[int, ...], default=(32, 16)
        Width of hidden layers, e.g. ``(32, 16)``.
    activation : str, default="leaky_relu"
        Hidden activation: ``"relu"`` or ``"leaky_relu"``.
    use_sigmoid : bool, default=True
        If True, squashes output to (0, 1) via Sigmoid.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_layers: tuple[int, ...] = (32, 16),
        activation: str = "leaky_relu",
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_layers = tuple(hidden_layers)
        self.use_sigmoid = use_sigmoid

        act_cls = nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.ReLU

        # --- Build sequential MLP ---
        layers: list[nn.Module] = []
        prev_width = latent_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_width, width))
            layers.append(act_cls())
            prev_width = width
        # Output layer → scalar logit
        layers.append(nn.Linear(prev_width, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Classify latent representations as prior (real) or encoder (fake).

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor: shape ``(B, latent_dim)`` or ``(B, latent_dim, T)``
            or ``(B, T, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(B, 1)``.
        """
        if z.ndim == 3:
            # If shape is (B, latent_dim, T), transpose to (B, T, latent_dim)
            if z.shape[1] == self.latent_dim:
                z_in = z.transpose(1, 2)
            else:
                z_in = z
            out = self.net(z_in)  # (B, T, 1)
            return out.mean(dim=1)  # aggregate over time -> (B, 1)
        elif z.ndim == 2:
            return self.net(z)  # (B, 1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {z.shape}")


# ---------------------------------------------------------------------------
# 2. Adversarial Autoencoder Orchestrator
# ---------------------------------------------------------------------------
class AdversarialAutoencoder(nn.Module):
    """Sequence-Aware Adversarial Autoencoder.

    Combines ``Conv1dEncoder``, ``Conv1dDecoder``, and ``Discriminator``.
    """

    def __init__(
        self,
        input_dim: int = 86,
        window_size: int = 256,
        latent_dim: int = 16,
        encoder_channels: tuple[int, int, int] = (64, 32, 16),
        encoder_kernels: tuple[int, int, int] = (5, 3, 3),
        pool_size: int = 2,
        activation: str = "relu",
        discriminator_hidden: tuple[int, ...] = (32, 16),
        discriminator_activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder_channels = tuple(encoder_channels)
        self.encoder_kernels = tuple(encoder_kernels)
        self.pool_size = pool_size
        self.activation = activation
        self.discriminator_hidden = tuple(discriminator_hidden)
        self.discriminator_activation = discriminator_activation

        # --- Reused encoder + decoder from vanilla AE ---
        self.encoder = Conv1dEncoder(
            input_dim=input_dim,
            window_size=window_size,
            conv_channels=self.encoder_channels,
            conv_kernels=self.encoder_kernels,
            pool_size=pool_size,
            latent_dim=latent_dim,
            activation=activation,
        )
        self.decoder = Conv1dDecoder(
            latent_dim=latent_dim,
            input_dim=input_dim,
            window_size=window_size,
            encoder_channels=self.encoder_channels,
            pool_size=pool_size,
            activation=activation,
        )

        # --- Adversarial discriminator ---
        self.discriminator = Discriminator(
            latent_dim=latent_dim,
            hidden_layers=self.discriminator_hidden,
            activation=discriminator_activation,
            use_sigmoid=True,
        )

    def _format_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3-D tensor (B, W, D) or (B, D, W), got ndim={x.ndim} with shape {x.shape}"
            )
        _, d1, d2 = x.shape
        if d1 == self.input_dim and d2 == self.window_size:
            return x, False
        if d1 == self.window_size and d2 == self.input_dim:
            return x.transpose(1, 2), True
        if d1 == self.input_dim:
            return x, False
        if d2 == self.input_dim:
            return x.transpose(1, 2), True
        raise ValueError(
            f"Input shape {x.shape} is incompatible with expected "
            f"input_dim={self.input_dim} and window_size={self.window_size}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent representation z."""
        x_conv, _ = self._format_input(x)
        return self.encoder(x_conv)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation z to reconstructed sequence."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass (reconstruction). Preserves input layout."""
        x_conv, was_transposed = self._format_input(x)
        z = self.encoder(x_conv)
        x_hat = self.decoder(z)
        if was_transposed:
            return x_hat.transpose(1, 2)
        return x_hat

    def discriminate(self, z: torch.Tensor) -> torch.Tensor:
        """Run discriminator on latent representations."""
        return self.discriminator(z)

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: Literal["none", "mean", "sample"] = "sample",
        metric: Literal["mse", "mae"] = "mae",
    ) -> torch.Tensor:
        """Compute per-sample reconstruction error (same API as vanilla AE)."""
        with torch.no_grad():
            x_hat = self.forward(x)

        x_orig, _ = self._format_input(x)
        x_rec, _ = self._format_input(x_hat)

        if metric == "mse":
            diff = (x_orig - x_rec) ** 2
        elif metric == "mae":
            diff = torch.abs(x_orig - x_rec)
        else:
            raise ValueError(f"Unsupported metric: '{metric}', must be 'mse' or 'mae'")

        if reduction == "none":
            return diff
        if reduction == "mean":
            return diff.mean()
        if reduction == "sample":
            dims = tuple(range(1, diff.ndim))
            return diff.mean(dim=dims)
        raise ValueError(f"Unsupported reduction: '{reduction}'")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> AdversarialAutoencoder:
        """Instantiate an AdversarialAutoencoder from config dictionary."""
        if config is None:
            config = load_config()

        runtime_params = SequenceAutoencoder._discover_runtime_params(config)
        input_dim = int(runtime_params.get("input_dim", get_param(config, "model.input_dim", 86)))
        window_size = int(get_param(config, "data.window_size", 256))
        latent_dim = int(get_param(config, "model.latent_dim", 16))

        enc_channels = tuple(get_param(config, "model.encoder.conv_channels", [64, 32, 16]))
        enc_kernels = tuple(get_param(config, "model.encoder.conv_kernels", [5, 3, 3]))
        pool_size = int(get_param(config, "model.encoder.pool_size", 2))
        activation = str(get_param(config, "model.encoder.activation", "relu"))

        disc_hidden = tuple(
            get_param(config, "model.discriminator.hidden_layers", [32, 16])
        )
        disc_act = str(get_param(config, "model.discriminator.activation", "leaky_relu"))

        return cls(
            input_dim=input_dim,
            window_size=window_size,
            latent_dim=latent_dim,
            encoder_channels=enc_channels,
            encoder_kernels=enc_kernels,
            pool_size=pool_size,
            activation=activation,
            discriminator_hidden=disc_hidden,
            discriminator_activation=disc_act,
        )


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------
def train_epoch_aae(
    model: AdversarialAutoencoder,
    dataloader: DataLoader,
    optimizer_ae: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    reconstruction_weight: float = 1.0,
    adversarial_weight: float = 0.1,
    discriminator_updates_per_gen: int = 1,
    metric: Literal["mse", "mae"] = "mae",
) -> tuple[float, float, float]:
    """Train AAE for one epoch with alternating Phase D and Phase G updates."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_adv_loss = 0.0
    total_samples = 0

    bce = nn.BCELoss()
    rec_criterion = nn.L1Loss() if metric == "mae" else nn.MSELoss()

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        bs = x.size(0)

        x_conv, _ = model._format_input(x)

        # ------------------------------------------------------------------
        # Phase 1: Discriminator updates
        # ------------------------------------------------------------------
        for _ in range(discriminator_updates_per_gen):
            optimizer_disc.zero_grad()

            with torch.no_grad():
                z_encoded = model.encode(x)

            # Sample prior with matching shape
            z_prior = torch.randn_like(z_encoded)

            d_real = model.discriminate(z_prior)
            d_fake = model.discriminate(z_encoded)

            loss_d_real = bce(d_real, torch.ones_like(d_real))
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            loss_disc = (loss_d_real + loss_d_fake) / 2.0

            loss_disc.backward()
            optimizer_disc.step()

        # ------------------------------------------------------------------
        # Phase 2: Generator (Encoder + Decoder) update
        # ------------------------------------------------------------------
        optimizer_ae.zero_grad()

        z = model.encode(x)
        x_hat = model.decode(z)
        d_fake_new = model.discriminate(z)

        loss_recon = rec_criterion(x_hat, x_conv)
        loss_adv = bce(d_fake_new, torch.ones_like(d_fake_new))

        loss_ae = reconstruction_weight * loss_recon + adversarial_weight * loss_adv
        loss_ae.backward()
        optimizer_ae.step()

        total_loss += loss_ae.item() * bs
        total_recon_loss += loss_recon.item() * bs
        total_adv_loss += loss_adv.item() * bs
        total_samples += bs

    avg_total = total_loss / max(1, total_samples)
    avg_recon = total_recon_loss / max(1, total_samples)
    avg_adv = total_adv_loss / max(1, total_samples)
    return avg_total, avg_recon, avg_adv


def validate_aae(
    model: AdversarialAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: Literal["mse", "mae"] = "mae",
) -> float:
    """Compute validation reconstruction loss (normal windows only)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.L1Loss() if metric == "mae" else nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            bs = x.size(0)
            x_conv, _ = model._format_input(x)
            x_hat = model.decode(model.encode(x))
            loss = criterion(x_hat, x_conv)
            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / max(1, total_samples)


def fit_aae(
    model: AdversarialAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    reconstruction_weight: float = 1.0,
    adversarial_weight: float = 0.1,
    discriminator_updates_per_gen: int = 1,
    patience: int = 10,
    min_delta: float = 1e-4,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    metric: Literal["mse", "mae"] = "mae",
) -> dict[str, list[float]]:
    """Complete AAE training loop with early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.to(dev)

    from src.models.autoencoder import EarlyStopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer_ae = torch.optim.Adam(ae_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "adv_loss": [],
    }

    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_total, tr_recon, tr_adv = train_epoch_aae(
            model=model,
            dataloader=train_loader,
            optimizer_ae=optimizer_ae,
            optimizer_disc=optimizer_disc,
            device=dev,
            reconstruction_weight=reconstruction_weight,
            adversarial_weight=adversarial_weight,
            discriminator_updates_per_gen=discriminator_updates_per_gen,
            metric=metric,
        )
        val_loss = validate_aae(model, val_loader, dev, metric=metric)

        history["train_loss"].append(tr_total)
        history["recon_loss"].append(tr_recon)
        history["adv_loss"].append(tr_adv)
        history["val_loss"].append(val_loss)

        if early_stopping(val_loss, model):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Restore best weights
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    if ckpt_path:
        best_val_loss = -early_stopping.best_score if early_stopping.best_score is not None else float("inf")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "best_epoch": early_stopping.best_epoch,
                "best_val_loss": best_val_loss,
            },
            ckpt_path,
        )
        logger.info("Saved AAE checkpoint to %s", ckpt_path)

    return history
