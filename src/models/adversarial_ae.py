"""Adversarial Autoencoder (AAE) for anomaly detection (project_plan.md §4.0).

Extends the vanilla ``SequenceAutoencoder`` (Fase 3) by adding a discriminator
that regularises the latent space to match a standard Gaussian prior ``N(0, I)``.

Architecture overview (§4.1.3):
-----------------------------------------
Encoder  (Conv1d):   (B, 82, W)  → z ∈ ℝ^(B, latent_dim)     [shared with AE]
Decoder  (Conv1d):   z           → x̂ ∈ ℝ^(B, 82, W)           [shared with AE]
Discriminator (MLP): z           → D(z) ∈ [0, 1]

Training (alternating, per Makhzani et al. 2015):
    Phase D:  maximize  E[log D(z_prior)] + E[log(1 − D(E(x)))]
    Phase G:  minimize  λ_rec · MSE(x, x̂) + λ_adv · E[log(1 − D(E(x)))]
              ≡         λ_rec · MSE(x, x̂) − λ_adv · log D(E(x))   (non-saturating)

Input Layout Support (§4.1.2):
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

from src.models.autoencoder import Conv1dDecoder, Conv1dEncoder
from src.utils.config import get_param, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """MLP discriminator for the AAE latent space.

    Maps a latent vector ``z ∈ ℝ^(latent_dim,)`` to a scalar probability
    that ``z`` came from the prior ``N(0, I)`` (i.e. "real") rather than
    from the encoder ``E(x)`` (i.e. "fake").

    Parameters
    ----------
    latent_dim : int, default=16
        Dimension of the latent space (input size).
    hidden_layers : tuple[int, ...], default=(32, 16)
        Width of hidden layers, e.g. ``(32, 16)`` → Linear(16→32) → ReLU
        → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid.
    activation : str, default="leaky_relu"
        Hidden activation: ``"relu"`` or ``"leaky_relu"``.
    use_sigmoid : bool, default=True
        If **True**, the output is squashed to ``(0, 1)`` via Sigmoid.
        Set **False** for BCEWithLogitsLoss (numerically more stable).
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
        """Classify latent vectors as prior (real) or encoder (fake).

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(B, 1)`` (or logits if ``use_sigmoid=False``).
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# 2. Adversarial Autoencoder
# ---------------------------------------------------------------------------
class AdversarialAutoencoder(nn.Module):
    """Adversarial Autoencoder combining Encoder, Decoder, and Discriminator.

    The encoder and decoder are **reused directly** from the vanilla AE (§4.1.3):
    the adversarial component only adds a discriminator on the latent vector.

    Parameters
    ----------
    input_dim : int, default=82
        Number of sensor channels (features).
    window_size : int, default=16
        Number of consecutive timesteps per window (W).
    latent_dim : int, default=16
        Dimension of latent bottleneck vector z.
    encoder_channels : tuple[int, int], default=(128, 64)
        Conv channels in encoder (shared with AE).
    encoder_kernels : tuple[int, int], default=(5, 3)
        Conv kernel sizes in encoder.
    pool_size : int, default=2
        Pooling kernel size.
    decoder_channels : tuple[int, int], default=(64, 128)
        Conv channels in decoder (reverse of encoder).
    activation : str, default="relu"
        Activation in conv layers.
    discriminator_hidden : tuple[int, ...], default=(32, 16)
        Hidden layer widths of the discriminator MLP.
    discriminator_activation : str, default="leaky_relu"
        Activation in discriminator.
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
        discriminator_hidden: tuple[int, ...] = (32, 16),
        discriminator_activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim

        # --- Reused encoder + decoder from the vanilla AE ---
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

        # --- Adversarial discriminator (new component) ---
        self.discriminator = Discriminator(
            latent_dim=latent_dim,
            hidden_layers=discriminator_hidden,
            activation=discriminator_activation,
            use_sigmoid=True,  # BCELoss expects probabilities
        )

    # ------------------------------------------------------------------
    # Forward / encode / decode (mirror SequenceAutoencoder API)
    # ------------------------------------------------------------------
    def _format_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Verify and standardise input shape to ``(B, input_dim, W)``.

        Accepts both ``(B, input_dim, W)`` and ``(B, W, input_dim)``.
        Returns the formatted tensor and a flag indicating whether a
        transpose is needed on the way out.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3-D tensor (B, W, D) or (B, D, W), got ndim={x.ndim} "
                f"with shape {x.shape}"
            )
        _, d1, d2 = x.shape
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
        """Encode sequence to latent vector z (same API as ``SequenceAutoencoder``)."""
        x_conv, _ = self._format_input(x)
        return self.encoder(x_conv)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to reconstructed sequence ``(B, input_dim, W)``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end adversarial autoencoder forward pass.

        Reconstructs the input sequence (preserving caller layout).
        Does **not** apply the discriminator; use :meth:`discriminate` for that.
        """
        x_conv, was_transposed = self._format_input(x)
        z = self.encoder(x_conv)
        x_hat = self.decoder(z)
        if was_transposed:
            return x_hat.transpose(1, 2)
        return x_hat

    def discriminate(self, z: torch.Tensor) -> torch.Tensor:
        """Run the discriminator on latent vectors.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(B, 1)``.
        """
        return self.discriminator(z)

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: Literal["none", "mean", "sample"] = "sample",
        metric: Literal["mse", "mae"] = "mse",
    ) -> torch.Tensor:
        """Compute per-sample reconstruction error (identical to vanilla AE).

        The anomaly score for AAE is still reconstruction error; the adversarial
        component regularises the latent space **during training** but does not
        change the inference-time scoring.
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
            dims = tuple(range(1, diff.ndim))
            return diff.mean(dim=dims)
        raise ValueError(
            f"Unsupported reduction: '{reduction}', must be 'none', 'mean', or 'sample'"
        )

    # ------------------------------------------------------------------
    # Config-driven factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> AdversarialAutoencoder:
        """Instantiate an ``AdversarialAutoencoder`` from a config dictionary.

        Mirrors ``SequenceAutoencoder.from_config`` so that the same
        ``params.yaml`` layout is used for both models.
        """
        if config is None:
            config = load_config()

        input_dim = int(get_param(config, "model.input_dim", 82))
        window_size = int(get_param(config, "model.window_size", 16))
        latent_dim = int(get_param(config, "model.latent_dim", 16))

        enc_channels = tuple(get_param(config, "model.encoder.conv_channels", [128, 64]))
        enc_kernels = tuple(get_param(config, "model.encoder.conv_kernels", [5, 3]))
        pool_size = int(get_param(config, "model.encoder.pool_size", 2))
        dec_channels = tuple(get_param(config, "model.decoder.conv_channels", [64, 128]))
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
            decoder_channels=dec_channels,
            activation=activation,
            discriminator_hidden=disc_hidden,
            discriminator_activation=disc_act,
        )


# ---------------------------------------------------------------------------
# 3. Early Stopping (re-exported from autoencoder for API parity)
# ---------------------------------------------------------------------------
# Reuse the same EarlyStopping from autoencoder — single source of truth.
from src.models.autoencoder import EarlyStopping  # noqa: E402


# ---------------------------------------------------------------------------
# 4. AAE Training Utilities
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
    metric: str = "mse",
) -> tuple[float, float, float]:
    """Train the AAE for one epoch with alternating discriminator/generator updates.

    Parameters
    ----------
    model : AdversarialAutoencoder
        The AAE model (encoder + decoder + discriminator).
    dataloader : DataLoader
        Provides batches of ``(window, label)`` — only the window is used.
    optimizer_ae : torch.optim.Optimizer
        Optimizer for encoder + decoder parameters.
    optimizer_disc : torch.optim.Optimizer
        Optimizer for discriminator parameters (separate optimizer).
    device : torch.device
        Hardware device.
    reconstruction_weight : float
        Weight for the MSE/MAE reconstruction loss.
    adversarial_weight : float
        Weight for the adversarial (generator fooling) loss.
    discriminator_updates_per_gen : int
        Number of discriminator updates per one generator (encoder+decoder) update.
    metric : {"mse", "mae"}
        Reconstruction metric.

    Returns
    -------
    tuple[float, float, float]
        ``(avg_total_loss, avg_recon_loss, avg_adv_loss)`` over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_adv_loss = 0.0
    total_samples = 0

    bce = nn.BCELoss()
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    rec_criterion = mse_loss if metric == "mse" else mae_loss

    latent_dim = model.latent_dim

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        bs = x.size(0)

        # --- Format input to (B, input_dim, W) for Conv1d ---
        x_conv, _ = model._format_input(x)

        # ==================================================================
        # Phase 1 — Update Discriminator  D(x): maximize
        #           log D(z_prior) + log(1 − D(E(x)))
        # Repeated ``discriminator_updates_per_gen`` times per generator update.
        # ==================================================================
        for _ in range(discriminator_updates_per_gen):
            optimizer_disc.zero_grad()

            # Real samples from prior
            z_prior = torch.randn(bs, latent_dim, device=device)
            d_real = model.discriminate(z_prior)

            # Fake samples from encoder (detach so gradients don't flow to encoder)
            with torch.no_grad():
                z_encoded = model.encode(x)
            d_fake = model.discriminate(z_encoded)

            # BCE for discriminator: real → 1, fake → 0
            loss_d_real = bce(d_real, torch.ones_like(d_real))
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            loss_disc = (loss_d_real + loss_d_fake) / 2.0

            loss_disc.backward()
            optimizer_disc.step()

        # ==================================================================
        # Phase 2 — Update Generator (encoder + decoder)  G(z): minimize
        #           λ_rec · recon(x, x̂) + λ_adv · E[-log D(E(x))]   (non-saturating)
        # ==================================================================
        optimizer_ae.zero_grad()

        z = model.encode(x)
        x_hat = model.decode(z)
        d_fake_new = model.discriminate(z)  # recompute with updated z (no grad to D here)

        loss_recon = rec_criterion(x_hat, x_conv)
        # Generator wants D to think encoded z is real → target = 1
        loss_adv = bce(d_fake_new, torch.ones_like(d_fake_new))

        loss_ae = reconstruction_weight * loss_recon + adversarial_weight * loss_adv

        loss_ae.backward()
        optimizer_ae.step()

        # Accumulate stats
        total_loss += loss_ae.item() * bs
        total_recon_loss += loss_recon.item() * bs
        total_adv_loss += loss_adv.item() * bs
        total_samples += bs

    avg_total = total_loss / max(1, total_samples)
    avg_recon = total_recon_loss / max(1, total_samples)
    avg_adv = total_adv_loss / max(1, total_samples)
    return avg_total, avg_recon, avg_adv


def evaluate_epoch_aae(
    model: AdversarialAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "mse",
) -> float:
    """Evaluate reconstruction loss on validation data (no discriminator needed)."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    rec_criterion = nn.MSELoss() if metric == "mse" else nn.L1Loss()

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x_conv, _ = model._format_input(x)
            x_hat = model(x)
            x_hat_conv, _ = model._format_input(x_hat)
            loss = rec_criterion(x_hat_conv, x_conv)
            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return running_loss / max(1, total_samples)


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
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    metric: str = "mse",
) -> dict[str, list[float]]:
    """Complete training loop for the AdversarialAutoencoder.

    Uses **separate optimizers**: one for encoder+decoder (generator), one for
    the discriminator.  Uses standard BCELoss for the adversarial objective.

    Parameters
    ----------
    model : AdversarialAutoencoder
    train_loader, val_loader : DataLoader
    epochs : int
        Maximum training epochs.
    learning_rate : float
    weight_decay : float
    reconstruction_weight : float
        Weight for MSE/MAE reconstruction loss.
    adversarial_weight : float
        Weight for adversarial (generator fooling) loss.
    discriminator_updates_per_gen : int
        Discriminator updates per encoder-decoder update.
    patience : int
        Early stopping patience on validation reconstruction loss.
    min_delta : float
        Minimum improvement for early stopping.
    checkpoint_path : str | Path | None
    device : str | torch.device
    metric : {"mse", "mae"}

    Returns
    -------
    dict[str, list[float]]
        History with ``train_loss``, ``val_loss``, ``recon_loss``, ``adv_loss``.
    """
    dev = torch.device(device)
    model.to(dev)

    # --- Separate optimizers ---
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    disc_params = list(model.discriminator.parameters())

    optimizer_ae = torch.optim.Adam(
        ae_params, lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999)
    )
    optimizer_disc = torch.optim.Adam(
        disc_params, lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999)
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "adv_loss": [],
    }

    logger.info("Starting AdversarialAutoencoder training on %s (%d epochs)", dev, epochs)

    for epoch in range(1, epochs + 1):
        train_loss, recon_loss, adv_loss = train_epoch_aae(
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
        val_loss = evaluate_epoch_aae(model, val_loader, dev, metric=metric)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(recon_loss)
        history["adv_loss"].append(adv_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%3d — Train Loss: %.6f — Val Loss: %.6f "
                "(recon=%.4f adv=%.4f)",
                epoch, epochs, train_loss, val_loss, recon_loss, adv_loss,
            )

        if early_stopping(val_loss, model):
            logger.info(
                "Early stopping triggered at epoch %d",
                epoch,
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
                "discriminator_hidden": list(model.discriminator.hidden_layers),
            },
            ckpt,
        )
        logger.info("Saved best AAE checkpoint to %s", ckpt)

    return history
