"""Unit tests for the sequence-aware 1D-CNN autoencoder (Fase 3) and Adversarial Autoencoder (Fase 4).

Verifies encoder, decoder, autoencoder orchestrator, discriminator, shape handling,
reconstruction error calculation, gradient backprop, and training loops.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import (
    Conv1dDecoder,
    Conv1dEncoder,
    EarlyStopping,
    SequenceAutoencoder,
    fit_autoencoder,
)
from src.models.adversarial_ae import (
    AdversarialAutoencoder,
    Discriminator,
    fit_aae,
)
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_config() -> dict:
    return load_config()


@pytest.fixture
def batch_conv_format() -> torch.Tensor:
    """Batch in standard Conv1d format: (B=4, C=86, W=16)."""
    torch.manual_seed(42)
    return torch.randn(4, 86, 16)


@pytest.fixture
def batch_seq_format() -> torch.Tensor:
    """Batch in standard sequence/DataLoader format: (B=4, W=16, C=86)."""
    torch.manual_seed(42)
    return torch.randn(4, 16, 86)


# ---------------------------------------------------------------------------
# Test Conv1dEncoder
# ---------------------------------------------------------------------------
class TestConv1dEncoder:
    def test_default_output_shape(self, batch_conv_format):
        encoder = Conv1dEncoder(input_dim=86, window_size=16, latent_dim=16)
        z = encoder(batch_conv_format)
        assert z.shape == (4, 16, 4), f"Expected shape (4, 16, 4), got {z.shape}"

    def test_custom_latent_dim(self, batch_conv_format):
        encoder = Conv1dEncoder(input_dim=86, window_size=16, latent_dim=32)
        z = encoder(batch_conv_format)
        assert z.shape == (4, 32, 4)

    def test_single_sample_batch(self):
        encoder = Conv1dEncoder(input_dim=86, window_size=16, latent_dim=16)
        x = torch.randn(1, 86, 16)
        z = encoder(x)
        assert z.shape == (1, 16, 4)


# ---------------------------------------------------------------------------
# Test Conv1dDecoder
# ---------------------------------------------------------------------------
class TestConv1dDecoder:
    def test_default_output_shape(self):
        decoder = Conv1dDecoder(latent_dim=16, input_dim=86, window_size=16)
        z = torch.randn(4, 16, 4)
        x_rec = decoder(z)
        assert x_rec.shape == (4, 86, 16), f"Expected (4, 86, 16), got {x_rec.shape}"

    @pytest.mark.parametrize("w", [8, 16, 32, 64])
    def test_variable_window_sizes(self, w):
        decoder = Conv1dDecoder(latent_dim=16, input_dim=86, window_size=w)
        z = torch.randn(2, 16, w // 4)
        x_rec = decoder(z)
        assert x_rec.shape == (2, 86, w), f"Window size {w} produced shape {x_rec.shape}"


# ---------------------------------------------------------------------------
# Test SequenceAutoencoder
# ---------------------------------------------------------------------------
class TestSequenceAutoencoder:
    def test_forward_conv_format(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        x_hat = model(batch_conv_format)
        assert x_hat.shape == batch_conv_format.shape

    def test_forward_seq_format(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        x_hat = model(batch_seq_format)
        assert x_hat.shape == batch_seq_format.shape

    def test_encode_decode_methods(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        z = model.encode(batch_conv_format)
        assert z.shape == (4, 16, 4)
        x_rec = model.decode(z)
        assert x_rec.shape == (4, 86, 16)

    def test_gradients_backprop(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        torch.optim.Adam(model.parameters(), lr=1e-3)
        x_hat = model(batch_conv_format)
        loss = nn.functional.mse_loss(x_hat, batch_conv_format)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive gradients"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_invalid_input_ndim(self):
        model = SequenceAutoencoder(input_dim=86, window_size=16)
        with pytest.raises(ValueError, match="Expected 3-D tensor"):
            model(torch.randn(86, 16))

    def test_from_config(self, default_config):
        model = SequenceAutoencoder.from_config(default_config)
        assert model.input_dim in (82, 86)
        assert model.window_size in (128, 256)
        assert model.latent_dim == 32


# ---------------------------------------------------------------------------
# Test Reconstruction Error
# ---------------------------------------------------------------------------
class TestReconstructionError:
    def test_sample_reduction(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16)
        errors = model.compute_reconstruction_error(batch_seq_format, reduction="sample")
        assert errors.shape == (4,), f"Expected shape (4,), got {errors.shape}"
        assert (errors >= 0).all(), "Reconstruction error must be non-negative"

    def test_mean_reduction(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=86, window_size=16)
        mean_err = model.compute_reconstruction_error(batch_seq_format, reduction="mean")
        assert mean_err.ndim == 0, "Expected scalar"
        assert mean_err.item() >= 0


# ---------------------------------------------------------------------------
# Test Training Utilities
# ---------------------------------------------------------------------------
class TestTrainingUtilities:
    def test_early_stopping_trigger(self):
        model = nn.Linear(10, 10)
        es = EarlyStopping(patience=3, min_delta=1e-3)

        assert not es(1.0, model)
        assert not es(1.0, model)
        assert not es(1.0, model)
        assert es(1.0, model)
        assert es.early_stop is True

    def test_fit_autoencoder_smoke(self):
        model = SequenceAutoencoder(input_dim=86, window_size=16, latent_dim=8)

        x_dummy = torch.randn(32, 16, 86)
        y_dummy = torch.zeros(32, 1)
        ds = TensorDataset(x_dummy, y_dummy)
        loader = DataLoader(ds, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_ckpt.pth"
            history = fit_autoencoder(
                model=model,
                train_loader=loader,
                val_loader=loader,
                epochs=2,
                patience=2,
                checkpoint_path=ckpt_path,
                device="cpu",
            )

            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2
            assert ckpt_path.exists()


# ---------------------------------------------------------------------------
# Test Discriminator (Fase 4.0)
# ---------------------------------------------------------------------------
class TestDiscriminator:
    @pytest.mark.parametrize("hidden_layers", [(16, 8), (32, 16), (128, 64, 32)])
    def test_output_shape(self, hidden_layers):
        disc = Discriminator(latent_dim=16, hidden_layers=hidden_layers, use_sigmoid=True)
        z = torch.randn(4, 16)
        out = disc(z)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_single_sample(self):
        disc = Discriminator(latent_dim=16)
        z = torch.randn(1, 16)
        out = disc(z)
        assert out.shape == (1, 1)

    def test_3d_temporal_input(self):
        disc = Discriminator(latent_dim=16)
        z = torch.randn(4, 16, 4)  # (B, latent_dim, T)
        out = disc(z)
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Test AdversarialAutoencoder (Fase 4.0)
# ---------------------------------------------------------------------------
class TestAdversarialAutoencoder:
    def test_forward_conv_format(self, batch_conv_format):
        model = AdversarialAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        x_hat = model(batch_conv_format)
        assert x_hat.shape == batch_conv_format.shape

    def test_forward_seq_format(self, batch_seq_format):
        model = AdversarialAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        x_hat = model(batch_seq_format)
        assert x_hat.shape == batch_seq_format.shape

    def test_encode_decode(self, batch_conv_format):
        model = AdversarialAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        z = model.encode(batch_conv_format)
        assert z.shape == (4, 16, 4)
        x_rec = model.decode(z)
        assert x_rec.shape == (4, 86, 16)

    def test_discriminate(self, batch_conv_format):
        model = AdversarialAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        z = model.encode(batch_conv_format)
        probs = model.discriminate(z)
        assert probs.shape == (4, 1)

    def test_reconstruction_error(self, batch_seq_format):
        model = AdversarialAutoencoder(input_dim=86, window_size=16, latent_dim=16)
        errors = model.compute_reconstruction_error(batch_seq_format, reduction="sample")
        assert errors.shape == (4,)
        assert (errors >= 0).all()

    def test_from_config(self, default_config):
        model = AdversarialAutoencoder.from_config(default_config)
        assert model.encoder is not None
        assert model.decoder is not None
        assert model.discriminator is not None
        assert model.latent_dim == 32


class TestAdversarialTraining:
    def test_fit_aae_smoke(self):
        model = AdversarialAutoencoder(
            input_dim=86, window_size=16, latent_dim=8,
            discriminator_hidden=(16, 8),
        )

        x_dummy = torch.randn(32, 16, 86)
        y_dummy = torch.zeros(32, 1)
        ds = TensorDataset(x_dummy, y_dummy)
        loader = DataLoader(ds, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_aae_ckpt.pth"
            history = fit_aae(
                model=model,
                train_loader=loader,
                val_loader=loader,
                epochs=2,
                learning_rate=1e-3,
                reconstruction_weight=1.0,
                adversarial_weight=0.1,
                patience=2,
                checkpoint_path=str(ckpt_path),
                device="cpu",
            )

            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2
            assert "adv_loss" in history
            assert len(history["adv_loss"]) == 2
            assert ckpt_path.exists()
