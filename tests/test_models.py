"""Unit tests for the sequence-aware 1D-CNN autoencoder (Fase 3).

Verifies encoder, decoder, autoencoder orchestrator, shape handling,
reconstruction error calculation, gradient backprop, and early stopping.
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
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_config() -> dict:
    return load_config()


@pytest.fixture
def batch_conv_format() -> torch.Tensor:
    """Batch in standard Conv1d format: (B=4, C=82, W=16)."""
    torch.manual_seed(42)
    return torch.randn(4, 82, 16)


@pytest.fixture
def batch_seq_format() -> torch.Tensor:
    """Batch in standard sequence/DataLoader format: (B=4, W=16, C=82)."""
    torch.manual_seed(42)
    return torch.randn(4, 16, 82)


# ---------------------------------------------------------------------------
# Test Conv1dEncoder
# ---------------------------------------------------------------------------
class TestConv1dEncoder:
    def test_default_output_shape(self, batch_conv_format):
        encoder = Conv1dEncoder(input_dim=82, window_size=16, latent_dim=16)
        z = encoder(batch_conv_format)
        assert z.shape == (4, 16), f"Expected shape (4, 16), got {z.shape}"

    def test_custom_latent_dim(self, batch_conv_format):
        encoder = Conv1dEncoder(input_dim=82, window_size=16, latent_dim=32)
        z = encoder(batch_conv_format)
        assert z.shape == (4, 32)

    def test_single_sample_batch(self):
        encoder = Conv1dEncoder(input_dim=82, window_size=16, latent_dim=16)
        x = torch.randn(1, 82, 16)
        z = encoder(x)
        assert z.shape == (1, 16)


# ---------------------------------------------------------------------------
# Test Conv1dDecoder
# ---------------------------------------------------------------------------
class TestConv1dDecoder:
    def test_default_output_shape(self):
        decoder = Conv1dDecoder(latent_dim=16, input_dim=82, window_size=16)
        z = torch.randn(4, 16)
        x_rec = decoder(z)
        assert x_rec.shape == (4, 82, 16), f"Expected (4, 82, 16), got {x_rec.shape}"

    @pytest.mark.parametrize("w", [8, 16, 32, 64])
    def test_variable_window_sizes(self, w):
        decoder = Conv1dDecoder(latent_dim=16, input_dim=82, window_size=w)
        z = torch.randn(2, 16)
        x_rec = decoder(z)
        assert x_rec.shape == (2, 82, w), f"Window size {w} produced shape {x_rec.shape}"


# ---------------------------------------------------------------------------
# Test SequenceAutoencoder
# ---------------------------------------------------------------------------
class TestSequenceAutoencoder:
    def test_forward_conv_format(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16, latent_dim=16)
        x_hat = model(batch_conv_format)
        assert x_hat.shape == batch_conv_format.shape

    def test_forward_seq_format(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16, latent_dim=16)
        x_hat = model(batch_seq_format)
        # Should return (B, W, D) matching the caller's format
        assert x_hat.shape == batch_seq_format.shape

    def test_encode_decode_methods(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16, latent_dim=16)
        z = model.encode(batch_conv_format)
        assert z.shape == (4, 16)
        x_rec = model.decode(z)
        assert x_rec.shape == (4, 82, 16)

    def test_gradients_backprop(self, batch_conv_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16, latent_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_hat = model(batch_conv_format)
        loss = nn.functional.mse_loss(x_hat, batch_conv_format)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive gradients"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_invalid_input_ndim(self):
        model = SequenceAutoencoder(input_dim=82, window_size=16)
        with pytest.raises(ValueError, match="Expected 3-D tensor"):
            model(torch.randn(82, 16))

    def test_invalid_input_dims(self):
        model = SequenceAutoencoder(input_dim=82, window_size=16)
        with pytest.raises(ValueError, match="incompatible"):
            model(torch.randn(4, 50, 50))

    def test_from_config(self, default_config):
        model = SequenceAutoencoder.from_config(default_config)
        assert model.input_dim == 82
        assert model.window_size == 16
        assert model.latent_dim == 16


# ---------------------------------------------------------------------------
# Test Reconstruction Error
# ---------------------------------------------------------------------------
class TestReconstructionError:
    def test_sample_reduction(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16)
        errors = model.compute_reconstruction_error(batch_seq_format, reduction="sample")
        assert errors.shape == (4,), f"Expected shape (4,), got {errors.shape}"
        assert (errors >= 0).all(), "Reconstruction error must be non-negative"

    def test_mean_reduction(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16)
        mean_err = model.compute_reconstruction_error(batch_seq_format, reduction="mean")
        assert mean_err.ndim == 0, "Expected scalar"
        assert mean_err.item() >= 0

    def test_mae_metric(self, batch_seq_format):
        model = SequenceAutoencoder(input_dim=82, window_size=16)
        errors_mae = model.compute_reconstruction_error(
            batch_seq_format, reduction="sample", metric="mae"
        )
        assert errors_mae.shape == (4,)
        assert (errors_mae >= 0).all()


# ---------------------------------------------------------------------------
# Test EarlyStopping and Fit
# ---------------------------------------------------------------------------
class TestTrainingUtilities:
    def test_early_stopping_trigger(self):
        model = nn.Linear(10, 10)
        es = EarlyStopping(patience=3, min_delta=1e-3)

        # Non-improving losses: 1.0, 1.0, 1.0, 1.0
        assert not es(1.0, model)
        assert not es(1.0, model)
        assert not es(1.0, model)
        # 4th time with patience=3 triggers early stopping
        assert es(1.0, model)
        assert es.early_stop is True

    def test_fit_autoencoder_smoke(self):
        """Smoke test running fit_autoencoder on a tiny dataset for 2 epochs."""
        model = SequenceAutoencoder(input_dim=82, window_size=16, latent_dim=8)

        # Synthetic data: 32 windows of length 16 × 82 features
        x_dummy = torch.randn(32, 16, 82)
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
            loaded = torch.load(ckpt_path, weights_only=False)
            assert "model_state_dict" in loaded