"""Tests for PyTorch Lightning training module."""

from __future__ import annotations

import pytest
import torch

from src.train import LitNoduleDetection

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestLitNoduleDetection:
    """Test suite for LitNoduleDetection."""

    def test_init_default(self):
        """Test default initialization."""
        model = LitNoduleDetection()

        assert model.hparams.model_depth == 4
        assert model.hparams.init_features == 32
        assert model.hparams.seg_weight == 1.0
        assert model.hparams.center_weight == 1.0
        assert model.hparams.size_weight == 1.0
        assert model.hparams.triage_weight == 1.0
        assert model.hparams.learning_rate == 1e-4
        assert model.hparams.weight_decay == 1e-5
        assert model.hparams.max_epochs == 100
        assert model.learning_rate == 1e-4
        assert model.weight_decay == 1e-5
        assert model.max_epochs == 100

    def test_init_custom_model_params(self):
        """Test initialization with custom model parameters."""
        model = LitNoduleDetection(
            model_depth=5,
            init_features=64,
        )

        assert model.hparams.model_depth == 5
        assert model.hparams.init_features == 64
        assert model.model.backbone.depth == 5
        assert model.model.backbone.init_features == 64

    def test_init_custom_task_weights(self):
        """Test initialization with custom task weights."""
        model = LitNoduleDetection(
            seg_weight=2.0,
            center_weight=1.5,
            size_weight=0.5,
            triage_weight=3.0,
        )

        assert model.hparams.seg_weight == 2.0
        assert model.hparams.center_weight == 1.5
        assert model.hparams.size_weight == 0.5
        assert model.hparams.triage_weight == 3.0
        assert model.loss_fn.seg_weight == 2.0
        assert model.loss_fn.center_weight == 1.5
        assert model.loss_fn.size_weight == 0.5
        assert model.loss_fn.triage_weight == 3.0

    def test_init_custom_optimizer_params(self):
        """Test initialization with custom optimizer parameters."""
        model = LitNoduleDetection(
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=200,
        )

        assert model.hparams.learning_rate == 1e-3
        assert model.hparams.weight_decay == 1e-4
        assert model.hparams.max_epochs == 200
        assert model.learning_rate == 1e-3
        assert model.weight_decay == 1e-4
        assert model.max_epochs == 200

    def test_init_custom_loss_kwargs(self):
        """Test initialization with custom loss kwargs."""
        model = LitNoduleDetection(
            seg_loss_kwargs={"smooth": 2.0},
            center_loss_kwargs={"alpha": 0.5, "gamma": 3.0},
            size_loss_kwargs={"beta": 0.5},
            triage_loss_kwargs={"pos_weight": 3.0},
        )

        assert model.loss_fn.seg_loss.smooth == 2.0
        assert model.loss_fn.center_loss.alpha == 0.5
        assert model.loss_fn.center_loss.gamma == 3.0
        assert model.loss_fn.size_loss.beta == 0.5
        assert model.loss_fn.triage_loss.pos_weight == 3.0

    def test_forward_basic(self):
        """Test basic forward pass."""
        model = LitNoduleDetection()
        x = torch.randn(1, 1, 64, 64, 64)

        predictions = model(x)

        assert isinstance(predictions, dict)
        assert "segmentation" in predictions
        assert "center" in predictions
        assert "size" in predictions
        assert "triage" in predictions
        assert predictions["segmentation"].shape == (1, 1, 64, 64, 64)
        assert predictions["center"].shape == (1, 1, 64, 64, 64)
        assert predictions["size"].shape == (1, 3, 1, 1, 1)
        assert predictions["triage"].shape == (1, 1, 1, 1, 1)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = LitNoduleDetection()

        # Test with smaller volumes to avoid memory crashes
        for batch_size in [1, 2]:
            x = torch.randn(batch_size, 1, 48, 48, 48)
            predictions = model(x)

            assert predictions["segmentation"].shape == (batch_size, 1, 48, 48, 48)
            assert predictions["center"].shape == (batch_size, 1, 48, 48, 48)
            assert predictions["size"].shape == (batch_size, 3, 1, 1, 1)
            assert predictions["triage"].shape == (batch_size, 1, 1, 1, 1)

    def test_task_weight_propagation(self):
        """Test that task weights are properly propagated to loss function."""
        model = LitNoduleDetection(
            seg_weight=3.0,
            center_weight=2.5,
            size_weight=1.5,
            triage_weight=4.0,
        )

        weights = model.loss_fn.get_task_weights()

        assert weights["segmentation"] == 3.0
        assert weights["center"] == 2.5
        assert weights["size"] == 1.5
        assert weights["triage"] == 4.0

    def test_consistent_outputs(self):
        """Test that forward pass produces consistent outputs for same input."""
        model = LitNoduleDetection()
        model.eval()

        x = torch.randn(1, 1, 48, 48, 48)

        with torch.no_grad():
            pred1 = model(x)
            pred2 = model(x)

        # Outputs should be identical
        assert torch.allclose(pred1["segmentation"], pred2["segmentation"])
        assert torch.allclose(pred1["center"], pred2["center"])
        assert torch.allclose(pred1["size"], pred2["size"])
        assert torch.allclose(pred1["triage"], pred2["triage"])

    def test_optimizer_parameters(self):
        """Test optimizer has correct parameters."""
        model = LitNoduleDetection(learning_rate=2e-4, weight_decay=1e-6)

        config = model.configure_optimizers()
        optimizer = config["optimizer"]

        # Check that optimizer has model parameters
        param_count = sum(p.numel() for p in optimizer.param_groups[0]["params"])
        model_param_count = sum(p.numel() for p in model.parameters())

        assert param_count == model_param_count

    def test_hyperparameter_saving(self):
        """Test that hyperparameters are saved correctly."""
        model = LitNoduleDetection(
            model_depth=5,
            init_features=64,
            seg_weight=2.0,
            center_weight=1.5,
            learning_rate=1e-3,
            max_epochs=200,
        )

        # Check hparams
        assert model.hparams.model_depth == 5
        assert model.hparams.init_features == 64
        assert model.hparams.seg_weight == 2.0
        assert model.hparams.center_weight == 1.5
        assert model.hparams.learning_rate == 1e-3
        assert model.hparams.max_epochs == 200

    def test_training_step(self):
        """Test training step."""
        model = LitNoduleDetection()

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_validation_step(self):
        """Test validation step."""
        model = LitNoduleDetection()

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_test_step(self):
        """Test test step."""
        model = LitNoduleDetection()

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.test_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = LitNoduleDetection(learning_rate=1e-3, weight_decay=1e-4, max_epochs=50)

        config = model.configure_optimizers()

        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["interval"] == "epoch"
        assert config["lr_scheduler"]["frequency"] == 1

        optimizer = config["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_loss_reduction_toggling(self):
        """Test that loss reduction mode is properly toggled during steps."""
        model = LitNoduleDetection()

        # Initial reduction should be mean
        assert model.loss_fn.reduction == "mean"

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        # After training step, should be back to mean
        model.training_step(batch, 0)
        assert model.loss_fn.reduction == "mean"

        # After validation step, should be back to mean
        model.validation_step(batch, 0)
        assert model.loss_fn.reduction == "mean"

        # After test step, should be back to mean
        model.test_step(batch, 0)
        assert model.loss_fn.reduction == "mean"

    def test_on_train_epoch_end(self):
        """Test on_train_epoch_end hook logs learning rate."""
        from unittest.mock import MagicMock

        model = LitNoduleDetection(learning_rate=1e-3)

        # Mock the trainer and optimizer
        model.trainer = MagicMock()
        optimizer = MagicMock()
        optimizer.param_groups = [{"lr": 1e-3}]
        model.trainer.optimizers = [optimizer]

        # Mock the log method to track calls
        log_calls = []
        original_log = model.log

        def mock_log(name, value, **kwargs):
            log_calls.append((name, value, kwargs))
            return original_log(name, value, **kwargs)

        model.log = mock_log

        # Call the hook
        model.on_train_epoch_end()

        # Verify learning rate was logged
        assert len(log_calls) == 1
        assert log_calls[0][0] == "train/lr"
        assert log_calls[0][1] == 1e-3
        assert log_calls[0][2]["on_step"] is False
        assert log_calls[0][2]["on_epoch"] is True
