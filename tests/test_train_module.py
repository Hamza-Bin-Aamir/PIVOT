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


class TestMixedPrecisionTraining:
    """Test suite for mixed precision training functionality."""

    def test_precision_default_fp32(self):
        """Test default precision is FP32."""
        model = LitNoduleDetection()

        assert model.hparams.precision == "32"
        assert model.precision == "32"

    def test_precision_fp16_mixed(self):
        """Test FP16 mixed precision initialization."""
        model = LitNoduleDetection(precision="16-mixed")

        assert model.hparams.precision == "16-mixed"
        assert model.precision == "16-mixed"

    def test_precision_bf16_mixed(self):
        """Test BF16 mixed precision initialization."""
        model = LitNoduleDetection(precision="bf16-mixed")

        assert model.hparams.precision == "bf16-mixed"
        assert model.precision == "bf16-mixed"

    def test_precision_invalid_value(self):
        """Test that invalid precision values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid precision"):
            LitNoduleDetection(precision="invalid")

        with pytest.raises(ValueError, match="Invalid precision"):
            LitNoduleDetection(precision="16")

        with pytest.raises(ValueError, match="Invalid precision"):
            LitNoduleDetection(precision="bf16")

    def test_precision_valid_values_accepted(self):
        """Test that all valid precision values are accepted."""
        valid_precisions = ["32", "16-mixed", "bf16-mixed"]

        for precision in valid_precisions:
            model = LitNoduleDetection(precision=precision)
            assert model.precision == precision

    def test_precision_preserved_in_hparams(self):
        """Test that precision is saved in hyperparameters."""
        model = LitNoduleDetection(precision="16-mixed")

        # Precision should be in hparams for checkpointing
        assert "precision" in model.hparams
        assert model.hparams.precision == "16-mixed"

    def test_forward_with_fp16_mixed(self):
        """Test forward pass works with FP16 mixed precision."""
        model = LitNoduleDetection(precision="16-mixed")

        # Use smaller batch/volume to avoid memory issues in tests
        x = torch.randn(1, 1, 32, 32, 32)

        predictions = model(x)

        # Verify outputs are generated correctly
        assert isinstance(predictions, dict)
        assert "segmentation" in predictions
        assert "center" in predictions
        assert "size" in predictions
        assert "triage" in predictions

    def test_forward_with_bf16_mixed(self):
        """Test forward pass works with BF16 mixed precision."""
        model = LitNoduleDetection(precision="bf16-mixed")

        x = torch.randn(1, 1, 32, 32, 32)

        predictions = model(x)

        # Verify outputs are generated correctly
        assert isinstance(predictions, dict)
        assert "segmentation" in predictions
        assert "center" in predictions
        assert "size" in predictions
        assert "triage" in predictions

    def test_training_step_with_fp16_mixed(self):
        """Test training step with FP16 mixed precision."""
        model = LitNoduleDetection(precision="16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        # Verify loss is computed correctly
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_with_bf16_mixed(self):
        """Test training step with BF16 mixed precision."""
        model = LitNoduleDetection(precision="bf16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        # Verify loss is computed correctly
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step_with_mixed_precision(self):
        """Test validation step with mixed precision."""
        model = LitNoduleDetection(precision="16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_test_step_with_mixed_precision(self):
        """Test test step with mixed precision."""
        model = LitNoduleDetection(precision="bf16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.test_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_precision_with_custom_model_params(self):
        """Test that precision works with custom model parameters."""
        model = LitNoduleDetection(
            model_depth=5,
            init_features=64,
            precision="16-mixed",
        )

        assert model.hparams.model_depth == 5
        assert model.hparams.init_features == 64
        assert model.hparams.precision == "16-mixed"

        # Model should still be created correctly
        # Use larger input for depth=5 model to avoid BatchNorm issues
        x = torch.randn(1, 1, 64, 64, 64)
        predictions = model(x)
        assert predictions["segmentation"].shape == (1, 1, 64, 64, 64)

    def test_precision_with_custom_loss_weights(self):
        """Test that precision works with custom loss weights."""
        model = LitNoduleDetection(
            seg_weight=2.0,
            center_weight=1.5,
            precision="bf16-mixed",
        )

        assert model.hparams.seg_weight == 2.0
        assert model.hparams.center_weight == 1.5
        assert model.hparams.precision == "bf16-mixed"

    def test_optimizer_config_with_mixed_precision(self):
        """Test that optimizer configuration works with mixed precision."""
        model = LitNoduleDetection(
            learning_rate=1e-3,
            weight_decay=1e-4,
            precision="16-mixed",
        )

        config = model.configure_optimizers()

        # Optimizer should be configured normally
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["optimizer"].param_groups[0]["lr"] == 1e-3
        assert config["optimizer"].param_groups[0]["weight_decay"] == 1e-4

    def test_backward_pass_with_fp16_mixed(self):
        """Test that backward pass works with FP16 mixed precision."""
        model = LitNoduleDetection(precision="16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        # Verify backward pass doesn't crash
        loss.backward()

        # Verify gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients

    def test_backward_pass_with_bf16_mixed(self):
        """Test that backward pass works with BF16 mixed precision."""
        model = LitNoduleDetection(precision="bf16-mixed")

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        # Verify backward pass doesn't crash
        loss.backward()

        # Verify gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients

    def test_precision_numerical_stability(self):
        """Test that mixed precision produces reasonable loss values."""
        torch.manual_seed(42)

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        # Test FP32
        torch.manual_seed(42)
        model_fp32 = LitNoduleDetection(precision="32")
        loss_fp32 = model_fp32.training_step(batch, 0)

        # Test FP16
        torch.manual_seed(42)
        model_fp16 = LitNoduleDetection(precision="16-mixed")
        loss_fp16 = model_fp16.training_step(batch, 0)

        # Test BF16
        torch.manual_seed(42)
        model_bf16 = LitNoduleDetection(precision="bf16-mixed")
        loss_bf16 = model_bf16.training_step(batch, 0)

        # All losses should be finite and positive
        assert torch.isfinite(loss_fp32)
        assert torch.isfinite(loss_fp16)
        assert torch.isfinite(loss_bf16)
        assert loss_fp32.item() > 0
        assert loss_fp16.item() > 0
        assert loss_bf16.item() > 0

    def test_precision_case_sensitive(self):
        """Test that precision parameter is case-sensitive."""
        # Valid lowercase values should work
        model = LitNoduleDetection(precision="32")
        assert model.precision == "32"

        # Uppercase should fail
        with pytest.raises(ValueError):
            LitNoduleDetection(precision="FP32")

        with pytest.raises(ValueError):
            LitNoduleDetection(precision="16-MIXED")

        with pytest.raises(ValueError):
            LitNoduleDetection(precision="BF16-MIXED")


class TestHardNegativeMiningIntegration:
    """Test suite for hard negative mining integration in training module."""

    def test_init_with_hard_negative_mining_disabled(self):
        """Test initialization with hard negative mining disabled (default)."""
        model = LitNoduleDetection()

        assert model.hparams.use_hard_negative_mining is False
        assert not hasattr(model, "hard_negative_ratio") or model.hparams.hard_negative_ratio == 3.0

    def test_init_with_hard_negative_mining_enabled(self):
        """Test initialization with hard negative mining enabled."""
        model = LitNoduleDetection(use_hard_negative_mining=True)

        assert model.hparams.use_hard_negative_mining is True
        assert model.use_hard_negative_mining is True

    def test_init_with_custom_hard_negative_ratio(self):
        """Test initialization with custom hard negative ratio."""
        model = LitNoduleDetection(
            use_hard_negative_mining=True, hard_negative_ratio=5.0, min_negative_samples=200
        )

        assert model.hparams.use_hard_negative_mining is True
        assert model.hparams.hard_negative_ratio == 5.0
        assert model.hparams.min_negative_samples == 200

    def test_init_invalid_hard_negative_ratio(self):
        """Test that invalid hard_negative_ratio raises ValueError."""
        with pytest.raises(ValueError, match="hard_negative_ratio must be positive"):
            LitNoduleDetection(use_hard_negative_mining=True, hard_negative_ratio=0.0)

        with pytest.raises(ValueError, match="hard_negative_ratio must be positive"):
            LitNoduleDetection(use_hard_negative_mining=True, hard_negative_ratio=-1.0)

    def test_init_invalid_min_negative_samples(self):
        """Test that invalid min_negative_samples raises ValueError."""
        with pytest.raises(ValueError, match="min_negative_samples must be >= 0"):
            LitNoduleDetection(use_hard_negative_mining=True, min_negative_samples=-1)

    def test_center_loss_wrapped_when_enabled(self):
        """Test that center loss is wrapped with HardNegativeMiningLoss when enabled."""
        from src.loss.hard_negative_mining import HardNegativeMiningLoss

        model = LitNoduleDetection(use_hard_negative_mining=True)

        # Center loss should be wrapped
        assert isinstance(model.loss_fn.center_loss, HardNegativeMiningLoss)

    def test_center_loss_not_wrapped_when_disabled(self):
        """Test that center loss is not wrapped when hard negative mining is disabled."""
        from src.loss.focal import FocalLoss
        from src.loss.hard_negative_mining import HardNegativeMiningLoss

        model = LitNoduleDetection(use_hard_negative_mining=False)

        # Center loss should be original FocalLoss, not wrapped
        assert isinstance(model.loss_fn.center_loss, FocalLoss)
        assert not isinstance(model.loss_fn.center_loss, HardNegativeMiningLoss)

    def test_training_step_with_hard_negative_mining(self):
        """Test training step with hard negative mining enabled."""
        model = LitNoduleDetection(use_hard_negative_mining=True, hard_negative_ratio=3.0)

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
        assert not torch.isnan(loss)

    def test_validation_step_with_hard_negative_mining(self):
        """Test validation step with hard negative mining enabled."""
        model = LitNoduleDetection(use_hard_negative_mining=True, hard_negative_ratio=3.0)

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_backward_pass_with_hard_negative_mining(self):
        """Test backward pass with hard negative mining enabled."""
        model = LitNoduleDetection(use_hard_negative_mining=True, hard_negative_ratio=3.0)

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)
        loss.backward()

        # Verify gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients

    def test_hard_negative_mining_with_mixed_precision(self):
        """Test hard negative mining works with mixed precision."""
        model = LitNoduleDetection(
            use_hard_negative_mining=True, hard_negative_ratio=3.0, precision="16-mixed"
        )

        assert model.use_hard_negative_mining is True
        assert model.precision == "16-mixed"

        batch = {
            "image": torch.randn(1, 1, 32, 32, 32),
            "segmentation": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "center": torch.randint(0, 2, (1, 1, 32, 32, 32)).float(),
            "size": torch.randn(1, 3, 1, 1, 1),
            "triage": torch.randint(0, 2, (1, 1, 1, 1, 1)).float(),
        }

        loss = model.training_step(batch, 0)

        assert torch.isfinite(loss)

    def test_hard_negative_mining_with_custom_loss_weights(self):
        """Test hard negative mining with custom task weights."""
        model = LitNoduleDetection(
            use_hard_negative_mining=True,
            hard_negative_ratio=4.0,
            center_weight=2.0,
            seg_weight=1.5,
        )

        assert model.hparams.center_weight == 2.0
        assert model.hparams.seg_weight == 1.5
        assert model.use_hard_negative_mining is True

    def test_hyperparameter_saving_with_hard_negative_mining(self):
        """Test that hard negative mining params are saved in hyperparameters."""
        model = LitNoduleDetection(
            use_hard_negative_mining=True, hard_negative_ratio=5.0, min_negative_samples=250
        )

        assert "use_hard_negative_mining" in model.hparams
        assert "hard_negative_ratio" in model.hparams
        assert "min_negative_samples" in model.hparams
        assert model.hparams.use_hard_negative_mining is True
        assert model.hparams.hard_negative_ratio == 5.0
        assert model.hparams.min_negative_samples == 250


class TestCheckpointing:
    """Test suite for model checkpointing functionality."""

    def test_init_default_checkpoint_params(self):
        """Test default checkpoint parameter initialization."""
        model = LitNoduleDetection()

        assert model.hparams.checkpoint_dir == "checkpoints"
        assert model.hparams.checkpoint_monitor == "val/loss"
        assert model.hparams.checkpoint_mode == "min"
        assert model.hparams.checkpoint_save_top_k == 3
        assert model.hparams.checkpoint_save_last is True
        assert model.hparams.checkpoint_every_n_epochs == 1
        assert model.hparams.checkpoint_filename == "epoch={epoch:02d}-val_loss={val/loss:.4f}"

    def test_init_custom_checkpoint_params(self):
        """Test initialization with custom checkpoint parameters."""
        model = LitNoduleDetection(
            checkpoint_dir="custom_checkpoints",
            checkpoint_monitor="val/dice",
            checkpoint_mode="max",
            checkpoint_save_top_k=5,
            checkpoint_save_last=False,
            checkpoint_every_n_epochs=2,
            checkpoint_filename="best-{epoch:03d}",
        )

        assert model.hparams.checkpoint_dir == "custom_checkpoints"
        assert model.hparams.checkpoint_monitor == "val/dice"
        assert model.hparams.checkpoint_mode == "max"
        assert model.hparams.checkpoint_save_top_k == 5
        assert model.hparams.checkpoint_save_last is False
        assert model.hparams.checkpoint_every_n_epochs == 2
        assert model.hparams.checkpoint_filename == "best-{epoch:03d}"

    def test_init_invalid_checkpoint_mode(self):
        """Test that invalid checkpoint_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid checkpoint_mode"):
            LitNoduleDetection(checkpoint_mode="invalid")

        with pytest.raises(ValueError, match="Invalid checkpoint_mode"):
            LitNoduleDetection(checkpoint_mode="maximize")

    def test_init_invalid_checkpoint_save_top_k(self):
        """Test that invalid checkpoint_save_top_k raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_save_top_k must be >= 1"):
            LitNoduleDetection(checkpoint_save_top_k=0)

        with pytest.raises(ValueError, match="checkpoint_save_top_k must be >= 1"):
            LitNoduleDetection(checkpoint_save_top_k=-1)

    def test_init_invalid_checkpoint_every_n_epochs(self):
        """Test that invalid checkpoint_every_n_epochs raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_every_n_epochs must be >= 1"):
            LitNoduleDetection(checkpoint_every_n_epochs=0)

        with pytest.raises(ValueError, match="checkpoint_every_n_epochs must be >= 1"):
            LitNoduleDetection(checkpoint_every_n_epochs=-1)

    def test_configure_callbacks_returns_list(self):
        """Test that configure_callbacks returns a list."""
        model = LitNoduleDetection()
        callbacks = model.configure_callbacks()

        assert isinstance(callbacks, list)
        assert len(callbacks) > 0

    def test_configure_callbacks_contains_checkpoint(self):
        """Test that configure_callbacks includes ModelCheckpoint callback."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection()
        callbacks = model.configure_callbacks()

        # Should have at least one ModelCheckpoint callback
        checkpoint_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(checkpoint_callbacks) == 1

    def test_checkpoint_callback_configuration(self):
        """Test that checkpoint callback is configured correctly."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(
            checkpoint_dir="test_checkpoints",
            checkpoint_monitor="val/dice",
            checkpoint_mode="max",
            checkpoint_save_top_k=10,
        )
        callbacks = model.configure_callbacks()

        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert checkpoint_cb.dirpath.endswith("test_checkpoints")
        assert checkpoint_cb.monitor == "val/dice"
        assert checkpoint_cb.mode == "max"
        assert checkpoint_cb.save_top_k == 10

    def test_checkpoint_callback_save_last(self):
        """Test checkpoint callback save_last configuration."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        # Test with save_last=True
        model = LitNoduleDetection(checkpoint_save_last=True)
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        assert checkpoint_cb.save_last is True

        # Test with save_last=False
        model = LitNoduleDetection(checkpoint_save_last=False)
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        assert checkpoint_cb.save_last is False

    def test_checkpoint_callback_every_n_epochs(self):
        """Test checkpoint callback every_n_epochs configuration."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(checkpoint_every_n_epochs=5)
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        assert checkpoint_cb.every_n_epochs == 5

    def test_checkpoint_callback_filename(self):
        """Test checkpoint callback filename configuration."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(checkpoint_filename="model-{epoch:04d}")
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        assert checkpoint_cb.filename == "model-{epoch:04d}"

    def test_hyperparameter_saving_with_checkpointing(self):
        """Test that checkpoint params are saved in hyperparameters."""
        model = LitNoduleDetection(
            checkpoint_dir="my_checkpoints",
            checkpoint_monitor="val/loss",
            checkpoint_mode="min",
            checkpoint_save_top_k=7,
            checkpoint_save_last=True,
            checkpoint_every_n_epochs=3,
        )

        assert "checkpoint_dir" in model.hparams
        assert "checkpoint_monitor" in model.hparams
        assert "checkpoint_mode" in model.hparams
        assert "checkpoint_save_top_k" in model.hparams
        assert "checkpoint_save_last" in model.hparams
        assert "checkpoint_every_n_epochs" in model.hparams

    def test_checkpoint_with_all_custom_params(self):
        """Test checkpoint configuration with all custom parameters."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(
            checkpoint_dir="all_custom",
            checkpoint_monitor="val/f1",
            checkpoint_mode="max",
            checkpoint_save_top_k=15,
            checkpoint_save_last=True,
            checkpoint_every_n_epochs=10,
            checkpoint_filename="best_model_{epoch}_{val/f1:.3f}",
        )

        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert checkpoint_cb.dirpath.endswith("all_custom")
        assert checkpoint_cb.monitor == "val/f1"
        assert checkpoint_cb.mode == "max"
        assert checkpoint_cb.save_top_k == 15
        assert checkpoint_cb.save_last is True
        assert checkpoint_cb.every_n_epochs == 10
        assert checkpoint_cb.filename == "best_model_{epoch}_{val/f1:.3f}"

    def test_checkpoint_mode_min_for_loss(self):
        """Test checkpoint mode='min' for loss metrics."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(checkpoint_monitor="val/loss", checkpoint_mode="min")
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert checkpoint_cb.monitor == "val/loss"
        assert checkpoint_cb.mode == "min"

    def test_checkpoint_mode_max_for_accuracy(self):
        """Test checkpoint mode='max' for accuracy/dice metrics."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        model = LitNoduleDetection(checkpoint_monitor="val/dice", checkpoint_mode="max")
        callbacks = model.configure_callbacks()
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert checkpoint_cb.monitor == "val/dice"
        assert checkpoint_cb.mode == "max"


class TestEarlyStopping:
    """Test suite for early stopping functionality."""

    def test_init_default_early_stopping_params(self):
        """Test default early stopping parameter initialization."""
        model = LitNoduleDetection()

        assert model.hparams.early_stopping_monitor == "val/loss"
        assert model.hparams.early_stopping_patience == 10
        assert model.hparams.early_stopping_mode == "min"
        assert model.hparams.early_stopping_min_delta == 0.0

    def test_init_custom_early_stopping_params(self):
        """Test initialization with custom early stopping parameters."""
        model = LitNoduleDetection(
            early_stopping_monitor="val/dice",
            early_stopping_patience=20,
            early_stopping_mode="max",
            early_stopping_min_delta=0.001,
        )

        assert model.hparams.early_stopping_monitor == "val/dice"
        assert model.hparams.early_stopping_patience == 20
        assert model.hparams.early_stopping_mode == "max"
        assert model.hparams.early_stopping_min_delta == 0.001

    def test_init_invalid_early_stopping_mode(self):
        """Test that invalid early_stopping_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid early_stopping_mode"):
            LitNoduleDetection(early_stopping_mode="invalid")

        with pytest.raises(ValueError, match="Invalid early_stopping_mode"):
            LitNoduleDetection(early_stopping_mode="minimize")

    def test_init_invalid_early_stopping_patience(self):
        """Test that invalid early_stopping_patience raises ValueError."""
        with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
            LitNoduleDetection(early_stopping_patience=0)

        with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
            LitNoduleDetection(early_stopping_patience=-1)

    def test_init_invalid_early_stopping_min_delta(self):
        """Test that invalid early_stopping_min_delta raises ValueError."""
        with pytest.raises(ValueError, match="early_stopping_min_delta must be >= 0"):
            LitNoduleDetection(early_stopping_min_delta=-0.001)

        with pytest.raises(ValueError, match="early_stopping_min_delta must be >= 0"):
            LitNoduleDetection(early_stopping_min_delta=-1.0)

    def test_configure_callbacks_contains_early_stopping(self):
        """Test that configure_callbacks includes EarlyStopping callback."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection()
        callbacks = model.configure_callbacks()

        # Should have at least one EarlyStopping callback
        early_stopping_callbacks = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]
        assert len(early_stopping_callbacks) == 1

    def test_early_stopping_callback_configuration(self):
        """Test that early stopping callback is configured correctly."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection(
            early_stopping_monitor="val/f1",
            early_stopping_patience=25,
            early_stopping_mode="max",
            early_stopping_min_delta=0.01,
        )
        callbacks = model.configure_callbacks()

        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))

        assert early_stopping_cb.monitor == "val/f1"
        assert early_stopping_cb.patience == 25
        assert early_stopping_cb.mode == "max"
        assert early_stopping_cb.min_delta == 0.01

    def test_early_stopping_mode_min_for_loss(self):
        """Test early stopping mode='min' for loss metrics."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection(early_stopping_monitor="val/loss", early_stopping_mode="min")
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))

        assert early_stopping_cb.monitor == "val/loss"
        assert early_stopping_cb.mode == "min"

    def test_early_stopping_mode_max_for_accuracy(self):
        """Test early stopping mode='max' for accuracy/dice metrics."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection(early_stopping_monitor="val/dice", early_stopping_mode="max")
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))

        assert early_stopping_cb.monitor == "val/dice"
        assert early_stopping_cb.mode == "max"

    def test_early_stopping_with_custom_patience(self):
        """Test early stopping with custom patience values."""
        from lightning.pytorch.callbacks import EarlyStopping

        # Short patience
        model = LitNoduleDetection(early_stopping_patience=5)
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        assert early_stopping_cb.patience == 5

        # Long patience
        model = LitNoduleDetection(early_stopping_patience=100)
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        assert early_stopping_cb.patience == 100

    def test_early_stopping_with_min_delta(self):
        """Test early stopping with minimum delta for improvement."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection(early_stopping_min_delta=0.005)
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        # Lightning adjusts min_delta based on mode, so check absolute value
        assert abs(early_stopping_cb.min_delta) == 0.005

    def test_hyperparameter_saving_with_early_stopping(self):
        """Test that early stopping params are saved in hyperparameters."""
        model = LitNoduleDetection(
            early_stopping_monitor="val/accuracy",
            early_stopping_patience=15,
            early_stopping_mode="max",
            early_stopping_min_delta=0.002,
        )

        assert "early_stopping_monitor" in model.hparams
        assert "early_stopping_patience" in model.hparams
        assert "early_stopping_mode" in model.hparams
        assert "early_stopping_min_delta" in model.hparams

    def test_early_stopping_and_checkpointing_together(self):
        """Test that both early stopping and checkpointing work together."""
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

        model = LitNoduleDetection(
            checkpoint_monitor="val/dice",
            checkpoint_mode="max",
            early_stopping_monitor="val/loss",
            early_stopping_mode="min",
            early_stopping_patience=20,
        )
        callbacks = model.configure_callbacks()

        # Should have both callbacks
        checkpoint_cbs = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        early_stopping_cbs = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

        assert len(checkpoint_cbs) == 1
        assert len(early_stopping_cbs) == 1

        # Verify they can monitor different metrics
        assert checkpoint_cbs[0].monitor == "val/dice"
        assert early_stopping_cbs[0].monitor == "val/loss"

    def test_early_stopping_with_zero_min_delta(self):
        """Test early stopping with zero minimum delta."""
        from lightning.pytorch.callbacks import EarlyStopping

        model = LitNoduleDetection(early_stopping_min_delta=0.0)
        callbacks = model.configure_callbacks()
        early_stopping_cb = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        assert early_stopping_cb.min_delta == 0.0
