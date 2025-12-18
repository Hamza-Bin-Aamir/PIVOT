"""Tests for training progress callback."""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning as L
import pytest
import torch

from src.train.progress_callback import TrainingProgressCallback


class TestTrainingProgressCallback:
    """Test suite for TrainingProgressCallback."""

    def test_init(self):
        """Test callback initialization."""
        callback = TrainingProgressCallback()

        assert callback.state_tracker is not None
        assert callback.train_metrics is not None
        assert callback.val_metrics is not None
        assert callback.state_tracker.phase == "idle"
        assert callback.state_tracker.epoch == 0
        assert callback.state_tracker.step_count == 0

    def test_on_train_epoch_start(self):
        """Test on_train_epoch_start sets correct state."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 5
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)

        assert callback.state_tracker.phase == "train"
        assert callback.state_tracker.epoch == 5
        assert callback.state_tracker.step_count == 0

    def test_on_train_epoch_start_resets_metrics(self):
        """Test on_train_epoch_start resets training metrics."""
        callback = TrainingProgressCallback()
        callback.train_metrics.add("loss", 0.5)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)

        assert len(callback.train_metrics) == 0

    def test_on_train_batch_end_increments_step(self):
        """Test on_train_batch_end increments step count."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        assert callback.state_tracker.step_count == 0

        callback.on_train_batch_end(trainer, pl_module, None, None, 0)
        assert callback.state_tracker.step_count == 1

        callback.on_train_batch_end(trainer, pl_module, None, None, 1)
        assert callback.state_tracker.step_count == 2

    def test_on_train_batch_end_collects_metrics(self):
        """Test on_train_batch_end collects training metrics."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "train/accuracy": torch.tensor(0.85),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        assert callback.train_metrics.has_metric("loss")
        assert callback.train_metrics.has_metric("accuracy")
        assert callback.train_metrics.get_latest("loss") == pytest.approx(0.5)
        assert callback.train_metrics.get_latest("accuracy") == pytest.approx(0.85)

    def test_on_train_batch_end_skips_epoch_metrics(self):
        """Test on_train_batch_end skips metrics ending with _epoch."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "train/loss_epoch": torch.tensor(0.4),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        assert callback.train_metrics.get_count("loss") == 1
        assert not callback.train_metrics.has_metric("loss_epoch")

    def test_on_train_batch_end_handles_invalid_metrics(self):
        """Test on_train_batch_end handles non-finite metrics gracefully."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/loss": torch.tensor(float("nan")),
            "train/valid": torch.tensor(0.5),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Should skip NaN but collect valid metric
        assert not callback.train_metrics.has_metric("loss")
        assert callback.train_metrics.has_metric("valid")

    def test_on_train_batch_end_handles_empty_metrics(self):
        """Test on_train_batch_end handles empty callback_metrics."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        assert len(callback.train_metrics) == 0

    def test_on_validation_epoch_start(self):
        """Test on_validation_epoch_start sets correct state."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 3
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)

        assert callback.state_tracker.phase == "validation"
        assert callback.state_tracker.epoch == 3
        assert callback.state_tracker.step_count == 0

    def test_on_validation_epoch_start_resets_metrics(self):
        """Test on_validation_epoch_start resets validation metrics."""
        callback = TrainingProgressCallback()
        callback.val_metrics.add("loss", 0.3)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)

        assert len(callback.val_metrics) == 0

    def test_on_validation_batch_end_increments_step(self):
        """Test on_validation_batch_end increments step count."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        assert callback.state_tracker.step_count == 0

        callback.on_validation_batch_end(trainer, pl_module, None, None, 0)
        assert callback.state_tracker.step_count == 1

        callback.on_validation_batch_end(trainer, pl_module, None, None, 1)
        assert callback.state_tracker.step_count == 2

    def test_on_validation_batch_end_collects_metrics(self):
        """Test on_validation_batch_end collects validation metrics."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "val/accuracy": torch.tensor(0.9),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_validation_batch_end(trainer, pl_module, None, None, 0)

        assert callback.val_metrics.has_metric("loss")
        assert callback.val_metrics.has_metric("accuracy")
        assert callback.val_metrics.get_latest("loss") == pytest.approx(0.3)
        assert callback.val_metrics.get_latest("accuracy") == pytest.approx(0.9)

    def test_on_validation_batch_end_skips_epoch_metrics(self):
        """Test on_validation_batch_end skips metrics ending with _epoch."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "val/loss_epoch": torch.tensor(0.25),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_validation_batch_end(trainer, pl_module, None, None, 0)

        assert callback.val_metrics.get_count("loss") == 1
        assert not callback.val_metrics.has_metric("loss_epoch")

    def test_on_train_epoch_end_sets_idle(self):
        """Test on_train_epoch_end sets state to idle."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        assert callback.state_tracker.phase == "train"

        callback.on_train_epoch_end(trainer, pl_module)
        assert callback.state_tracker.phase == "idle"

    def test_on_validation_epoch_end_sets_idle(self):
        """Test on_validation_epoch_end sets state to idle."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        assert callback.state_tracker.phase == "validation"

        callback.on_validation_epoch_end(trainer, pl_module)
        assert callback.state_tracker.phase == "idle"

    def test_get_train_stats_empty(self):
        """Test get_train_stats returns empty dict when no metrics."""
        callback = TrainingProgressCallback()

        stats = callback.get_train_stats()

        assert stats == {}

    def test_get_train_stats_with_metrics(self):
        """Test get_train_stats returns aggregated statistics."""
        callback = TrainingProgressCallback()
        callback.train_metrics.add("loss", 0.5)
        callback.train_metrics.add("loss", 0.3)

        stats = callback.get_train_stats()

        assert "loss" in stats
        assert stats["loss"]["mean"] == pytest.approx(0.4)
        assert stats["loss"]["count"] == 2

    def test_get_val_stats_empty(self):
        """Test get_val_stats returns empty dict when no metrics."""
        callback = TrainingProgressCallback()

        stats = callback.get_val_stats()

        assert stats == {}

    def test_get_val_stats_with_metrics(self):
        """Test get_val_stats returns aggregated statistics."""
        callback = TrainingProgressCallback()
        callback.val_metrics.add("loss", 0.3)
        callback.val_metrics.add("loss", 0.2)

        stats = callback.get_val_stats()

        assert "loss" in stats
        assert stats["loss"]["mean"] == pytest.approx(0.25)
        assert stats["loss"]["count"] == 2

    def test_get_progress_summary_initial(self):
        """Test get_progress_summary with initial state."""
        callback = TrainingProgressCallback()

        summary = callback.get_progress_summary()

        assert "state" in summary
        assert "train_stats" in summary
        assert "val_stats" in summary
        assert summary["state"]["phase"] == "idle"
        assert summary["state"]["epoch"] == 0
        assert summary["state"]["step"] == 0
        assert summary["train_stats"] == {}
        assert summary["val_stats"] == {}

    def test_get_progress_summary_with_data(self):
        """Test get_progress_summary with training data."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 5
        trainer.callback_metrics = {"train/loss": torch.tensor(0.5)}
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        summary = callback.get_progress_summary()

        assert summary["state"]["phase"] == "train"
        assert summary["state"]["epoch"] == 5
        assert summary["state"]["step"] == 1
        assert "loss" in summary["train_stats"]

    def test_complete_training_workflow(self):
        """Test complete training and validation workflow."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        pl_module = MagicMock(spec=L.LightningModule)

        # Training phase
        callback.on_train_epoch_start(trainer, pl_module)
        assert callback.state_tracker.is_training()

        for i in range(3):
            trainer.callback_metrics = {"train/loss": torch.tensor(0.5 - i * 0.1)}
            callback.on_train_batch_end(trainer, pl_module, None, None, i)

        assert callback.state_tracker.step_count == 3
        assert callback.train_metrics.get_count("loss") == 3

        callback.on_train_epoch_end(trainer, pl_module)
        assert callback.state_tracker.is_idle()

        # Validation phase
        callback.on_validation_epoch_start(trainer, pl_module)
        assert callback.state_tracker.is_validating()

        for i in range(2):
            trainer.callback_metrics = {"val/loss": torch.tensor(0.3 - i * 0.05)}
            callback.on_validation_batch_end(trainer, pl_module, None, None, i)

        assert callback.state_tracker.step_count == 2
        assert callback.val_metrics.get_count("loss") == 2

        callback.on_validation_epoch_end(trainer, pl_module)
        assert callback.state_tracker.is_idle()

        # Check final statistics
        train_stats = callback.get_train_stats()
        val_stats = callback.get_val_stats()

        assert train_stats["loss"]["count"] == 3
        assert val_stats["loss"]["count"] == 2

    def test_multiple_epochs(self):
        """Test metrics reset across multiple epochs."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        pl_module = MagicMock(spec=L.LightningModule)

        # Epoch 0
        trainer.current_epoch = 0
        callback.on_train_epoch_start(trainer, pl_module)
        trainer.callback_metrics = {"train/loss": torch.tensor(0.5)}
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)
        callback.on_train_epoch_end(trainer, pl_module)

        assert callback.train_metrics.get_count("loss") == 1

        # Epoch 1
        trainer.current_epoch = 1
        callback.on_train_epoch_start(trainer, pl_module)

        # Metrics should be reset
        assert callback.train_metrics.get_count("loss") == 0

        trainer.callback_metrics = {"train/loss": torch.tensor(0.3)}
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        assert callback.train_metrics.get_count("loss") == 1

    def test_callback_with_none_callback_metrics(self):
        """Test callback handles None callback_metrics."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = None
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Should not crash
        assert callback.state_tracker.step_count == 1
        assert len(callback.train_metrics) == 0

    def test_on_validation_batch_end_with_dataloader_idx(self):
        """Test on_validation_batch_end accepts dataloader_idx parameter."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {"val/loss": torch.tensor(0.3)}
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_validation_batch_end(
            trainer, pl_module, None, None, 0, _dataloader_idx=1
        )

        assert callback.state_tracker.step_count == 1
        assert callback.val_metrics.has_metric("loss")

    def test_on_train_batch_end_handles_non_numeric_metrics(self):
        """Test on_train_batch_end handles non-numeric metrics gracefully."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "train/valid": torch.tensor(0.5),
            "train/string": "not a number",
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Should collect valid metric and skip string
        assert callback.train_metrics.has_metric("valid")
        assert not callback.train_metrics.has_metric("string")

    def test_on_validation_batch_end_handles_non_numeric_metrics(self):
        """Test on_validation_batch_end handles non-numeric metrics gracefully."""
        callback = TrainingProgressCallback()
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/valid": torch.tensor(0.3),
            "val/string": "not a number",
        }
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_validation_epoch_start(trainer, pl_module)
        callback.on_validation_batch_end(trainer, pl_module, None, None, 0)

        # Should collect valid metric and skip string
        assert callback.val_metrics.has_metric("valid")
        assert not callback.val_metrics.has_metric("string")
