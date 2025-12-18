"""Tests for real-time metrics streamer."""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning as L
import pytest
import torch

from src.train.metrics_streamer import MetricsStreamer


class TestMetricsStreamer:
    """Test suite for MetricsStreamer."""

    def test_init(self):
        """Test initialization creates empty callback list."""
        streamer = MetricsStreamer()

        assert streamer.get_callback_count() == 0
        assert streamer._callbacks == []

    def test_register_callback(self):
        """Test registering a callback function."""
        streamer = MetricsStreamer()
        callback = MagicMock()

        streamer.register_callback(callback)

        assert streamer.get_callback_count() == 1
        assert callback in streamer._callbacks

    def test_register_multiple_callbacks(self):
        """Test registering multiple callback functions."""
        streamer = MetricsStreamer()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        streamer.register_callback(callback1)
        streamer.register_callback(callback2)
        streamer.register_callback(callback3)

        assert streamer.get_callback_count() == 3
        assert callback1 in streamer._callbacks
        assert callback2 in streamer._callbacks
        assert callback3 in streamer._callbacks

    def test_register_callback_not_callable_raises_error(self):
        """Test registering non-callable raises TypeError."""
        streamer = MetricsStreamer()

        with pytest.raises(TypeError, match="Callback must be callable"):
            streamer.register_callback("not a function")  # type: ignore[arg-type]

    def test_register_callback_none_raises_error(self):
        """Test registering None raises TypeError."""
        streamer = MetricsStreamer()

        with pytest.raises(TypeError, match="Callback must be callable"):
            streamer.register_callback(None)  # type: ignore[arg-type]

    def test_unregister_callback(self):
        """Test unregistering a callback function."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)

        streamer.unregister_callback(callback)

        assert streamer.get_callback_count() == 0
        assert callback not in streamer._callbacks

    def test_unregister_callback_not_registered_raises_error(self):
        """Test unregistering non-registered callback raises ValueError."""
        streamer = MetricsStreamer()
        callback = MagicMock()

        with pytest.raises(ValueError, match="Callback not registered"):
            streamer.unregister_callback(callback)

    def test_unregister_callback_removes_only_specified(self):
        """Test unregistering removes only the specified callback."""
        streamer = MetricsStreamer()
        callback1 = MagicMock()
        callback2 = MagicMock()
        streamer.register_callback(callback1)
        streamer.register_callback(callback2)

        streamer.unregister_callback(callback1)

        assert streamer.get_callback_count() == 1
        assert callback1 not in streamer._callbacks
        assert callback2 in streamer._callbacks

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        streamer = MetricsStreamer()
        callback1 = MagicMock()
        callback2 = MagicMock()
        streamer.register_callback(callback1)
        streamer.register_callback(callback2)

        streamer.clear_callbacks()

        assert streamer.get_callback_count() == 0
        assert streamer._callbacks == []

    def test_clear_callbacks_empty(self):
        """Test clearing callbacks when none registered."""
        streamer = MetricsStreamer()

        streamer.clear_callbacks()

        assert streamer.get_callback_count() == 0

    def test_get_callback_count_zero(self):
        """Test get_callback_count returns 0 initially."""
        streamer = MetricsStreamer()

        assert streamer.get_callback_count() == 0

    def test_get_callback_count_multiple(self):
        """Test get_callback_count returns correct count."""
        streamer = MetricsStreamer()
        streamer.register_callback(MagicMock())
        streamer.register_callback(MagicMock())
        streamer.register_callback(MagicMock())

        assert streamer.get_callback_count() == 3

    def test_emit_calls_callback(self):
        """Test _emit calls registered callback."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)

        streamer._emit("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_calls_multiple_callbacks(self):
        """Test _emit calls all registered callbacks."""
        streamer = MetricsStreamer()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()
        streamer.register_callback(callback1)
        streamer.register_callback(callback2)
        streamer.register_callback(callback3)

        streamer._emit("test_event", {"data": 123})

        callback1.assert_called_once_with("test_event", {"data": 123})
        callback2.assert_called_once_with("test_event", {"data": 123})
        callback3.assert_called_once_with("test_event", {"data": 123})

    def test_emit_handles_callback_exception(self):
        """Test _emit continues if callback raises exception."""
        streamer = MetricsStreamer()
        failing_callback = MagicMock(side_effect=RuntimeError("Test error"))
        success_callback = MagicMock()
        streamer.register_callback(failing_callback)
        streamer.register_callback(success_callback)

        # Should not raise exception
        streamer._emit("test_event", {"data": "test"})

        failing_callback.assert_called_once()
        success_callback.assert_called_once()

    def test_emit_no_callbacks_registered(self):
        """Test _emit does nothing when no callbacks registered."""
        streamer = MetricsStreamer()

        # Should not raise exception
        streamer._emit("test_event", {"data": "test"})

    def test_on_train_epoch_start_emits_event(self):
        """Test on_train_epoch_start emits correct event."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 5
        trainer.global_step = 100
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_start(trainer, pl_module)

        callback.assert_called_once_with(
            "train_epoch_start",
            {"epoch": 5, "global_step": 100},
        )

    def test_on_train_batch_end_emits_event_with_metrics(self):
        """Test on_train_batch_end emits event with metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 3
        trainer.global_step = 50
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "train/accuracy": torch.tensor(0.85),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_batch_end(trainer, pl_module, None, None, 10)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "train_batch"
        assert call_args[1]["epoch"] == 3
        assert call_args[1]["batch_idx"] == 10
        assert call_args[1]["global_step"] == 50
        assert call_args[1]["metrics"]["train/loss"] == pytest.approx(0.5)
        assert call_args[1]["metrics"]["train/accuracy"] == pytest.approx(0.85)

    def test_on_train_batch_end_skips_non_train_metrics(self):
        """Test on_train_batch_end only includes train/ metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "val/loss": torch.tensor(0.3),
            "other_metric": torch.tensor(0.7),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_batch_end(trainer, pl_module, None, None, 0)

        call_args = callback.call_args[0]
        assert call_args[0] == "train_batch"
        assert "train/loss" in call_args[1]["metrics"]
        assert "val/loss" not in call_args[1]["metrics"]
        assert "other_metric" not in call_args[1]["metrics"]

    def test_on_train_batch_end_no_metrics_no_emit(self):
        """Test on_train_batch_end doesn't emit if no train metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_batch_end(trainer, pl_module, None, None, 0)

        callback.assert_not_called()

    def test_on_train_batch_end_empty_callback_metrics(self):
        """Test on_train_batch_end handles None callback_metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = None
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_batch_end(trainer, pl_module, None, None, 0)

        callback.assert_not_called()

    def test_on_train_epoch_end_emits_event_with_metrics(self):
        """Test on_train_epoch_end emits event with metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 2
        trainer.global_step = 200
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.4),
            "train/accuracy": torch.tensor(0.9),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_end(trainer, pl_module)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "train_epoch_end"
        assert call_args[1]["epoch"] == 2
        assert call_args[1]["global_step"] == 200
        assert call_args[1]["metrics"]["train/loss"] == pytest.approx(0.4)
        assert call_args[1]["metrics"]["train/accuracy"] == pytest.approx(0.9)

    def test_on_train_epoch_end_no_metrics_no_emit(self):
        """Test on_train_epoch_end doesn't emit if no train metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_end(trainer, pl_module)

        callback.assert_not_called()

    def test_on_validation_epoch_start_emits_event(self):
        """Test on_validation_epoch_start emits correct event."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 7
        trainer.global_step = 300
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_epoch_start(trainer, pl_module)

        callback.assert_called_once_with(
            "validation_epoch_start",
            {"epoch": 7, "global_step": 300},
        )

    def test_on_validation_batch_end_emits_event_with_metrics(self):
        """Test on_validation_batch_end emits event with metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 1
        trainer.global_step = 25
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "val/accuracy": torch.tensor(0.92),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_batch_end(trainer, pl_module, None, None, 5)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "validation_batch"
        assert call_args[1]["epoch"] == 1
        assert call_args[1]["batch_idx"] == 5
        assert call_args[1]["global_step"] == 25
        assert call_args[1]["metrics"]["val/loss"] == pytest.approx(0.3)
        assert call_args[1]["metrics"]["val/accuracy"] == pytest.approx(0.92)

    def test_on_validation_batch_end_skips_non_val_metrics(self):
        """Test on_validation_batch_end only includes val/ metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "train/loss": torch.tensor(0.5),
            "other_metric": torch.tensor(0.7),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_batch_end(trainer, pl_module, None, None, 0)

        call_args = callback.call_args[0]
        assert call_args[0] == "validation_batch"
        assert "val/loss" in call_args[1]["metrics"]
        assert "train/loss" not in call_args[1]["metrics"]
        assert "other_metric" not in call_args[1]["metrics"]

    def test_on_validation_batch_end_no_metrics_no_emit(self):
        """Test on_validation_batch_end doesn't emit if no val metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_batch_end(trainer, pl_module, None, None, 0)

        callback.assert_not_called()

    def test_on_validation_epoch_end_emits_event_with_metrics(self):
        """Test on_validation_epoch_end emits event with metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 4
        trainer.global_step = 150
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.25),
            "val/accuracy": torch.tensor(0.95),
        }
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_epoch_end(trainer, pl_module)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "validation_epoch_end"
        assert call_args[1]["epoch"] == 4
        assert call_args[1]["global_step"] == 150
        assert call_args[1]["metrics"]["val/loss"] == pytest.approx(0.25)
        assert call_args[1]["metrics"]["val/accuracy"] == pytest.approx(0.95)

    def test_on_validation_epoch_end_no_metrics_no_emit(self):
        """Test on_validation_epoch_end doesn't emit if no val metrics."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_epoch_end(trainer, pl_module)

        callback.assert_not_called()

    def test_complete_training_workflow(self):
        """Test complete training workflow emits all events."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 0
        pl_module = MagicMock(spec=L.LightningModule)

        # Training epoch start
        streamer.on_train_epoch_start(trainer, pl_module)

        # Training batch
        trainer.callback_metrics = {"train/loss": torch.tensor(0.5)}
        streamer.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Training epoch end
        streamer.on_train_epoch_end(trainer, pl_module)

        # Validation epoch start
        streamer.on_validation_epoch_start(trainer, pl_module)

        # Validation batch
        trainer.callback_metrics = {"val/loss": torch.tensor(0.3)}
        streamer.on_validation_batch_end(trainer, pl_module, None, None, 0)

        # Validation epoch end
        streamer.on_validation_epoch_end(trainer, pl_module)

        assert callback.call_count == 6
        event_types = [c[0][0] for c in callback.call_args_list]
        assert "train_epoch_start" in event_types
        assert "train_batch" in event_types
        assert "train_epoch_end" in event_types
        assert "validation_epoch_start" in event_types
        assert "validation_batch" in event_types
        assert "validation_epoch_end" in event_types

    def test_multiple_callbacks_receive_same_data(self):
        """Test multiple callbacks receive identical data."""
        streamer = MetricsStreamer()
        callback1 = MagicMock()
        callback2 = MagicMock()
        streamer.register_callback(callback1)
        streamer.register_callback(callback2)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 1
        trainer.global_step = 10
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_start(trainer, pl_module)

        callback1.assert_called_once()
        callback2.assert_called_once()
        assert callback1.call_args == callback2.call_args

    def test_callback_can_be_lambda(self):
        """Test that lambda functions can be used as callbacks."""
        streamer = MetricsStreamer()
        events = []
        streamer.register_callback(lambda event, data: events.append((event, data)))
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 0
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_start(trainer, pl_module)

        assert len(events) == 1
        assert events[0][0] == "train_epoch_start"
        assert events[0][1] == {"epoch": 0, "global_step": 0}

    def test_on_validation_batch_end_with_dataloader_idx(self):
        """Test on_validation_batch_end accepts dataloader_idx parameter."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.callback_metrics = {"val/loss": torch.tensor(0.3)}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_batch_end(
            trainer, pl_module, None, None, 0, _dataloader_idx=1
        )

        callback.assert_called_once()
        assert callback.call_args[0][0] == "validation_batch"

    def test_on_train_batch_end_only_non_train_metrics(self):
        """Test on_train_batch_end with only non-train metrics doesn't emit."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {"val/loss": torch.tensor(0.3)}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_batch_end(trainer, pl_module, None, None, 0)

        callback.assert_not_called()

    def test_on_train_epoch_end_only_non_train_metrics(self):
        """Test on_train_epoch_end with only non-train metrics doesn't emit."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {"val/loss": torch.tensor(0.3)}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_train_epoch_end(trainer, pl_module)

        callback.assert_not_called()

    def test_on_validation_batch_end_only_non_val_metrics(self):
        """Test on_validation_batch_end with only non-val metrics doesn't emit."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {"train/loss": torch.tensor(0.5)}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_batch_end(trainer, pl_module, None, None, 0)

        callback.assert_not_called()

    def test_on_validation_epoch_end_only_non_val_metrics(self):
        """Test on_validation_epoch_end with only non-val metrics doesn't emit."""
        streamer = MetricsStreamer()
        callback = MagicMock()
        streamer.register_callback(callback)
        trainer = MagicMock(spec=L.Trainer)
        trainer.callback_metrics = {"train/loss": torch.tensor(0.5)}
        pl_module = MagicMock(spec=L.LightningModule)

        streamer.on_validation_epoch_end(trainer, pl_module)

        callback.assert_not_called()
