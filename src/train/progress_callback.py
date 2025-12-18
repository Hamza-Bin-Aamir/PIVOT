"""Training progress callback for PyTorch Lightning.

This module provides a callback that tracks training progress including
state transitions, metric collection, and progress updates during training.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from .metrics_collector import EpochMetricsCollector
from .state_tracker import TrainingStateTracker


class TrainingProgressCallback(Callback):
    """PyTorch Lightning callback for tracking training progress.

    This callback integrates state tracking and metrics collection to monitor
    training progress. It hooks into Lightning's training lifecycle to:
    - Track current epoch, step, and phase
    - Collect and aggregate metrics per epoch
    - Provide progress introspection

    Attributes:
        state_tracker: Tracks current training state (epoch, step, phase)
        train_metrics: Collects training metrics for current epoch
        val_metrics: Collects validation metrics for current epoch

    Example:
        >>> callback = TrainingProgressCallback()
        >>> trainer = L.Trainer(callbacks=[callback])
        >>> # During training, access state and metrics
        >>> callback.state_tracker.epoch
        5
        >>> callback.train_metrics.get_mean("loss")
        0.35
    """

    def __init__(self) -> None:
        """Initialize the training progress callback."""
        super().__init__()
        self.state_tracker = TrainingStateTracker()
        self.train_metrics = EpochMetricsCollector()
        self.val_metrics = EpochMetricsCollector()

    def on_train_epoch_start(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when the train epoch begins.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being trained
        """
        self.state_tracker.start_epoch(trainer.current_epoch, "train")
        self.train_metrics.reset()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _outputs: torch.Tensor | Mapping[str, Any] | None,
        _batch: dict | None,
        _batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being trained
            _outputs: Outputs from training_step
            _batch: Current batch
            _batch_idx: Index of current batch
        """
        self.state_tracker.increment_step()

        # Collect metrics from trainer's logged metrics
        if trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if key.startswith("train/") and not key.endswith("_epoch"):
                    metric_name = key.replace("train/", "")
                    with contextlib.suppress(ValueError, TypeError):
                        self.train_metrics.add(metric_name, float(value))

    def on_validation_epoch_start(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when the validation epoch begins.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being validated
        """
        self.state_tracker.start_epoch(trainer.current_epoch, "validation")
        self.val_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _outputs: torch.Tensor | Mapping[str, Any] | None,
        _batch: dict | None,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being validated
            _outputs: Outputs from validation_step
            _batch: Current batch
            _batch_idx: Index of current batch
            _dataloader_idx: Index of current dataloader
        """
        self.state_tracker.increment_step()

        # Collect metrics from trainer's logged metrics
        if trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if key.startswith("val/") and not key.endswith("_epoch"):
                    metric_name = key.replace("val/", "")
                    with contextlib.suppress(ValueError, TypeError):
                        self.val_metrics.add(metric_name, float(value))

    def on_train_epoch_end(
        self, _trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when the train epoch ends.

        Args:
            _trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being trained
        """
        self.state_tracker.end_epoch()

    def on_validation_epoch_end(
        self, _trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when the validation epoch ends.

        Args:
            _trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module being validated
        """
        self.state_tracker.end_epoch()

    def get_train_stats(self) -> dict[str, dict[str, float]]:
        """Get aggregated training statistics for current epoch.

        Returns:
            Dictionary mapping metric names to statistics dictionaries
            with keys: mean, std, min, max, count
        """
        return self.train_metrics.compute()

    def get_val_stats(self) -> dict[str, dict[str, float]]:
        """Get aggregated validation statistics for current epoch.

        Returns:
            Dictionary mapping metric names to statistics dictionaries
            with keys: mean, std, min, max, count
        """
        return self.val_metrics.compute()

    def get_progress_summary(
        self,
    ) -> dict[str, dict[str, int | str] | dict[str, dict[str, float]]]:
        """Get comprehensive progress summary.

        Returns:
            Dictionary containing:
                - state: Current training state (phase, epoch, step)
                - train_stats: Training metrics statistics
                - val_stats: Validation metrics statistics
        """
        return {
            "state": self.state_tracker.get_state(),
            "train_stats": self.get_train_stats(),
            "val_stats": self.get_val_stats(),
        }
