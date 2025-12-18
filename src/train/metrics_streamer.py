"""Real-time metrics streaming for training monitoring.

This module provides a callback-based system for streaming training metrics
in real-time to external consumers such as APIs, dashboards, or monitoring systems.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import lightning as L
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import torch


class MetricsStreamer(Callback):
    """Streams training metrics in real-time via callbacks.

    This class extends PyTorch Lightning's Callback to emit training metrics
    to registered callback functions. It enables real-time monitoring by allowing
    external systems to subscribe to metric updates.

    Attributes:
        callbacks: List of callback functions to invoke when metrics are emitted

    Example:
        >>> def print_metrics(event_type: str, data: dict):
        ...     print(f"{event_type}: {data}")
        >>> streamer = MetricsStreamer()
        >>> streamer.register_callback(print_metrics)
        >>> trainer = L.Trainer(callbacks=[streamer])
        >>> # During training, print_metrics will be called with updates
    """

    def __init__(self) -> None:
        """Initialize the metrics streamer."""
        super().__init__()
        self._callbacks: list[Callable[[str, dict[str, Any]], None]] = []

    def register_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Register a callback function to receive metric events.

        Args:
            callback: Function that takes (event_type, data) where:
                - event_type: Type of event (e.g., 'train_batch', 'train_epoch')
                - data: Dictionary containing event data and metrics

        Raises:
            TypeError: If callback is not callable
        """
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)}")
        self._callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Unregister a previously registered callback.

        Args:
            callback: The callback function to remove

        Raises:
            ValueError: If callback is not registered
        """
        if callback not in self._callbacks:
            raise ValueError("Callback not registered")
        self._callbacks.remove(callback)

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        self._callbacks.clear()

    def get_callback_count(self) -> int:
        """Get the number of registered callbacks.

        Returns:
            Number of registered callbacks
        """
        return len(self._callbacks)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event_type: Type of event being emitted
            data: Event data to pass to callbacks
        """
        for callback in self._callbacks:
            with contextlib.suppress(Exception):
                # Silently ignore callback errors to prevent training interruption
                callback(event_type, data)

    def on_train_epoch_start(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when training epoch starts.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
        """
        self._emit(
            "train_epoch_start",
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
            },
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _outputs: torch.Tensor | Mapping[str, Any] | None,
        _batch: Any,  # noqa: ANN401
        _batch_idx: int,
    ) -> None:
        """Called when training batch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
            _outputs: Training step outputs
            _batch: Current batch
            _batch_idx: Batch index
        """
        if trainer.callback_metrics:
            metrics = {
                key: float(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("train/")
            }
            if metrics:
                self._emit(
                    "train_batch",
                    {
                        "epoch": trainer.current_epoch,
                        "batch_idx": _batch_idx,
                        "global_step": trainer.global_step,
                        "metrics": metrics,
                    },
                )

    def on_train_epoch_end(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when training epoch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
        """
        if trainer.callback_metrics:
            metrics = {
                key: float(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("train/")
            }
            if metrics:
                self._emit(
                    "train_epoch_end",
                    {
                        "epoch": trainer.current_epoch,
                        "global_step": trainer.global_step,
                        "metrics": metrics,
                    },
                )

    def on_validation_epoch_start(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when validation epoch starts.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
        """
        self._emit(
            "validation_epoch_start",
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
            },
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _outputs: torch.Tensor | Mapping[str, Any] | None,
        _batch: Any,  # noqa: ANN401
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Called when validation batch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
            _outputs: Validation step outputs
            _batch: Current batch
            _batch_idx: Batch index
            _dataloader_idx: Dataloader index
        """
        if trainer.callback_metrics:
            metrics = {
                key: float(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("val/")
            }
            if metrics:
                self._emit(
                    "validation_batch",
                    {
                        "epoch": trainer.current_epoch,
                        "batch_idx": _batch_idx,
                        "global_step": trainer.global_step,
                        "metrics": metrics,
                    },
                )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, _pl_module: L.LightningModule
    ) -> None:
        """Called when validation epoch ends.

        Args:
            trainer: PyTorch Lightning trainer
            _pl_module: PyTorch Lightning module
        """
        if trainer.callback_metrics:
            metrics = {
                key: float(value)
                for key, value in trainer.callback_metrics.items()
                if key.startswith("val/")
            }
            if metrics:
                self._emit(
                    "validation_epoch_end",
                    {
                        "epoch": trainer.current_epoch,
                        "global_step": trainer.global_step,
                        "metrics": metrics,
                    },
                )
