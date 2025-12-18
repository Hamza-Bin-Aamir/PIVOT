"""Training state tracker for monitoring training progress.

This module provides a lightweight state tracker that monitors the current
training phase, epoch, and step count during model training.
"""

from __future__ import annotations

from typing import Literal

TrainingPhase = Literal["train", "validation", "test", "idle"]


class TrainingStateTracker:
    """Tracks the current state of training process.

    This class maintains the current training phase, epoch number, and step count.
    It provides methods to update and query the training state.

    Attributes:
        phase: Current training phase ('train', 'validation', 'test', or 'idle')
        epoch: Current epoch number (0-indexed)
        step: Current step within the epoch (0-indexed)

    Example:
        >>> tracker = TrainingStateTracker()
        >>> tracker.start_epoch(0, "train")
        >>> tracker.step_count
        0
        >>> tracker.increment_step()
        >>> tracker.step_count
        1
        >>> tracker.is_training()
        True
    """

    def __init__(self) -> None:
        """Initialize the training state tracker."""
        self._phase: TrainingPhase = "idle"
        self._epoch: int = 0
        self._step: int = 0

    @property
    def phase(self) -> TrainingPhase:
        """Get the current training phase.

        Returns:
            Current phase: 'train', 'validation', 'test', or 'idle'
        """
        return self._phase

    @property
    def epoch(self) -> int:
        """Get the current epoch number.

        Returns:
            Current epoch (0-indexed)
        """
        return self._epoch

    @property
    def step_count(self) -> int:
        """Get the current step count within the epoch.

        Returns:
            Current step (0-indexed)
        """
        return self._step

    def start_epoch(self, epoch: int, phase: TrainingPhase) -> None:
        """Start a new epoch with the specified phase.

        Args:
            epoch: Epoch number (0-indexed)
            phase: Training phase ('train', 'validation', 'test', or 'idle')

        Raises:
            ValueError: If epoch is negative or phase is invalid
        """
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")

        valid_phases: set[TrainingPhase] = {"train", "validation", "test", "idle"}
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of {valid_phases}")

        self._epoch = epoch
        self._phase = phase
        self._step = 0

    def increment_step(self) -> None:
        """Increment the step count by 1."""
        self._step += 1

    def end_epoch(self) -> None:
        """Mark the end of the current epoch.

        Resets the phase to 'idle' and step count to 0.
        """
        self._phase = "idle"
        self._step = 0

    def reset(self) -> None:
        """Reset the tracker to initial state.

        Sets phase to 'idle', epoch to 0, and step to 0.
        """
        self._phase = "idle"
        self._epoch = 0
        self._step = 0

    def is_training(self) -> bool:
        """Check if currently in training phase.

        Returns:
            True if phase is 'train', False otherwise
        """
        return self._phase == "train"

    def is_validating(self) -> bool:
        """Check if currently in validation phase.

        Returns:
            True if phase is 'validation', False otherwise
        """
        return self._phase == "validation"

    def is_testing(self) -> bool:
        """Check if currently in test phase.

        Returns:
            True if phase is 'test', False otherwise
        """
        return self._phase == "test"

    def is_idle(self) -> bool:
        """Check if currently idle (not training, validating, or testing).

        Returns:
            True if phase is 'idle', False otherwise
        """
        return self._phase == "idle"

    def get_state(self) -> dict[str, int | str]:
        """Get the current state as a dictionary.

        Returns:
            Dictionary containing 'phase', 'epoch', and 'step'
        """
        return {
            "phase": self._phase,
            "epoch": self._epoch,
            "step": self._step,
        }
