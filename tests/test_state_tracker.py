"""Tests for training state tracker."""

from __future__ import annotations

import pytest

from src.train.state_tracker import TrainingStateTracker


class TestTrainingStateTracker:
    """Test suite for TrainingStateTracker."""

    def test_init_default_state(self):
        """Test initialization with default state."""
        tracker = TrainingStateTracker()

        assert tracker.phase == "idle"
        assert tracker.epoch == 0
        assert tracker.step_count == 0
        assert tracker.is_idle()
        assert not tracker.is_training()
        assert not tracker.is_validating()
        assert not tracker.is_testing()

    def test_start_epoch_train(self):
        """Test starting a training epoch."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(5, "train")

        assert tracker.phase == "train"
        assert tracker.epoch == 5
        assert tracker.step_count == 0
        assert tracker.is_training()
        assert not tracker.is_idle()

    def test_start_epoch_validation(self):
        """Test starting a validation epoch."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(3, "validation")

        assert tracker.phase == "validation"
        assert tracker.epoch == 3
        assert tracker.step_count == 0
        assert tracker.is_validating()
        assert not tracker.is_training()

    def test_start_epoch_test(self):
        """Test starting a test epoch."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(10, "test")

        assert tracker.phase == "test"
        assert tracker.epoch == 10
        assert tracker.step_count == 0
        assert tracker.is_testing()
        assert not tracker.is_idle()

    def test_start_epoch_idle(self):
        """Test starting an idle epoch."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "idle")

        assert tracker.phase == "idle"
        assert tracker.epoch == 0
        assert tracker.step_count == 0
        assert tracker.is_idle()

    def test_start_epoch_resets_step_count(self):
        """Test that starting a new epoch resets step count."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "train")
        tracker.increment_step()
        tracker.increment_step()
        assert tracker.step_count == 2

        tracker.start_epoch(1, "train")
        assert tracker.step_count == 0

    def test_start_epoch_negative_raises_error(self):
        """Test that negative epoch raises ValueError."""
        tracker = TrainingStateTracker()

        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            tracker.start_epoch(-1, "train")

    def test_start_epoch_invalid_phase_raises_error(self):
        """Test that invalid phase raises ValueError."""
        tracker = TrainingStateTracker()

        with pytest.raises(ValueError, match="Invalid phase"):
            tracker.start_epoch(0, "invalid")  # type: ignore[arg-type]

    def test_increment_step_single(self):
        """Test incrementing step count once."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "train")

        tracker.increment_step()
        assert tracker.step_count == 1

    def test_increment_step_multiple(self):
        """Test incrementing step count multiple times."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "train")

        for i in range(10):
            tracker.increment_step()
            assert tracker.step_count == i + 1

    def test_increment_step_across_phases(self):
        """Test that step count increments independently in each phase."""
        tracker = TrainingStateTracker()

        tracker.start_epoch(0, "train")
        tracker.increment_step()
        tracker.increment_step()
        assert tracker.step_count == 2

        tracker.start_epoch(0, "validation")
        tracker.increment_step()
        assert tracker.step_count == 1

    def test_end_epoch_sets_idle(self):
        """Test that ending epoch sets phase to idle."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(5, "train")
        tracker.increment_step()

        tracker.end_epoch()

        assert tracker.phase == "idle"
        assert tracker.epoch == 5  # Epoch number should remain
        assert tracker.step_count == 0

    def test_end_epoch_resets_step(self):
        """Test that ending epoch resets step count."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "train")
        tracker.increment_step()
        tracker.increment_step()

        tracker.end_epoch()

        assert tracker.step_count == 0

    def test_reset_full(self):
        """Test full reset of tracker state."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(10, "train")
        tracker.increment_step()
        tracker.increment_step()

        tracker.reset()

        assert tracker.phase == "idle"
        assert tracker.epoch == 0
        assert tracker.step_count == 0
        assert tracker.is_idle()

    def test_is_training_true(self):
        """Test is_training returns True in training phase."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "train")

        assert tracker.is_training()

    def test_is_training_false(self):
        """Test is_training returns False in non-training phases."""
        tracker = TrainingStateTracker()

        tracker.start_epoch(0, "validation")
        assert not tracker.is_training()

        tracker.start_epoch(0, "test")
        assert not tracker.is_training()

        tracker.start_epoch(0, "idle")
        assert not tracker.is_training()

    def test_is_validating_true(self):
        """Test is_validating returns True in validation phase."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "validation")

        assert tracker.is_validating()

    def test_is_validating_false(self):
        """Test is_validating returns False in non-validation phases."""
        tracker = TrainingStateTracker()

        tracker.start_epoch(0, "train")
        assert not tracker.is_validating()

        tracker.start_epoch(0, "test")
        assert not tracker.is_validating()

        tracker.start_epoch(0, "idle")
        assert not tracker.is_validating()

    def test_is_testing_true(self):
        """Test is_testing returns True in test phase."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(0, "test")

        assert tracker.is_testing()

    def test_is_testing_false(self):
        """Test is_testing returns False in non-test phases."""
        tracker = TrainingStateTracker()

        tracker.start_epoch(0, "train")
        assert not tracker.is_testing()

        tracker.start_epoch(0, "validation")
        assert not tracker.is_testing()

        tracker.start_epoch(0, "idle")
        assert not tracker.is_testing()

    def test_is_idle_true(self):
        """Test is_idle returns True in idle phase."""
        tracker = TrainingStateTracker()

        assert tracker.is_idle()

        tracker.start_epoch(0, "idle")
        assert tracker.is_idle()

    def test_is_idle_false(self):
        """Test is_idle returns False in non-idle phases."""
        tracker = TrainingStateTracker()

        tracker.start_epoch(0, "train")
        assert not tracker.is_idle()

        tracker.start_epoch(0, "validation")
        assert not tracker.is_idle()

        tracker.start_epoch(0, "test")
        assert not tracker.is_idle()

    def test_get_state_initial(self):
        """Test get_state returns correct initial state."""
        tracker = TrainingStateTracker()

        state = tracker.get_state()

        assert state == {
            "phase": "idle",
            "epoch": 0,
            "step": 0,
        }

    def test_get_state_after_updates(self):
        """Test get_state returns correct state after updates."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(5, "train")
        tracker.increment_step()
        tracker.increment_step()

        state = tracker.get_state()

        assert state == {
            "phase": "train",
            "epoch": 5,
            "step": 2,
        }

    def test_get_state_returns_copy(self):
        """Test that get_state returns independent dictionary."""
        tracker = TrainingStateTracker()
        tracker.start_epoch(1, "train")

        state1 = tracker.get_state()
        state2 = tracker.get_state()

        assert state1 == state2
        assert state1 is not state2  # Different objects

    def test_epoch_property_readonly(self):
        """Test that epoch property cannot be set directly."""
        tracker = TrainingStateTracker()

        with pytest.raises(AttributeError):
            tracker.epoch = 10  # type: ignore[misc]

    def test_phase_property_readonly(self):
        """Test that phase property cannot be set directly."""
        tracker = TrainingStateTracker()

        with pytest.raises(AttributeError):
            tracker.phase = "train"  # type: ignore[misc]

    def test_step_count_property_readonly(self):
        """Test that step_count property cannot be set directly."""
        tracker = TrainingStateTracker()

        with pytest.raises(AttributeError):
            tracker.step_count = 5  # type: ignore[misc]

    def test_complete_training_workflow(self):
        """Test a complete training workflow with multiple epochs."""
        tracker = TrainingStateTracker()

        # Epoch 0 - Training
        tracker.start_epoch(0, "train")
        assert tracker.epoch == 0
        assert tracker.phase == "train"
        for _ in range(5):
            tracker.increment_step()
        assert tracker.step_count == 5

        # Epoch 0 - Validation
        tracker.start_epoch(0, "validation")
        assert tracker.epoch == 0
        assert tracker.phase == "validation"
        assert tracker.step_count == 0
        tracker.increment_step()
        tracker.increment_step()
        assert tracker.step_count == 2
        tracker.end_epoch()

        # Epoch 1 - Training
        tracker.start_epoch(1, "train")
        assert tracker.epoch == 1
        assert tracker.phase == "train"
        assert tracker.step_count == 0
        tracker.increment_step()
        assert tracker.step_count == 1

        # Reset
        tracker.reset()
        assert tracker.phase == "idle"
        assert tracker.epoch == 0
        assert tracker.step_count == 0
