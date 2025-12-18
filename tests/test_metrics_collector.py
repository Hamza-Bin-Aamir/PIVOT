"""Tests for epoch metrics collector."""

from __future__ import annotations

import pytest

from src.train.metrics_collector import EpochMetricsCollector


class TestEpochMetricsCollector:
    """Test suite for EpochMetricsCollector."""

    def test_init_empty(self):
        """Test initialization creates empty collector."""
        collector = EpochMetricsCollector()

        assert len(collector) == 0
        assert collector.get_metric_names() == []
        assert repr(collector) == "EpochMetricsCollector(empty)"

    def test_add_single_metric(self):
        """Test adding a single metric value."""
        collector = EpochMetricsCollector()

        collector.add("loss", 0.5)

        assert collector.has_metric("loss")
        assert collector.get_count("loss") == 1
        assert collector.get_mean("loss") == 0.5
        assert collector.get_latest("loss") == 0.5
        assert "loss" in collector

    def test_add_multiple_values_same_metric(self):
        """Test adding multiple values to the same metric."""
        collector = EpochMetricsCollector()

        collector.add("loss", 0.5)
        collector.add("loss", 0.3)
        collector.add("loss", 0.7)

        assert collector.get_count("loss") == 3
        assert collector.get_mean("loss") == 0.5
        assert collector.get_latest("loss") == 0.7

    def test_add_multiple_different_metrics(self):
        """Test adding values to different metrics."""
        collector = EpochMetricsCollector()

        collector.add("loss", 0.5)
        collector.add("accuracy", 0.85)
        collector.add("f1_score", 0.78)

        assert len(collector) == 3
        assert collector.has_metric("loss")
        assert collector.has_metric("accuracy")
        assert collector.has_metric("f1_score")
        assert set(collector.get_metric_names()) == {"loss", "accuracy", "f1_score"}

    def test_add_empty_name_raises_error(self):
        """Test that empty metric name raises ValueError."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            collector.add("", 0.5)

    def test_add_whitespace_name_raises_error(self):
        """Test that whitespace-only metric name raises ValueError."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            collector.add("   ", 0.5)

    def test_add_nan_raises_error(self):
        """Test that NaN value raises ValueError."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric value must be finite"):
            collector.add("loss", float("nan"))

    def test_add_positive_infinity_raises_error(self):
        """Test that positive infinity raises ValueError."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric value must be finite"):
            collector.add("loss", float("inf"))

    def test_add_negative_infinity_raises_error(self):
        """Test that negative infinity raises ValueError."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric value must be finite"):
            collector.add("loss", float("-inf"))

    def test_add_batch_single(self):
        """Test adding a batch of metrics."""
        collector = EpochMetricsCollector()

        collector.add_batch({"loss": 0.5, "accuracy": 0.85})

        assert collector.has_metric("loss")
        assert collector.has_metric("accuracy")
        assert collector.get_mean("loss") == 0.5
        assert collector.get_mean("accuracy") == 0.85

    def test_add_batch_multiple_times(self):
        """Test adding batches multiple times."""
        collector = EpochMetricsCollector()

        collector.add_batch({"loss": 0.5, "accuracy": 0.8})
        collector.add_batch({"loss": 0.3, "accuracy": 0.9})

        assert collector.get_count("loss") == 2
        assert collector.get_count("accuracy") == 2
        assert collector.get_mean("loss") == pytest.approx(0.4)
        assert collector.get_mean("accuracy") == pytest.approx(0.85)

    def test_add_batch_empty(self):
        """Test adding empty batch does nothing."""
        collector = EpochMetricsCollector()

        collector.add_batch({})

        assert len(collector) == 0

    def test_add_batch_with_invalid_value(self):
        """Test that add_batch validates all values."""
        collector = EpochMetricsCollector()

        with pytest.raises(ValueError, match="Metric value must be finite"):
            collector.add_batch({"loss": 0.5, "invalid": float("nan")})

    def test_compute_single_metric(self):
        """Test computing statistics for a single metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.3)
        collector.add("loss", 0.5)
        collector.add("loss", 0.7)

        stats = collector.compute()

        assert "loss" in stats
        assert stats["loss"]["mean"] == 0.5
        assert abs(stats["loss"]["std"] - 0.16329931618554518) < 1e-10
        assert stats["loss"]["min"] == 0.3
        assert stats["loss"]["max"] == 0.7
        assert stats["loss"]["count"] == 3

    def test_compute_multiple_metrics(self):
        """Test computing statistics for multiple metrics."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("loss", 0.3)
        collector.add("accuracy", 0.8)
        collector.add("accuracy", 0.9)

        stats = collector.compute()

        assert len(stats) == 2
        assert "loss" in stats
        assert "accuracy" in stats
        assert stats["loss"]["mean"] == pytest.approx(0.4)
        assert stats["accuracy"]["mean"] == pytest.approx(0.85)

    def test_compute_empty_collector(self):
        """Test computing on empty collector returns empty dict."""
        collector = EpochMetricsCollector()

        stats = collector.compute()

        assert stats == {}

    def test_compute_after_reset_with_empty_metric(self):
        """Test that compute skips metrics with no values."""
        collector = EpochMetricsCollector()
        # Manually create an empty metric list (edge case)
        collector._metrics["empty"] = []
        collector.add("valid", 0.5)

        stats = collector.compute()

        # Should only include "valid", not "empty"
        assert "valid" in stats
        assert "empty" not in stats
        assert len(stats) == 1

    def test_compute_single_value(self):
        """Test computing statistics for a single value."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)

        stats = collector.compute()

        assert stats["loss"]["mean"] == 0.5
        assert stats["loss"]["std"] == 0.0
        assert stats["loss"]["min"] == 0.5
        assert stats["loss"]["max"] == 0.5
        assert stats["loss"]["count"] == 1

    def test_reset_clears_all_metrics(self):
        """Test reset clears all collected metrics."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("accuracy", 0.8)

        collector.reset()

        assert len(collector) == 0
        assert not collector.has_metric("loss")
        assert not collector.has_metric("accuracy")
        assert collector.compute() == {}

    def test_reset_empty_collector(self):
        """Test reset on empty collector does nothing."""
        collector = EpochMetricsCollector()

        collector.reset()

        assert len(collector) == 0

    def test_get_count_existing_metric(self):
        """Test get_count for existing metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("loss", 0.3)

        assert collector.get_count("loss") == 2

    def test_get_count_nonexistent_metric(self):
        """Test get_count for nonexistent metric returns 0."""
        collector = EpochMetricsCollector()

        assert collector.get_count("nonexistent") == 0

    def test_get_mean_existing_metric(self):
        """Test get_mean for existing metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.3)
        collector.add("loss", 0.5)
        collector.add("loss", 0.7)

        assert collector.get_mean("loss") == 0.5

    def test_get_mean_nonexistent_metric(self):
        """Test get_mean for nonexistent metric returns None."""
        collector = EpochMetricsCollector()

        assert collector.get_mean("nonexistent") is None

    def test_get_mean_empty_metric(self):
        """Test get_mean after reset returns None."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.reset()

        assert collector.get_mean("loss") is None

    def test_has_metric_true(self):
        """Test has_metric returns True for existing metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)

        assert collector.has_metric("loss")

    def test_has_metric_false(self):
        """Test has_metric returns False for nonexistent metric."""
        collector = EpochMetricsCollector()

        assert not collector.has_metric("loss")

    def test_get_metric_names_multiple(self):
        """Test get_metric_names returns all metric names."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("accuracy", 0.8)
        collector.add("f1_score", 0.75)

        names = collector.get_metric_names()

        assert set(names) == {"loss", "accuracy", "f1_score"}

    def test_get_metric_names_empty(self):
        """Test get_metric_names returns empty list for empty collector."""
        collector = EpochMetricsCollector()

        assert collector.get_metric_names() == []

    def test_get_latest_existing_metric(self):
        """Test get_latest returns most recent value."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("loss", 0.3)
        collector.add("loss", 0.7)

        assert collector.get_latest("loss") == 0.7

    def test_get_latest_nonexistent_metric(self):
        """Test get_latest returns None for nonexistent metric."""
        collector = EpochMetricsCollector()

        assert collector.get_latest("nonexistent") is None

    def test_get_latest_single_value(self):
        """Test get_latest with single value."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)

        assert collector.get_latest("loss") == 0.5

    def test_len_empty(self):
        """Test __len__ returns 0 for empty collector."""
        collector = EpochMetricsCollector()

        assert len(collector) == 0

    def test_len_multiple_metrics(self):
        """Test __len__ returns number of metrics."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("accuracy", 0.8)
        collector.add("f1_score", 0.75)

        assert len(collector) == 3

    def test_contains_true(self):
        """Test __contains__ returns True for existing metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)

        assert "loss" in collector

    def test_contains_false(self):
        """Test __contains__ returns False for nonexistent metric."""
        collector = EpochMetricsCollector()

        assert "loss" not in collector

    def test_repr_empty(self):
        """Test __repr__ for empty collector."""
        collector = EpochMetricsCollector()

        assert repr(collector) == "EpochMetricsCollector(empty)"

    def test_repr_single_metric(self):
        """Test __repr__ for single metric."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("loss", 0.3)

        assert repr(collector) == "EpochMetricsCollector(loss:2)"

    def test_repr_multiple_metrics(self):
        """Test __repr__ for multiple metrics."""
        collector = EpochMetricsCollector()
        collector.add("loss", 0.5)
        collector.add("accuracy", 0.8)

        repr_str = repr(collector)
        assert "EpochMetricsCollector(" in repr_str
        assert "loss:1" in repr_str
        assert "accuracy:1" in repr_str

    def test_is_finite_nan(self):
        """Test _is_finite returns False for NaN."""
        assert not EpochMetricsCollector._is_finite(float("nan"))

    def test_is_finite_positive_inf(self):
        """Test _is_finite returns False for positive infinity."""
        assert not EpochMetricsCollector._is_finite(float("inf"))

    def test_is_finite_negative_inf(self):
        """Test _is_finite returns False for negative infinity."""
        assert not EpochMetricsCollector._is_finite(float("-inf"))

    def test_is_finite_normal_values(self):
        """Test _is_finite returns True for normal values."""
        assert EpochMetricsCollector._is_finite(0.0)
        assert EpochMetricsCollector._is_finite(1.5)
        assert EpochMetricsCollector._is_finite(-10.7)
        assert EpochMetricsCollector._is_finite(1e10)
        assert EpochMetricsCollector._is_finite(-1e-10)

    def test_workflow_complete_epoch(self):
        """Test complete workflow for an epoch."""
        collector = EpochMetricsCollector()

        # Simulate batch updates
        for i in range(5):
            collector.add_batch(
                {
                    "loss": 0.5 - i * 0.05,
                    "accuracy": 0.7 + i * 0.05,
                }
            )

        # Check intermediate state
        assert collector.get_count("loss") == 5
        assert collector.get_count("accuracy") == 5

        # Compute final statistics
        stats = collector.compute()

        assert stats["loss"]["mean"] == pytest.approx(0.4, rel=1e-6)
        assert stats["accuracy"]["mean"] == pytest.approx(0.8, rel=1e-6)
        assert stats["loss"]["count"] == 5
        assert stats["accuracy"]["count"] == 5

        # Reset for next epoch
        collector.reset()
        assert len(collector) == 0

    def test_negative_values(self):
        """Test that negative values work correctly."""
        collector = EpochMetricsCollector()
        collector.add("metric", -0.5)
        collector.add("metric", -0.3)
        collector.add("metric", -0.7)

        stats = collector.compute()

        assert stats["metric"]["mean"] == -0.5
        assert stats["metric"]["min"] == -0.7
        assert stats["metric"]["max"] == -0.3

    def test_zero_values(self):
        """Test that zero values work correctly."""
        collector = EpochMetricsCollector()
        collector.add("metric", 0.0)
        collector.add("metric", 0.0)

        stats = collector.compute()

        assert stats["metric"]["mean"] == 0.0
        assert stats["metric"]["std"] == 0.0
        assert stats["metric"]["min"] == 0.0
        assert stats["metric"]["max"] == 0.0

    def test_large_values(self):
        """Test that large values work correctly."""
        collector = EpochMetricsCollector()
        collector.add("metric", 1e10)
        collector.add("metric", 2e10)

        stats = collector.compute()

        assert stats["metric"]["mean"] == 1.5e10
        assert stats["metric"]["min"] == 1e10
        assert stats["metric"]["max"] == 2e10

    def test_small_values(self):
        """Test that small values work correctly."""
        collector = EpochMetricsCollector()
        collector.add("metric", 1e-10)
        collector.add("metric", 2e-10)

        stats = collector.compute()

        assert stats["metric"]["mean"] == pytest.approx(1.5e-10, rel=1e-6)
        assert stats["metric"]["min"] == 1e-10
        assert stats["metric"]["max"] == 2e-10
