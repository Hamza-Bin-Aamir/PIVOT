"""Epoch metrics collector for aggregating training and validation metrics.

This module provides a collector that accumulates metrics during an epoch
and computes aggregated statistics (mean, std, min, max) at the end.
"""

from __future__ import annotations


class EpochMetricsCollector:
    """Collects and aggregates metrics during a training or validation epoch.

    This class maintains running totals of metrics and computes summary statistics
    at the end of an epoch. It supports multiple metrics simultaneously.

    Attributes:
        metrics: Dictionary mapping metric names to lists of values

    Example:
        >>> collector = EpochMetricsCollector()
        >>> collector.add("loss", 0.5)
        >>> collector.add("loss", 0.3)
        >>> collector.add("accuracy", 0.85)
        >>> stats = collector.compute()
        >>> stats["loss"]["mean"]
        0.4
        >>> collector.reset()
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._metrics: dict[str, list[float]] = {}

    def add(self, name: str, value: float) -> None:
        """Add a metric value to the collector.

        Args:
            name: Name of the metric (e.g., 'loss', 'accuracy')
            value: Metric value to add

        Raises:
            ValueError: If name is empty or value is not finite
        """
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")

        # Check for NaN and infinity
        if not self._is_finite(value):
            raise ValueError(f"Metric value must be finite, got {value}")

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(value)

    def add_batch(self, metrics: dict[str, float]) -> None:
        """Add multiple metrics at once.

        Args:
            metrics: Dictionary mapping metric names to values

        Raises:
            ValueError: If any metric name is empty or value is not finite
        """
        for name, value in metrics.items():
            self.add(name, value)

    def compute(self) -> dict[str, dict[str, float]]:
        """Compute aggregated statistics for all collected metrics.

        Returns:
            Dictionary mapping metric names to statistics dictionaries.
            Each statistics dictionary contains:
                - 'mean': Average value
                - 'std': Standard deviation
                - 'min': Minimum value
                - 'max': Maximum value
                - 'count': Number of samples

        Example:
            >>> collector = EpochMetricsCollector()
            >>> collector.add("loss", 0.5)
            >>> collector.add("loss", 0.3)
            >>> stats = collector.compute()
            >>> stats["loss"]
            {'mean': 0.4, 'std': 0.1, 'min': 0.3, 'max': 0.5, 'count': 2}
        """
        result: dict[str, dict[str, float]] = {}

        for name, values in self._metrics.items():
            if not values:
                continue

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance**0.5

            result[name] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "count": float(len(values)),
            }

        return result

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()

    def get_count(self, name: str) -> int:
        """Get the number of samples for a specific metric.

        Args:
            name: Name of the metric

        Returns:
            Number of samples collected for the metric, or 0 if metric not found
        """
        return len(self._metrics.get(name, []))

    def get_mean(self, name: str) -> float | None:
        """Get the current mean value for a specific metric.

        Args:
            name: Name of the metric

        Returns:
            Mean value, or None if metric has no samples
        """
        values = self._metrics.get(name, [])
        if not values:
            return None
        return sum(values) / len(values)

    def has_metric(self, name: str) -> bool:
        """Check if a metric has been collected.

        Args:
            name: Name of the metric

        Returns:
            True if the metric has at least one sample, False otherwise
        """
        return name in self._metrics and len(self._metrics[name]) > 0

    def get_metric_names(self) -> list[str]:
        """Get all metric names with collected samples.

        Returns:
            List of metric names that have at least one sample
        """
        return [name for name in self._metrics if self._metrics[name]]

    def get_latest(self, name: str) -> float | None:
        """Get the most recently added value for a metric.

        Args:
            name: Name of the metric

        Returns:
            Most recent value, or None if metric has no samples
        """
        values = self._metrics.get(name, [])
        if not values:
            return None
        return values[-1]

    @staticmethod
    def _is_finite(value: float) -> bool:
        """Check if a value is finite (not NaN or infinity).

        Args:
            value: Value to check

        Returns:
            True if value is finite, False otherwise
        """
        # Check for NaN
        if value != value:  # NaN != NaN
            return False

        # Check for infinity
        return not (value == float("inf") or value == float("-inf"))

    def __len__(self) -> int:
        """Get the total number of metric names with samples.

        Returns:
            Number of unique metrics with at least one sample
        """
        return len(self.get_metric_names())

    def __contains__(self, name: str) -> bool:
        """Check if a metric name exists in the collector.

        Args:
            name: Name of the metric

        Returns:
            True if the metric has samples, False otherwise
        """
        return self.has_metric(name)

    def __repr__(self) -> str:
        """Get string representation of the collector.

        Returns:
            String showing metric names and sample counts
        """
        if not self._metrics:
            return "EpochMetricsCollector(empty)"

        metric_info = [f"{name}:{len(values)}" for name, values in self._metrics.items()]
        return f"EpochMetricsCollector({', '.join(metric_info)})"
