"""Metrics endpoint for retrieving training metrics.

This module provides API endpoints for accessing training and validation metrics,
including comprehensive metrics history and aggregated statistics.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..session_manager import TrainingSessionManager

router = APIRouter()

# Global session manager instance
_session_manager: TrainingSessionManager | None = None


def set_session_manager(manager: TrainingSessionManager) -> None:
    """Set the global session manager instance.

    Args:
        manager: TrainingSessionManager instance to use

    Raises:
        ValueError: If manager is None
    """
    global _session_manager  # noqa: PLW0603
    if manager is None:
        raise ValueError("Session manager cannot be None")
    _session_manager = manager


def get_session_manager() -> TrainingSessionManager:
    """Get the global session manager instance.

    Returns:
        TrainingSessionManager instance

    Raises:
        RuntimeError: If session manager not initialized
    """
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized")
    return _session_manager


class MetricPoint(BaseModel):
    """Model for a single metric data point.

    Attributes:
        epoch: Epoch number
        step: Training step number
        value: Metric value
        timestamp: ISO timestamp when metric was recorded
    """

    epoch: int
    step: int
    value: float
    timestamp: str


class MetricSeries(BaseModel):
    """Model for a metric time series.

    Attributes:
        name: Metric name (e.g., 'loss', 'accuracy')
        values: List of metric data points
        latest_value: Most recent metric value
        min_value: Minimum value in the series
        max_value: Maximum value in the series
        mean_value: Mean value across the series
    """

    name: str
    values: list[MetricPoint]
    latest_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        train_metrics: Training metrics by name
        val_metrics: Validation metrics by name
        total_metrics: Total number of unique metrics
    """

    session_id: str
    experiment_name: str
    train_metrics: dict[str, MetricSeries]
    val_metrics: dict[str, MetricSeries]
    total_metrics: int


@router.get("/metrics/{session_id}", response_model=MetricsResponse)
async def get_metrics(session_id: str) -> MetricsResponse:
    """Get all metrics for a training session.

    Args:
        session_id: Training session ID

    Returns:
        MetricsResponse containing all training and validation metrics

    Raises:
        HTTPException: If session manager not initialized or session not found
    """
    try:
        manager = get_session_manager()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        session_info = manager.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # For now, return empty metrics since we don't have metrics tracking yet
    # This will be populated when we integrate with metrics collector
    train_metrics: dict[str, MetricSeries] = {}
    val_metrics: dict[str, MetricSeries] = {}

    return MetricsResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        total_metrics=len(train_metrics) + len(val_metrics),
    )
