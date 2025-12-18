"""Graph data endpoints for dashboard visualization.

This module provides endpoints that return formatted data specifically
for plotting graphs in the frontend dashboard.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..session_manager import TrainingSessionManager

# Global session manager instance
_session_manager: TrainingSessionManager | None = None


def get_session_manager() -> TrainingSessionManager:
    """Get the global training session manager instance.

    Returns:
        TrainingSessionManager instance

    Raises:
        HTTPException: If session manager not initialized
    """
    if _session_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Training session manager not initialized",
        )
    return _session_manager


def set_session_manager(manager: TrainingSessionManager) -> None:
    """Set the global session manager.

    Args:
        manager: TrainingSessionManager instance to use globally
    """
    global _session_manager
    _session_manager = manager


def get_latest_session_id(manager: TrainingSessionManager) -> str:
    """Get the ID of the most recent training session.

    Args:
        manager: TrainingSessionManager instance

    Returns:
        Session ID of the most recent session

    Raises:
        HTTPException: If no sessions exist
    """
    sessions = manager.list_sessions()
    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found")

    # Sort by created_at timestamp and return the most recent
    latest = max(sessions, key=lambda s: s.created_at)
    return latest.session_id


router = APIRouter(prefix="/graphs", tags=["graphs"])


# Response Models


class DataPoint(BaseModel):
    """Single data point for plotting.

    Attributes:
        x: X-axis value (e.g., epoch number, step number)
        y: Y-axis value (e.g., loss value, metric value)
    """

    x: float
    y: float


class LossGraphResponse(BaseModel):
    """Response model for loss graph endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        train_loss: List of training loss data points
        val_loss: List of validation loss data points
        total_points: Total number of data points
    """

    session_id: str
    experiment_name: str
    train_loss: list[DataPoint]
    val_loss: list[DataPoint]
    total_points: int


@router.get("/loss", response_model=LossGraphResponse)
async def get_loss_graph() -> LossGraphResponse:
    """Get loss data formatted for plotting (latest session).

    This endpoint returns training and validation loss values formatted
    specifically for graph plotting in the dashboard. The x-axis represents
    epochs, and the y-axis represents loss values.

    Returns:
        LossGraphResponse with train and validation loss data points

    Raises:
        HTTPException: If session manager not initialized or no sessions exist

    Example:
        GET /api/v1/graphs/loss

        Response:
        {
            "session_id": "session_123",
            "experiment_name": "unet_baseline",
            "train_loss": [
                {"x": 1, "y": 0.523},
                {"x": 2, "y": 0.456}
            ],
            "val_loss": [
                {"x": 1, "y": 0.498},
                {"x": 2, "y": 0.421}
            ],
            "total_points": 4
        }

    Note:
        This endpoint defaults to the most recent training session.
        For specific sessions, use /graphs/loss/{session_id}
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    session_id = get_latest_session_id(manager)
    return await get_loss_graph_by_session(session_id)


@router.get("/loss/{session_id}", response_model=LossGraphResponse)
async def get_loss_graph_by_session(session_id: str) -> LossGraphResponse:
    """Get loss data formatted for plotting for a specific session.

    Args:
        session_id: Training session ID

    Returns:
        LossGraphResponse with train and validation loss data points

    Raises:
        HTTPException: If session manager not initialized or session not found
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_info = manager.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # For now, return empty data - will be populated when metrics collection is integrated
    # This structure is ready for frontend consumption
    train_loss: list[DataPoint] = []
    val_loss: list[DataPoint] = []

    return LossGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        train_loss=train_loss,
        val_loss=val_loss,
        total_points=len(train_loss) + len(val_loss),
    )


class MetricsGraphResponse(BaseModel):
    """Response model for metrics graph endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        metric_name: Name of the metric (e.g., 'accuracy', 'dice', 'iou')
        train_data: List of training metric data points
        val_data: List of validation metric data points
        total_points: Total number of data points
    """

    session_id: str
    experiment_name: str
    metric_name: str
    train_data: list[DataPoint]
    val_data: list[DataPoint]
    total_points: int


@router.get("/metrics/{metric_name}", response_model=MetricsGraphResponse)
async def get_metrics_graph(metric_name: str) -> MetricsGraphResponse:
    """Get metric data formatted for plotting (latest session).

    This endpoint returns training and validation metric values for a specific
    metric (e.g., 'accuracy', 'dice', 'iou') formatted for dashboard plotting.
    The x-axis represents epochs, and the y-axis represents metric values.

    Args:
        metric_name: Name of the metric to retrieve (e.g., 'accuracy', 'dice')

    Returns:
        MetricsGraphResponse with train and validation metric data points

    Raises:
        HTTPException: If session manager not initialized or no sessions exist

    Example:
        GET /api/v1/graphs/metrics/accuracy

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "metric_name": "accuracy",
            "train_data": [
                {"x": 1.0, "y": 0.75},
                {"x": 2.0, "y": 0.82}
            ],
            "val_data": [
                {"x": 1.0, "y": 0.72},
                {"x": 2.0, "y": 0.79}
            ],
            "total_points": 4
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_id = get_latest_session_id(manager)
        session_info = manager.get_session_info(session_id)
    except HTTPException:
        raise

    # For now, return empty data - will be populated when metrics collection is integrated
    # This structure is ready for frontend consumption
    train_data: list[DataPoint] = []
    val_data: list[DataPoint] = []

    return MetricsGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        metric_name=metric_name,
        train_data=train_data,
        val_data=val_data,
        total_points=len(train_data) + len(val_data),
    )


@router.get("/metrics/{metric_name}/{session_id}", response_model=MetricsGraphResponse)
async def get_metrics_graph_by_session(
    metric_name: str, session_id: str
) -> MetricsGraphResponse:
    """Get metric data formatted for plotting for a specific session.

    This endpoint returns training and validation metric values for a specific
    metric and session formatted for dashboard plotting.

    Args:
        metric_name: Name of the metric to retrieve (e.g., 'accuracy', 'dice')
        session_id: Training session ID

    Returns:
        MetricsGraphResponse with train and validation metric data points

    Raises:
        HTTPException: If session manager not initialized or session not found

    Example:
        GET /api/v1/graphs/metrics/accuracy/session_abc123

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "metric_name": "accuracy",
            "train_data": [
                {"x": 1.0, "y": 0.75},
                {"x": 2.0, "y": 0.82}
            ],
            "val_data": [
                {"x": 1.0, "y": 0.72},
                {"x": 2.0, "y": 0.79}
            ],
            "total_points": 4
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_info = manager.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # For now, return empty data - will be populated when metrics collection is integrated
    # This structure is ready for frontend consumption
    train_data: list[DataPoint] = []
    val_data: list[DataPoint] = []

    return MetricsGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        metric_name=metric_name,
        train_data=train_data,
        val_data=val_data,
        total_points=len(train_data) + len(val_data),
    )


class LearningRateGraphResponse(BaseModel):
    """Response model for learning rate graph endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        lr_data: List of learning rate data points
        total_points: Total number of data points
    """

    session_id: str
    experiment_name: str
    lr_data: list[DataPoint]
    total_points: int


@router.get("/learning-rate", response_model=LearningRateGraphResponse)
async def get_learning_rate_graph() -> LearningRateGraphResponse:
    """Get learning rate data formatted for plotting (latest session).

    This endpoint returns learning rate values over the course of training,
    formatted for dashboard plotting. The x-axis represents epochs or steps,
    and the y-axis represents the learning rate value.

    Useful for visualizing learning rate schedules (constant, decay, warmup, etc.)

    Returns:
        LearningRateGraphResponse with learning rate data points

    Raises:
        HTTPException: If session manager not initialized or no sessions exist

    Example:
        GET /api/v1/graphs/learning-rate

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "lr_data": [
                {"x": 1.0, "y": 0.001},
                {"x": 2.0, "y": 0.0009},
                {"x": 3.0, "y": 0.00081}
            ],
            "total_points": 3
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_id = get_latest_session_id(manager)
        session_info = manager.get_session_info(session_id)
    except HTTPException:
        raise

    # For now, return empty data - will be populated when LR tracking is integrated
    # This structure is ready for frontend consumption
    lr_data: list[DataPoint] = []

    return LearningRateGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        lr_data=lr_data,
        total_points=len(lr_data),
    )


@router.get("/learning-rate/{session_id}", response_model=LearningRateGraphResponse)
async def get_learning_rate_graph_by_session(
    session_id: str,
) -> LearningRateGraphResponse:
    """Get learning rate data formatted for plotting for a specific session.

    This endpoint returns learning rate values over the course of training
    for a specific session, formatted for dashboard plotting.

    Args:
        session_id: Training session ID

    Returns:
        LearningRateGraphResponse with learning rate data points

    Raises:
        HTTPException: If session manager not initialized or session not found

    Example:
        GET /api/v1/graphs/learning-rate/session_abc123

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "lr_data": [
                {"x": 1.0, "y": 0.001},
                {"x": 2.0, "y": 0.0009},
                {"x": 3.0, "y": 0.00081}
            ],
            "total_points": 3
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_info = manager.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # For now, return empty data - will be populated when LR tracking is integrated
    # This structure is ready for frontend consumption
    lr_data: list[DataPoint] = []

    return LearningRateGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        lr_data=lr_data,
        total_points=len(lr_data),
    )


class GPUUsageGraphResponse(BaseModel):
    """Response model for GPU usage graph endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        gpu_utilization: GPU compute utilization percentage over time
        memory_usage: GPU memory usage percentage over time
        total_points: Total number of data points
    """

    session_id: str
    experiment_name: str
    gpu_utilization: list[DataPoint]
    memory_usage: list[DataPoint]
    total_points: int


@router.get("/gpu-usage", response_model=GPUUsageGraphResponse)
async def get_gpu_usage_graph() -> GPUUsageGraphResponse:
    """Get GPU usage data formatted for plotting (latest session).

    This endpoint returns GPU utilization and memory usage percentages
    over the course of training, formatted for dashboard monitoring.
    The x-axis represents time (epochs or timestamps), and the y-axis
    represents percentage (0-100).

    Useful for monitoring resource utilization and identifying bottlenecks.

    Returns:
        GPUUsageGraphResponse with GPU utilization and memory usage data points

    Raises:
        HTTPException: If session manager not initialized or no sessions exist

    Example:
        GET /api/v1/graphs/gpu-usage

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "gpu_utilization": [
                {"x": 1.0, "y": 85.5},
                {"x": 2.0, "y": 87.2}
            ],
            "memory_usage": [
                {"x": 1.0, "y": 92.1},
                {"x": 2.0, "y": 91.8}
            ],
            "total_points": 4
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_id = get_latest_session_id(manager)
        session_info = manager.get_session_info(session_id)
    except HTTPException:
        raise

    # For now, return empty data - will be populated when GPU monitoring is integrated
    # This structure is ready for frontend consumption
    gpu_utilization: list[DataPoint] = []
    memory_usage: list[DataPoint] = []

    return GPUUsageGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_usage,
        total_points=len(gpu_utilization) + len(memory_usage),
    )


@router.get("/gpu-usage/{session_id}", response_model=GPUUsageGraphResponse)
async def get_gpu_usage_graph_by_session(
    session_id: str,
) -> GPUUsageGraphResponse:
    """Get GPU usage data formatted for plotting for a specific session.

    This endpoint returns GPU utilization and memory usage data for a
    specific training session, formatted for dashboard monitoring.

    Args:
        session_id: Training session ID

    Returns:
        GPUUsageGraphResponse with GPU utilization and memory usage data points

    Raises:
        HTTPException: If session manager not initialized or session not found

    Example:
        GET /api/v1/graphs/gpu-usage/session_abc123

        Response:
        {
            "session_id": "session_abc123",
            "experiment_name": "baseline_v1",
            "gpu_utilization": [
                {"x": 1.0, "y": 85.5},
                {"x": 2.0, "y": 87.2}
            ],
            "memory_usage": [
                {"x": 1.0, "y": 92.1},
                {"x": 2.0, "y": 91.8}
            ],
            "total_points": 4
        }
    """
    try:
        manager = get_session_manager()
    except HTTPException:
        raise

    try:
        session_info = manager.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # For now, return empty data - will be populated when GPU monitoring is integrated
    # This structure is ready for frontend consumption
    gpu_utilization: list[DataPoint] = []
    memory_usage: list[DataPoint] = []

    return GPUUsageGraphResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_usage,
        total_points=len(gpu_utilization) + len(memory_usage),
    )
