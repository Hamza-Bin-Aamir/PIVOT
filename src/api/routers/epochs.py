"""Epochs endpoint for retrieving training epoch information.

This module provides an API endpoint for accessing epoch-by-epoch training progress,
including epoch numbers, metrics, and status information.
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


def get_latest_session_id(manager: TrainingSessionManager) -> str:
    """Get the ID of the most recently created session.

    Args:
        manager: TrainingSessionManager instance

    Returns:
        Session ID of the latest session

    Raises:
        HTTPException: If no sessions exist
    """
    sessions = manager.list_sessions()
    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found")

    # Sort by created_at timestamp and return the most recent
    latest = max(sessions, key=lambda s: s.created_at)
    return latest.session_id


class EpochInfo(BaseModel):
    """Model for epoch information.

    Attributes:
        epoch: Epoch number
        status: Current status of the epoch (completed, in_progress, pending)
        train_metrics: Training metrics for the epoch (if available)
        val_metrics: Validation metrics for the epoch (if available)
    """

    epoch: int
    status: str
    train_metrics: dict[str, float] | None = None
    val_metrics: dict[str, float] | None = None


class EpochsResponse(BaseModel):
    """Response model for epochs endpoint.

    Attributes:
        session_id: Training session ID
        experiment_name: Name of the experiment
        total_epochs: Total number of epochs configured
        completed_epochs: Number of completed epochs
        current_epoch: Current epoch number (0 if not started)
        epochs: List of epoch information
    """

    session_id: str
    experiment_name: str
    total_epochs: int
    completed_epochs: int
    current_epoch: int
    epochs: list[EpochInfo]


@router.get("/epochs", response_model=EpochsResponse)
async def get_epochs_no_session() -> EpochsResponse:
    """Get epoch information for the most recent training session.

    Returns:
        EpochsResponse containing epoch-by-epoch information

    Raises:
        HTTPException: If session manager not initialized or no sessions exist
    """
    try:
        manager = get_session_manager()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    session_id = get_latest_session_id(manager)
    return await get_epochs(session_id=session_id)


@router.get("/epochs/{session_id}", response_model=EpochsResponse)
async def get_epochs(session_id: str) -> EpochsResponse:
    """Get epoch information for a training session.

    Args:
        session_id: Training session ID

    Returns:
        EpochsResponse containing epoch-by-epoch information

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

    # For now, return empty epochs list since we don't have epoch tracking yet
    # This will be populated when we integrate with metrics collection
    return EpochsResponse(
        session_id=session_id,
        experiment_name=session_info.experiment_name,
        total_epochs=0,  # TODO: Get from config
        completed_epochs=0,
        current_epoch=0,
        epochs=[],
    )
