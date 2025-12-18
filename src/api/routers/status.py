"""Training status endpoints.

This module provides endpoints for querying training status and session information.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..session_manager import TrainingSessionManager

router = APIRouter()

# Global session manager instance (will be initialized in main app)
_session_manager: TrainingSessionManager | None = None


def set_session_manager(manager: TrainingSessionManager) -> None:
    """Set the global session manager instance.

    Args:
        manager: TrainingSessionManager instance to use
    """
    global _session_manager
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


class StatusResponse(BaseModel):
    """Training status response model.

    Attributes:
        active_sessions: Number of currently active training sessions
        total_sessions: Total number of sessions in the system
        running_sessions: Number of sessions currently running
        timestamp: Current server timestamp in ISO format
        sessions: List of session information dictionaries
    """

    active_sessions: int
    total_sessions: int
    running_sessions: int
    timestamp: str
    sessions: list[dict]


@router.get("/status", response_model=StatusResponse)
async def get_training_status() -> StatusResponse:
    """Get overall training status.

    Returns:
        StatusResponse with current training status and active sessions

    Raises:
        HTTPException: If session manager not initialized

    Example:
        >>> # GET /api/v1/status
        >>> {
        >>>     "active_sessions": 2,
        >>>     "total_sessions": 5,
        >>>     "running_sessions": 1,
        >>>     "timestamp": "2025-12-18T10:30:00.123456",
        >>>     "sessions": [...]
        >>> }
    """
    try:
        manager = get_session_manager()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    sessions = manager.list_sessions()

    # Count sessions by status
    from ..process_manager import ProcessStatus

    running_count = sum(1 for s in sessions if s.status == ProcessStatus.RUNNING)
    active_count = sum(
        1
        for s in sessions
        if s.status
        in {ProcessStatus.RUNNING, ProcessStatus.STARTING, ProcessStatus.PAUSED}
    )

    return StatusResponse(
        active_sessions=active_count,
        total_sessions=len(sessions),
        running_sessions=running_count,
        timestamp=datetime.now().isoformat(),
        sessions=[s.to_dict() for s in sessions],
    )
