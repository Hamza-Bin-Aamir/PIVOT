"""Health check endpoints.

This module provides health check endpoints for monitoring service status.
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Service status ('healthy' or 'unhealthy')
        timestamp: Current server timestamp in ISO format
        uptime_seconds: Server uptime in seconds (placeholder for now)
    """

    status: str
    timestamp: str
    uptime_seconds: float = 0.0


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health status.

    Returns:
        HealthResponse with current status and timestamp

    Example:
        >>> # GET /api/v1/health
        >>> {
        >>>     "status": "healthy",
        >>>     "timestamp": "2025-12-18T10:30:00.123456",
        >>>     "uptime_seconds": 0.0
        >>> }
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
    )


@router.get("/ping")
async def ping() -> dict[str, str]:
    """Simple ping endpoint.

    Returns:
        Dictionary with 'pong' message

    Example:
        >>> # GET /api/v1/ping
        >>> {"message": "pong"}
    """
    return {"message": "pong"}
