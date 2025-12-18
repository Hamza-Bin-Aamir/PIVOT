"""Health check endpoints.

This module provides health check endpoints for monitoring service status,
including database connectivity, filesystem access, and dependency checks.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..database import get_db_session

router = APIRouter()

# Track application start time
_start_time = time.time()


class ComponentHealth(BaseModel):
    """Health status of a component."""

    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str | None = None
    details: dict[str, Any] | None = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""

    status: str  # Overall status
    timestamp: str
    uptime_seconds: float
    components: dict[str, ComponentHealth]


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Service status ('healthy' or 'unhealthy')
        timestamp: Current server timestamp in ISO format
        uptime_seconds: Server uptime in seconds
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
        >>>     "uptime_seconds": 123.45
        >>> }
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=time.time() - _start_time,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check with component status.

    Returns:
        Detailed health status including database, filesystem, and dependencies

    Example:
        >>> # GET /api/v1/health/detailed
        >>> {
        >>>     "status": "healthy",
        >>>     "timestamp": "2025-12-18T10:30:00",
        >>>     "uptime_seconds": 123.45,
        >>>     "components": {
        >>>         "database": {"status": "healthy"},
        >>>         "filesystem": {"status": "healthy"},
        >>>         "memory": {"status": "healthy", "details": {...}}
        >>>     }
        >>> }
    """
    components = {}

    # Check database
    try:
        db = get_db_session()
        from sqlalchemy import text
        db.execute(text('SELECT 1'))
        db.close()
        components['database'] = ComponentHealth(
            status='healthy',
            message='Database connection successful',
        )
    except Exception as e:
        components['database'] = ComponentHealth(
            status='unhealthy',
            message=f'Database connection failed: {e!s}',
        )

    # Check filesystem access
    try:
        test_dir = Path('logs')
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / '.health_check'
        test_file.write_text('test')
        test_file.unlink()
        components['filesystem'] = ComponentHealth(
            status='healthy',
            message='Filesystem writable',
        )
    except Exception as e:
        components['filesystem'] = ComponentHealth(
            status='unhealthy',
            message=f'Filesystem check failed: {e!s}',
        )

    # Check memory (if psutil available)
    try:
        import psutil
        mem = psutil.virtual_memory()
        status = 'healthy'
        if mem.percent > 90:
            status = 'unhealthy'
        elif mem.percent > 80:
            status = 'degraded'

        components['memory'] = ComponentHealth(
            status=status,
            message=f'Memory usage: {mem.percent:.1f}%',
            details={
                'percent': mem.percent,
                'available_gb': mem.available / (1024**3),
                'total_gb': mem.total / (1024**3),
            },
        )
    except ImportError:
        components['memory'] = ComponentHealth(
            status='healthy',
            message='psutil not installed, memory check skipped',
        )

    # Determine overall status
    statuses = [c.status for c in components.values()]
    if 'unhealthy' in statuses:
        overall_status = 'unhealthy'
    elif 'degraded' in statuses:
        overall_status = 'degraded'
    else:
        overall_status = 'healthy'

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=time.time() - _start_time,
        components=components,
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
