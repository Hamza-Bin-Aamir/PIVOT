"""System monitoring endpoints.

This module provides REST API endpoints for monitoring system resources
including CPU, memory, disk, and GPU usage.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..monitoring import monitor

router = APIRouter(prefix="/monitor", tags=["monitoring"])


class ResourceMetrics(BaseModel):
    """System resource metrics response."""

    timestamp: str
    system: dict[str, Any]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    disk: dict[str, Any]
    gpu: dict[str, Any]


@router.get("/resources", response_model=ResourceMetrics)
async def get_resource_metrics() -> ResourceMetrics:
    """Get current system resource metrics.

    Returns:
        Complete system metrics including CPU, memory, disk, and GPU

    Example:
        ```json
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "system": {"platform": "Linux", "hostname": "server01"},
            "cpu": {"percent": 45.2, "count": 8},
            "memory": {"percent": 62.1, "available": 8589934592},
            "disk": {"partitions": [...]},
            "gpu": {"available": true, "count": 2, "gpus": [...]}
        }
        ```
    """
    metrics = monitor.get_all_metrics()
    return ResourceMetrics(**metrics)


@router.get("/cpu")
async def get_cpu_metrics() -> dict[str, Any]:
    """Get CPU metrics only.

    Returns:
        CPU usage information
    """
    return monitor.get_cpu_info()


@router.get("/memory")
async def get_memory_metrics() -> dict[str, Any]:
    """Get memory metrics only.

    Returns:
        Memory usage information
    """
    return monitor.get_memory_info()


@router.get("/disk")
async def get_disk_metrics() -> dict[str, Any]:
    """Get disk metrics only.

    Returns:
        Disk usage information
    """
    return monitor.get_disk_info()


@router.get("/gpu")
async def get_gpu_metrics() -> dict[str, Any]:
    """Get GPU metrics only.

    Returns:
        GPU usage information if available
    """
    return monitor.get_gpu_info()
