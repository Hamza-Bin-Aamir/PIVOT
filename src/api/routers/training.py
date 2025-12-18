"""Training control endpoints.

This module provides endpoints for controlling training process lifecycle:
starting, pausing, resuming, and stopping training runs.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..process_manager import ProcessConfig, TrainingProcessManager

# Global process manager instance
_process_manager: TrainingProcessManager | None = None


def get_process_manager() -> TrainingProcessManager:
    """Get the global training process manager instance.

    Returns:
        TrainingProcessManager instance

    Raises:
        HTTPException: If process manager not initialized
    """
    if _process_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Training process manager not initialized",
        )
    return _process_manager


def initialize_process_manager(manager: TrainingProcessManager) -> None:
    """Initialize the global process manager.

    Args:
        manager: TrainingProcessManager instance to use globally
    """
    global _process_manager
    _process_manager = manager


router = APIRouter(prefix="/training", tags=["training-control"])


# Request/Response Models


class StartTrainingRequest(BaseModel):
    """Request model for starting training.

    Attributes:
        config_path: Path to training configuration YAML file
        use_uv: Whether to use 'uv run' for execution (default: True)
        python_executable: Python executable path (default: "python")
        environment: Additional environment variables
    """

    config_path: str = Field(..., description="Path to training config YAML file")
    use_uv: bool = Field(
        default=True, description="Whether to use 'uv run' for execution"
    )
    python_executable: str = Field(
        default="python", description="Python executable path"
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )


class StartTrainingResponse(BaseModel):
    """Response model for start training endpoint.

    Attributes:
        process_id: Unique identifier for the training process
        status: Current process status
        pid: Operating system process ID
        config_path: Path to configuration file used
        message: Success message
    """

    process_id: str
    status: str
    pid: int | None
    config_path: str
    message: str


@router.post("/start", response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest) -> StartTrainingResponse:
    """Start a new training process.

    This endpoint spawns a new training process with the specified configuration.
    The process runs in the background and can be monitored using the status endpoints.

    Args:
        request: Training start request with configuration

    Returns:
        StartTrainingResponse with process information

    Raises:
        HTTPException: If process manager not initialized, config file not found,
                      or process fails to start

    Example:
        POST /api/v1/training/start
        {
            "config_path": "configs/train.yaml",
            "use_uv": true,
            "environment": {"CUDA_VISIBLE_DEVICES": "0"}
        }
    """
    try:
        manager = get_process_manager()
    except HTTPException:
        raise

    # Validate config file exists
    config_file = Path(request.config_path)
    if not config_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Configuration file not found: {request.config_path}",
        )

    if config_file.suffix not in {".yaml", ".yml"}:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration file must be YAML (.yaml or .yml), got {config_file.suffix}",
        )

    # Create process configuration
    try:
        process_config = ProcessConfig(
            config_path=request.config_path,
            use_uv=request.use_uv,
            python_executable=request.python_executable,
            environment=request.environment,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Start the training process
    try:
        process_id = manager.start_process(process_config)
        process_info = manager.get_process_info(process_id)

        return StartTrainingResponse(
            process_id=process_id,
            status=process_info.status.value,
            pid=process_info.pid,
            config_path=request.config_path,
            message=f"Training process started successfully: {process_id}",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
