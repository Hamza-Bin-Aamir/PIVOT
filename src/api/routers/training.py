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


class ProcessControlRequest(BaseModel):
    """Request model for controlling a training process.

    Attributes:
        process_id: Unique identifier of the process to control
    """

    process_id: str = Field(..., description="Process ID to control")


class ProcessControlResponse(BaseModel):
    """Response model for process control operations.

    Attributes:
        process_id: Unique identifier of the process
        status: Current process status after the operation
        message: Success or informational message
    """

    process_id: str
    status: str
    message: str


@router.post("/pause", response_model=ProcessControlResponse)
async def pause_training(request: ProcessControlRequest) -> ProcessControlResponse:
    """Pause a running training process.

    This endpoint pauses a training process by sending a SIGSTOP signal (Unix only).
    The process can be resumed later using the resume endpoint.

    Args:
        request: Process control request with process_id

    Returns:
        ProcessControlResponse with updated status

    Raises:
        HTTPException: If process manager not initialized, process not found,
                      or process cannot be paused

    Example:
        POST /api/v1/training/pause
        {
            "process_id": "train_20251218_120000"
        }

    Note:
        Only RUNNING processes can be paused. Attempting to pause a process
        in any other state will result in an error.
    """
    try:
        manager = get_process_manager()
    except HTTPException:
        raise

    try:
        manager.pause_process(request.process_id)
        process_info = manager.get_process_info(request.process_id)

        return ProcessControlResponse(
            process_id=request.process_id,
            status=process_info.status.value,
            message=f"Training process paused successfully: {request.process_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/resume", response_model=ProcessControlResponse)
async def resume_training(request: ProcessControlRequest) -> ProcessControlResponse:
    """Resume a paused training process.

    This endpoint resumes a paused training process by sending a SIGCONT signal (Unix only).

    Args:
        request: Process control request with process_id

    Returns:
        ProcessControlResponse with updated status

    Raises:
        HTTPException: If process manager not initialized, process not found,
                      or process cannot be resumed

    Example:
        POST /api/v1/training/resume
        {
            "process_id": "train_20251218_120000"
        }

    Note:
        Only PAUSED processes can be resumed. Attempting to resume a process
        in any other state will result in an error.
    """
    try:
        manager = get_process_manager()
    except HTTPException:
        raise

    try:
        manager.resume_process(request.process_id)
        process_info = manager.get_process_info(request.process_id)

        return ProcessControlResponse(
            process_id=request.process_id,
            status=process_info.status.value,
            message=f"Training process resumed successfully: {request.process_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/stop", response_model=ProcessControlResponse)
async def stop_training(request: ProcessControlRequest) -> ProcessControlResponse:
    """Stop a running or paused training process.

    This endpoint gracefully stops a training process by sending SIGTERM.
    If the process doesn't terminate within 5 seconds, it will be forcefully
    killed with SIGKILL.

    Args:
        request: Process control request with process_id

    Returns:
        ProcessControlResponse with updated status

    Raises:
        HTTPException: If process manager not initialized, process not found,
                      or process cannot be stopped

    Example:
        POST /api/v1/training/stop
        {
            "process_id": "train_20251218_120000"
        }

    Note:
        Only RUNNING or PAUSED processes can be stopped. The process will
        be terminated gracefully, with force kill as fallback.
    """
    try:
        manager = get_process_manager()
    except HTTPException:
        raise

    try:
        manager.stop_process(request.process_id)
        process_info = manager.get_process_info(request.process_id)

        return ProcessControlResponse(
            process_id=request.process_id,
            status=process_info.status.value,
            message=f"Training process stopped successfully: {request.process_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
