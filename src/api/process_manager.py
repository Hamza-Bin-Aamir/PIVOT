"""Training process manager for spawning and controlling training jobs.

This module provides functionality to spawn training processes, monitor their
status, and control their lifecycle (pause, resume, stop).
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ProcessStatus(Enum):
    """Training process status."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class ProcessInfo:
    """Information about a training process.

    Attributes:
        process_id: Unique process identifier
        pid: Operating system process ID (None if not started)
        status: Current process status
        config_path: Path to training configuration file
        start_time: Process start timestamp
        end_time: Process end timestamp (None if still running)
        exit_code: Process exit code (None if still running)
        error_message: Error message if process failed
    """

    process_id: str
    pid: int | None = None
    status: ProcessStatus = ProcessStatus.IDLE
    config_path: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    exit_code: int | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert process info to dictionary.

        Returns:
            Dictionary representation of process info
        """
        return {
            "process_id": self.process_id,
            "pid": self.pid,
            "status": self.status.value,
            "config_path": self.config_path,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
        }


@dataclass
class ProcessConfig:
    """Configuration for spawning training process.

    Attributes:
        config_path: Path to training configuration YAML file
        python_executable: Path to Python executable (default: "python")
        training_script: Path to training script (default: "src/train/main.py")
        use_uv: Whether to use uv run instead of direct python execution
        environment: Additional environment variables
    """

    config_path: str | Path
    python_executable: str = "python"
    training_script: str = "src/train/main.py"
    use_uv: bool = True
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate process configuration."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        if config_path.suffix not in {".yaml", ".yml"}:
            raise ValueError(
                f"Config file must be YAML (.yaml or .yml), got {config_path.suffix}"
            )


class TrainingProcessManager:
    """Manager for training process lifecycle.

    This class handles spawning training processes as subprocesses,
    monitoring their status, and controlling their execution.

    Attributes:
        processes: Dictionary mapping process IDs to ProcessInfo objects
        _process_handles: Dictionary mapping process IDs to subprocess.Popen objects
        _lock: Thread lock for process dictionary access

    Example:
        >>> manager = TrainingProcessManager()
        >>> config = ProcessConfig(config_path="configs/train.yaml")
        >>> process_id = manager.start_process(config)
        >>> info = manager.get_process_info(process_id)
        >>> info.status
        <ProcessStatus.RUNNING: 'running'>
        >>> manager.stop_process(process_id)
    """

    def __init__(self) -> None:
        """Initialize the training process manager."""
        self.processes: dict[str, ProcessInfo] = {}
        self._process_handles: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def start_process(self, config: ProcessConfig) -> str:
        """Start a new training process.

        Args:
            config: Process configuration

        Returns:
            Unique process ID

        Raises:
            RuntimeError: If process fails to start
        """
        # Generate unique process ID
        process_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._lock:
            # Create process info
            info = ProcessInfo(
                process_id=process_id,
                status=ProcessStatus.STARTING,
                config_path=str(config.config_path),
                start_time=datetime.now(),
            )
            self.processes[process_id] = info

            try:
                # Build command
                if config.use_uv:
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        config.training_script,
                        "--config",
                        str(config.config_path),
                    ]
                else:
                    cmd = [
                        config.python_executable,
                        config.training_script,
                        "--config",
                        str(config.config_path),
                    ]

                # Start subprocess
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, **config.environment},
                )

                # Update process info
                info.pid = process.pid
                info.status = ProcessStatus.RUNNING
                self._process_handles[process_id] = process

                return process_id

            except Exception as e:
                # Mark as failed
                info.status = ProcessStatus.FAILED
                info.error_message = str(e)
                info.end_time = datetime.now()
                raise RuntimeError(f"Failed to start process: {e}") from e

    def stop_process(self, process_id: str) -> None:
        """Stop a running training process.

        Args:
            process_id: Process ID to stop

        Raises:
            ValueError: If process ID not found
            RuntimeError: If process cannot be stopped
        """
        with self._lock:
            if process_id not in self.processes:
                raise ValueError(f"Process not found: {process_id}")

            info = self.processes[process_id]
            process = self._process_handles.get(process_id)

            if process is None:
                raise RuntimeError(f"Process handle not found: {process_id}")

            if info.status not in {ProcessStatus.RUNNING, ProcessStatus.PAUSED}:
                raise RuntimeError(
                    f"Cannot stop process in {info.status.value} state"
                )

            try:
                info.status = ProcessStatus.STOPPING
                process.terminate()

                # Wait for graceful shutdown (max 5 seconds)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    process.kill()
                    process.wait()

                info.status = ProcessStatus.STOPPED
                info.end_time = datetime.now()
                info.exit_code = process.returncode

            except Exception as e:
                info.status = ProcessStatus.FAILED
                info.error_message = f"Failed to stop process: {e}"
                raise RuntimeError(f"Failed to stop process: {e}") from e

    def pause_process(self, process_id: str) -> None:
        """Pause a running training process.

        Args:
            process_id: Process ID to pause

        Raises:
            ValueError: If process ID not found
            RuntimeError: If process cannot be paused or not in RUNNING state

        Note:
            On Unix-like systems, this sends SIGSTOP to pause the process.
            The process can be resumed later with resume_process().
        """
        with self._lock:
            if process_id not in self.processes:
                raise ValueError(f"Process not found: {process_id}")

            info = self.processes[process_id]
            process = self._process_handles.get(process_id)

            if process is None:
                raise RuntimeError(f"Process handle not found: {process_id}")

            if info.status != ProcessStatus.RUNNING:
                raise RuntimeError(
                    f"Cannot pause process in {info.status.value} state. "
                    "Only RUNNING processes can be paused."
                )

            try:
                # Send SIGSTOP signal to pause the process (Unix only)
                import signal

                process.send_signal(signal.SIGSTOP)
                info.status = ProcessStatus.PAUSED

            except Exception as e:
                info.error_message = f"Failed to pause process: {e}"
                raise RuntimeError(f"Failed to pause process: {e}") from e

    def resume_process(self, process_id: str) -> None:
        """Resume a paused training process.

        Args:
            process_id: Process ID to resume

        Raises:
            ValueError: If process ID not found
            RuntimeError: If process cannot be resumed or not in PAUSED state

        Note:
            On Unix-like systems, this sends SIGCONT to resume the process.
        """
        with self._lock:
            if process_id not in self.processes:
                raise ValueError(f"Process not found: {process_id}")

            info = self.processes[process_id]
            process = self._process_handles.get(process_id)

            if process is None:
                raise RuntimeError(f"Process handle not found: {process_id}")

            if info.status != ProcessStatus.PAUSED:
                raise RuntimeError(
                    f"Cannot resume process in {info.status.value} state. "
                    "Only PAUSED processes can be resumed."
                )

            try:
                # Send SIGCONT signal to resume the process (Unix only)
                import signal

                process.send_signal(signal.SIGCONT)
                info.status = ProcessStatus.RUNNING

            except Exception as e:
                info.error_message = f"Failed to resume process: {e}"
                raise RuntimeError(f"Failed to resume process: {e}") from e

    def get_process_info(self, process_id: str) -> ProcessInfo:
        """Get information about a process.

        Args:
            process_id: Process ID

        Returns:
            ProcessInfo object

        Raises:
            ValueError: If process ID not found
        """
        with self._lock:
            if process_id not in self.processes:
                raise ValueError(f"Process not found: {process_id}")
            return self.processes[process_id]

    def list_processes(self) -> list[ProcessInfo]:
        """List all processes.

        Returns:
            List of ProcessInfo objects
        """
        with self._lock:
            return list(self.processes.values())

    def update_process_status(self, process_id: str) -> ProcessStatus:
        """Update and return current process status.

        Args:
            process_id: Process ID

        Returns:
            Current process status

        Raises:
            ValueError: If process ID not found
        """
        with self._lock:
            if process_id not in self.processes:
                raise ValueError(f"Process not found: {process_id}")

            info = self.processes[process_id]
            process = self._process_handles.get(process_id)

            if process is None:
                return info.status

            # Check if process has completed
            if info.status == ProcessStatus.RUNNING:
                poll_result = process.poll()
                if poll_result is not None:
                    # Process has terminated
                    info.exit_code = poll_result
                    info.end_time = datetime.now()

                    if poll_result == 0:
                        info.status = ProcessStatus.COMPLETED
                    else:
                        info.status = ProcessStatus.FAILED
                        # Try to read error output
                        try:
                            _, stderr = process.communicate(timeout=1)
                            info.error_message = stderr.decode()[:500]  # First 500 chars
                        except subprocess.TimeoutExpired:
                            pass

            return info.status

    def cleanup_completed(self) -> int:
        """Remove completed/failed/stopped processes from tracking.

        Returns:
            Number of processes cleaned up
        """
        with self._lock:
            completed_states = {
                ProcessStatus.COMPLETED,
                ProcessStatus.FAILED,
                ProcessStatus.STOPPED,
            }

            to_remove = [
                pid
                for pid, info in self.processes.items()
                if info.status in completed_states
            ]

            for pid in to_remove:
                del self.processes[pid]
                if pid in self._process_handles:
                    del self._process_handles[pid]

            return len(to_remove)
