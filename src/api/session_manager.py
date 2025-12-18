"""Training session manager for managing training lifecycle.

This module provides high-level session management for training jobs,
including session metadata, status tracking, and integration with the
process manager.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .process_manager import (
    ProcessConfig,
    ProcessStatus,
    TrainingProcessManager,
)


@dataclass
class SessionConfig:
    """Configuration for a training session.

    Attributes:
        config_path: Path to training configuration YAML file
        experiment_name: Name for the experiment/session
        description: Optional description of the session
        tags: List of tags for categorizing sessions
        use_uv: Whether to use uv run for executing training
    """

    config_path: str | Path
    experiment_name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    use_uv: bool = True

    def __post_init__(self) -> None:
        """Validate session configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")

        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")


@dataclass
class SessionInfo:
    """Information about a training session.

    Attributes:
        session_id: Unique session identifier
        experiment_name: Experiment name
        description: Session description
        tags: Session tags
        config_path: Path to configuration file
        created_at: Session creation timestamp
        started_at: Training start timestamp (None if not started)
        ended_at: Training end timestamp (None if not ended)
        process_id: Associated process ID (None if not started)
        status: Current session status
    """

    session_id: str
    experiment_name: str
    description: str
    tags: list[str]
    config_path: str
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    process_id: str | None = None
    status: ProcessStatus = ProcessStatus.IDLE

    def to_dict(self) -> dict[str, Any]:
        """Convert session info to dictionary.

        Returns:
            Dictionary representation of session info
        """
        return {
            "session_id": self.session_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "config_path": self.config_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "process_id": self.process_id,
            "status": self.status.value,
        }


class TrainingSessionManager:
    """Manager for training sessions.

    This class provides high-level management of training sessions,
    integrating with the process manager to handle training execution.

    Attributes:
        sessions: Dictionary mapping session IDs to SessionInfo objects
        _process_manager: TrainingProcessManager instance
        _lock: Thread lock for session dictionary access

    Example:
        >>> manager = TrainingSessionManager()
        >>> config = SessionConfig(
        ...     config_path="configs/train.yaml",
        ...     experiment_name="baseline_v1",
        ...     tags=["baseline", "experiment"],
        ... )
        >>> session_id = manager.create_session(config)
        >>> manager.start_session(session_id)
        >>> info = manager.get_session_info(session_id)
        >>> info.status
        <ProcessStatus.RUNNING: 'running'>
    """

    def __init__(self, process_manager: TrainingProcessManager | None = None) -> None:
        """Initialize the training session manager.

        Args:
            process_manager: Optional process manager. If None, creates new instance
        """
        self.sessions: dict[str, SessionInfo] = {}
        self._process_manager = process_manager or TrainingProcessManager()
        self._lock = threading.Lock()

    def create_session(self, config: SessionConfig) -> str:
        """Create a new training session.

        Args:
            config: Session configuration

        Returns:
            Unique session ID
        """
        with self._lock:
            # Generate unique session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create session info
            info = SessionInfo(
                session_id=session_id,
                experiment_name=config.experiment_name,
                description=config.description,
                tags=config.tags.copy(),
                config_path=str(config.config_path),
                created_at=datetime.now(),
            )

            self.sessions[session_id] = info
            return session_id

    def start_session(self, session_id: str) -> str:
        """Start a training session.

        Args:
            session_id: Session ID to start

        Returns:
            Process ID of started training process

        Raises:
            ValueError: If session not found
            RuntimeError: If session is not in IDLE state
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session not found: {session_id}")

            info = self.sessions[session_id]

            if info.status != ProcessStatus.IDLE:
                raise RuntimeError(
                    f"Cannot start session in {info.status.value} state"
                )

            # Create process configuration
            process_config = ProcessConfig(
                config_path=info.config_path,
                use_uv=True,
            )

            # Start training process
            process_id = self._process_manager.start_process(process_config)

            # Update session info
            info.process_id = process_id
            info.started_at = datetime.now()
            info.status = ProcessStatus.RUNNING

            return process_id

    def stop_session(self, session_id: str) -> None:
        """Stop a running training session.

        Args:
            session_id: Session ID to stop

        Raises:
            ValueError: If session not found
            RuntimeError: If session has no associated process
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session not found: {session_id}")

            info = self.sessions[session_id]

            if info.process_id is None:
                raise RuntimeError("Session has no associated process")

            # Stop the process
            self._process_manager.stop_process(info.process_id)

            # Update session status
            info.status = ProcessStatus.STOPPED
            info.ended_at = datetime.now()

    def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a session.

        Args:
            session_id: Session ID

        Returns:
            SessionInfo object

        Raises:
            ValueError: If session not found
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session not found: {session_id}")
            return self.sessions[session_id]

    def list_sessions(self, tags: list[str] | None = None) -> list[SessionInfo]:
        """List sessions, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by. Returns sessions
                 that have ALL specified tags

        Returns:
            List of SessionInfo objects
        """
        with self._lock:
            sessions = list(self.sessions.values())

            if tags:
                # Filter by tags - session must have all specified tags
                sessions = [
                    s for s in sessions if all(tag in s.tags for tag in tags)
                ]

            return sessions

    def update_session_status(self, session_id: str) -> ProcessStatus:
        """Update session status from associated process.

        Args:
            session_id: Session ID

        Returns:
            Updated session status

        Raises:
            ValueError: If session not found
            RuntimeError: If session has no associated process
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session not found: {session_id}")

            info = self.sessions[session_id]

            if info.process_id is None:
                return info.status

            # Get latest process status
            process_status = self._process_manager.update_process_status(info.process_id)

            # Update session status
            old_status = info.status
            info.status = process_status

            # Set end time if status changed to terminal state
            if old_status != process_status and process_status in {
                ProcessStatus.COMPLETED,
                ProcessStatus.FAILED,
                ProcessStatus.STOPPED,
            }:
                info.ended_at = datetime.now()

            return info.status

    def delete_session(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: Session ID to delete

        Raises:
            ValueError: If session not found
            RuntimeError: If session is currently running
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session not found: {session_id}")

            info = self.sessions[session_id]

            if info.status == ProcessStatus.RUNNING:
                raise RuntimeError("Cannot delete running session. Stop it first.")

            del self.sessions[session_id]

    def get_session_by_experiment(self, experiment_name: str) -> list[SessionInfo]:
        """Get all sessions for a given experiment name.

        Args:
            experiment_name: Experiment name to search for

        Returns:
            List of SessionInfo objects matching the experiment name
        """
        with self._lock:
            return [
                info
                for info in self.sessions.values()
                if info.experiment_name == experiment_name
            ]

    def cleanup_completed_sessions(self) -> int:
        """Remove completed/failed/stopped sessions from tracking.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            completed_states = {
                ProcessStatus.COMPLETED,
                ProcessStatus.FAILED,
                ProcessStatus.STOPPED,
            }

            to_remove = [
                sid
                for sid, info in self.sessions.items()
                if info.status in completed_states
            ]

            for sid in to_remove:
                del self.sessions[sid]

            return len(to_remove)
