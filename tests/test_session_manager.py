"""Tests for training session manager."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.process_manager import ProcessStatus, TrainingProcessManager
from src.api.session_manager import (
    SessionConfig,
    SessionInfo,
    TrainingSessionManager,
)


class TestSessionConfig:
    """Test suite for SessionConfig."""

    def test_default_config(self, tmp_path: Path):
        """Test default session configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        config = SessionConfig(
            config_path=config_file,
            experiment_name="test_experiment",
        )

        assert config.config_path == config_file
        assert config.experiment_name == "test_experiment"
        assert config.description == ""
        assert config.tags == []
        assert config.use_uv is True

    def test_custom_config(self, tmp_path: Path):
        """Test custom session configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        config = SessionConfig(
            config_path=config_file,
            experiment_name="custom_exp",
            description="Custom experiment description",
            tags=["baseline", "v1"],
            use_uv=False,
        )

        assert config.experiment_name == "custom_exp"
        assert config.description == "Custom experiment description"
        assert config.tags == ["baseline", "v1"]
        assert config.use_uv is False

    def test_empty_experiment_name_raises_error(self, tmp_path: Path):
        """Test that empty experiment name raises error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            SessionConfig(config_path=config_file, experiment_name="")

    def test_nonexistent_config_raises_error(self, tmp_path: Path):
        """Test that nonexistent config file raises error."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            SessionConfig(config_path=config_file, experiment_name="test")


class TestSessionInfo:
    """Test suite for SessionInfo."""

    def test_default_session_info(self):
        """Test default session info."""
        from datetime import datetime

        created_at = datetime.now()
        info = SessionInfo(
            session_id="session_123",
            experiment_name="exp1",
            description="Test",
            tags=["tag1"],
            config_path="/path/to/config.yaml",
            created_at=created_at,
        )

        assert info.session_id == "session_123"
        assert info.experiment_name == "exp1"
        assert info.description == "Test"
        assert info.tags == ["tag1"]
        assert info.created_at == created_at
        assert info.started_at is None
        assert info.ended_at is None
        assert info.process_id is None
        assert info.status == ProcessStatus.IDLE

    def test_to_dict(self):
        """Test converting session info to dictionary."""
        from datetime import datetime

        created_at = datetime.now()
        info = SessionInfo(
            session_id="session_456",
            experiment_name="exp2",
            description="Description",
            tags=["tag1", "tag2"],
            config_path="/path/config.yaml",
            created_at=created_at,
        )

        result = info.to_dict()

        assert result["session_id"] == "session_456"
        assert result["experiment_name"] == "exp2"
        assert result["description"] == "Description"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["config_path"] == "/path/config.yaml"
        assert result["created_at"] == created_at.isoformat()
        assert result["started_at"] is None
        assert result["ended_at"] is None
        assert result["process_id"] is None
        assert result["status"] == "idle"


class TestTrainingSessionManager:
    """Test suite for TrainingSessionManager."""

    def test_init_default(self):
        """Test manager initialization with default process manager."""
        manager = TrainingSessionManager()

        assert manager.sessions == {}
        assert isinstance(manager._process_manager, TrainingProcessManager)
        assert manager._lock is not None

    def test_init_custom_process_manager(self):
        """Test manager initialization with custom process manager."""
        process_manager = TrainingProcessManager()
        manager = TrainingSessionManager(process_manager=process_manager)

        assert manager._process_manager is process_manager

    def test_create_session(self, tmp_path: Path):
        """Test creating a new session."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()
        config = SessionConfig(
            config_path=config_file,
            experiment_name="baseline_v1",
            description="Baseline experiment",
            tags=["baseline", "v1"],
        )

        session_id = manager.create_session(config)

        assert session_id.startswith("session_")
        assert session_id in manager.sessions

        info = manager.sessions[session_id]
        assert info.experiment_name == "baseline_v1"
        assert info.description == "Baseline experiment"
        assert info.tags == ["baseline", "v1"]
        assert info.status == ProcessStatus.IDLE

    @patch.object(TrainingProcessManager, "start_process")
    def test_start_session(
        self, mock_start: MagicMock, tmp_path: Path
    ):
        """Test starting a session."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)

        process_id = manager.start_session(session_id)

        assert process_id == "process_123"
        info = manager.get_session_info(session_id)
        assert info.process_id == "process_123"
        assert info.status == ProcessStatus.RUNNING
        assert info.started_at is not None

        mock_start.assert_called_once()

    def test_start_session_not_found(self):
        """Test starting nonexistent session."""
        manager = TrainingSessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            manager.start_session("nonexistent")

    @patch.object(TrainingProcessManager, "start_process")
    def test_start_session_invalid_state(
        self, mock_start: MagicMock, tmp_path: Path
    ):
        """Test starting session in invalid state."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)
        manager.start_session(session_id)

        # Try to start again
        with pytest.raises(RuntimeError, match="Cannot start session"):
            manager.start_session(session_id)

    @patch.object(TrainingProcessManager, "start_process")
    @patch.object(TrainingProcessManager, "stop_process")
    def test_stop_session(
        self, mock_stop: MagicMock, mock_start: MagicMock, tmp_path: Path
    ):
        """Test stopping a session."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)
        manager.start_session(session_id)

        manager.stop_session(session_id)

        info = manager.get_session_info(session_id)
        assert info.status == ProcessStatus.STOPPED
        assert info.ended_at is not None

        mock_stop.assert_called_once_with("process_123")

    def test_stop_session_not_found(self):
        """Test stopping nonexistent session."""
        manager = TrainingSessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            manager.stop_session("nonexistent")

    def test_stop_session_no_process(self, tmp_path: Path):
        """Test stopping session with no associated process."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)

        with pytest.raises(RuntimeError, match="no associated process"):
            manager.stop_session(session_id)

    def test_get_session_info(self, tmp_path: Path):
        """Test getting session information."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)

        info = manager.get_session_info(session_id)

        assert info.session_id == session_id
        assert info.experiment_name == "exp1"

    def test_get_session_info_not_found(self):
        """Test getting info for nonexistent session."""
        manager = TrainingSessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            manager.get_session_info("nonexistent")

    def test_list_sessions(self, tmp_path: Path):
        """Test listing all sessions."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()

        assert manager.list_sessions() == []

        config1 = SessionConfig(
            config_path=config_file,
            experiment_name="exp1",
            tags=["baseline"],
        )
        session_id1 = manager.create_session(config1)

        time.sleep(1.1)  # Ensure different timestamp

        config2 = SessionConfig(
            config_path=config_file,
            experiment_name="exp2",
            tags=["improved"],
        )
        session_id2 = manager.create_session(config2)

        sessions = manager.list_sessions()
        assert len(sessions) == 2
        assert any(s.session_id == session_id1 for s in sessions)
        assert any(s.session_id == session_id2 for s in sessions)

    def test_list_sessions_filtered_by_tags(self, tmp_path: Path):
        """Test listing sessions filtered by tags."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()

        config1 = SessionConfig(
            config_path=config_file,
            experiment_name="exp1",
            tags=["baseline", "v1"],
        )
        manager.create_session(config1)

        time.sleep(1.1)

        config2 = SessionConfig(
            config_path=config_file,
            experiment_name="exp2",
            tags=["improved", "v2"],
        )
        session_id2 = manager.create_session(config2)

        time.sleep(1.1)

        config3 = SessionConfig(
            config_path=config_file,
            experiment_name="exp3",
            tags=["baseline", "v2"],
        )
        manager.create_session(config3)

        # Filter by single tag
        sessions = manager.list_sessions(tags=["baseline"])
        assert len(sessions) == 2

        # Filter by multiple tags
        sessions = manager.list_sessions(tags=["improved", "v2"])
        assert len(sessions) == 1
        assert sessions[0].session_id == session_id2

    @patch.object(TrainingProcessManager, "start_process")
    @patch.object(TrainingProcessManager, "update_process_status")
    def test_update_session_status(
        self, mock_update: MagicMock, mock_start: MagicMock, tmp_path: Path
    ):
        """Test updating session status."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"
        mock_update.return_value = ProcessStatus.COMPLETED

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)
        manager.start_session(session_id)

        status = manager.update_session_status(session_id)

        assert status == ProcessStatus.COMPLETED
        info = manager.get_session_info(session_id)
        assert info.status == ProcessStatus.COMPLETED
        assert info.ended_at is not None

    def test_update_status_not_found(self):
        """Test updating status for nonexistent session."""
        manager = TrainingSessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            manager.update_session_status("nonexistent")

    def test_delete_session(self, tmp_path: Path):
        """Test deleting a session."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)

        assert session_id in manager.sessions

        manager.delete_session(session_id)

        assert session_id not in manager.sessions

    def test_delete_session_not_found(self):
        """Test deleting nonexistent session."""
        manager = TrainingSessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            manager.delete_session("nonexistent")

    @patch.object(TrainingProcessManager, "start_process")
    def test_delete_running_session(
        self, mock_start: MagicMock, tmp_path: Path
    ):
        """Test deleting running session."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)
        manager.start_session(session_id)

        with pytest.raises(RuntimeError, match="Cannot delete running session"):
            manager.delete_session(session_id)

    def test_get_session_by_experiment(self, tmp_path: Path):
        """Test getting sessions by experiment name."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingSessionManager()

        config1 = SessionConfig(config_path=config_file, experiment_name="baseline")
        session_id1 = manager.create_session(config1)

        time.sleep(1.1)

        config2 = SessionConfig(config_path=config_file, experiment_name="baseline")
        session_id2 = manager.create_session(config2)

        time.sleep(1.1)

        config3 = SessionConfig(config_path=config_file, experiment_name="improved")
        manager.create_session(config3)

        sessions = manager.get_session_by_experiment("baseline")
        assert len(sessions) == 2
        assert any(s.session_id == session_id1 for s in sessions)
        assert any(s.session_id == session_id2 for s in sessions)

    @patch.object(TrainingProcessManager, "start_process")
    @patch.object(TrainingProcessManager, "stop_process")
    def test_cleanup_completed_sessions(
        self, mock_stop: MagicMock, mock_start: MagicMock, tmp_path: Path
    ):
        """Test cleaning up completed sessions."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_start.return_value = "process_123"

        manager = TrainingSessionManager()
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = manager.create_session(config)
        manager.start_session(session_id)
        manager.stop_session(session_id)

        assert len(manager.sessions) == 1

        count = manager.cleanup_completed_sessions()

        assert count == 1
        assert len(manager.sessions) == 0
