"""Tests for status endpoint."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestStatusEndpoint:
    """Test suite for GET /status endpoint."""

    def test_status_endpoint_no_sessions(self):
        """Test status endpoint with no active sessions."""
        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()

        assert data["active_sessions"] == 0
        assert data["total_sessions"] == 0
        assert data["running_sessions"] == 0
        assert "timestamp" in data
        assert data["sessions"] == []

    def test_status_endpoint_with_idle_sessions(self, tmp_path: Path):
        """Test status endpoint with idle sessions."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create idle sessions
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_manager.create_session(config)

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()

        assert data["total_sessions"] == 1
        assert data["active_sessions"] == 0  # IDLE is not active
        assert data["running_sessions"] == 0
        assert len(data["sessions"]) == 1

    def test_status_endpoint_with_running_sessions(self, tmp_path: Path):
        """Test status endpoint with running sessions."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_id = session_manager.create_session(config)

        # Mock the process manager to simulate starting
        with patch.object(
            session_manager._process_manager, "start_process", return_value="proc123"
        ):
            session_manager.start_session(session_id)

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()

        assert data["total_sessions"] == 1
        assert data["active_sessions"] == 1
        assert data["running_sessions"] == 1
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["status"] == "running"

    def test_status_endpoint_multiple_sessions(self, tmp_path: Path):
        """Test status endpoint with multiple sessions in different states."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create idle session
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_manager.create_session(config1)

        # Small delay to ensure different session ID
        time.sleep(1.1)

        # Create running session
        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Mock the process manager to simulate starting
        with patch.object(
            session_manager._process_manager, "start_process", return_value="proc456"
        ):
            session_manager.start_session(session_id2)

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()

        assert data["total_sessions"] == 2
        assert data["active_sessions"] == 1  # Only running session
        assert data["running_sessions"] == 1

    def test_status_endpoint_session_data_structure(self, tmp_path: Path):
        """Test that session data has correct structure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(
            config_path=config_file,
            experiment_name="test_exp",
            description="Test description",
            tags=["tag1", "tag2"],
        )
        session_manager.create_session(config)

        response = client.get("/api/v1/status")
        data = response.json()

        session = data["sessions"][0]
        assert "session_id" in session
        assert "experiment_name" in session
        assert session["experiment_name"] == "test_exp"
        assert session["description"] == "Test description"
        assert session["tags"] == ["tag1", "tag2"]
        assert "status" in session
    def test_status_endpoint_without_session_manager(self):
        """Test status endpoint when session manager not initialized."""
        from src.api.routers import status as status_module

        # Clear session manager
        status_module._session_manager = None

        app = create_app()  # No session manager
        client = TestClient(app)

        response = client.get("/api/v1/status")

        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]
        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]

    def test_status_response_timestamp_format(self, tmp_path: Path):
        """Test that timestamp is in ISO format."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get("/api/v1/status")
        data = response.json()

        # Verify timestamp is ISO format
        from datetime import datetime

        try:
            datetime.fromisoformat(data["timestamp"])
        except ValueError as e:
            raise AssertionError("Timestamp is not in ISO format") from e

    def test_status_endpoint_multiple_calls(self, tmp_path: Path):
        """Test that multiple calls to status endpoint work correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # First call
        response1 = client.get("/api/v1/status")
        assert response1.status_code == 200

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="exp1")
        session_manager.create_session(config)

        # Second call should show updated count
        response2 = client.get("/api/v1/status")
        assert response2.status_code == 200
        assert response2.json()["total_sessions"] == 1
