"""Tests for latest metrics endpoint."""

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestLatestMetricsEndpoint:
    """Test suite for GET /metrics/{session_id}/latest endpoint."""

    def test_latest_metrics_endpoint_valid_session(self, tmp_path: Path):
        """Test latest metrics endpoint with valid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}/latest")

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_exp"
        assert "epoch" in data
        assert "step" in data
        assert "train_metrics" in data
        assert "val_metrics" in data
        assert "timestamp" in data

    def test_latest_metrics_endpoint_invalid_session(self, tmp_path: Path):
        """Test latest metrics endpoint with invalid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get("/api/v1/metrics/invalid_session_id/latest")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_latest_metrics_endpoint_without_session_manager(self):
        """Test latest metrics endpoint when session manager not initialized."""
        from src.api.routers import metrics as metrics_module

        # Clear session manager
        metrics_module._session_manager = None

        app = create_app()  # No session manager
        client = TestClient(app)

        response = client.get("/api/v1/metrics/some_session_id/latest")

        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]

    def test_latest_metrics_response_structure(self, tmp_path: Path):
        """Test that latest metrics response has correct structure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(
            config_path=config_file,
            experiment_name="structured_exp",
        )
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}/latest")
        data = response.json()

        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "epoch" in data
        assert "step" in data
        assert "train_metrics" in data
        assert "val_metrics" in data
        assert "timestamp" in data

        # Verify types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["epoch"], int)
        assert isinstance(data["step"], int)
        assert isinstance(data["train_metrics"], dict)
        assert isinstance(data["val_metrics"], dict)
        assert isinstance(data["timestamp"], str)

    def test_latest_metrics_timestamp_format(self, tmp_path: Path):
        """Test that timestamp is in ISO format."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="time_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}/latest")
        data = response.json()

        # Verify timestamp is ISO format
        try:
            datetime.fromisoformat(data["timestamp"])
        except ValueError as e:
            raise AssertionError("Timestamp is not in ISO format") from e

    def test_latest_metrics_empty_for_new_session(self, tmp_path: Path):
        """Test that new sessions return empty metrics."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="new_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}/latest")
        data = response.json()

        # New session should have no metrics yet
        assert data["train_metrics"] == {}
        assert data["val_metrics"] == {}
        assert data["epoch"] == 0
        assert data["step"] == 0

    def test_latest_metrics_multiple_sessions(self, tmp_path: Path):
        """Test latest metrics endpoint with multiple sessions."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create two sessions
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_id1 = session_manager.create_session(config1)

        time.sleep(1.1)

        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Query first session
        response1 = client.get(f"/api/v1/metrics/{session_id1}/latest")
        assert response1.status_code == 200
        assert response1.json()["experiment_name"] == "exp1"

        # Query second session
        response2 = client.get(f"/api/v1/metrics/{session_id2}/latest")
        assert response2.status_code == 200
        assert response2.json()["experiment_name"] == "exp2"
