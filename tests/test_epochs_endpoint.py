"""Tests for epochs endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestEpochsEndpoint:
    """Test suite for GET /epochs/{session_id} endpoint."""

    def test_epochs_endpoint_valid_session(self, tmp_path: Path):
        """Test epochs endpoint with valid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/epochs/{session_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_exp"
        assert "total_epochs" in data
        assert "completed_epochs" in data
        assert "current_epoch" in data
        assert "epochs" in data
        assert isinstance(data["epochs"], list)

    def test_epochs_endpoint_invalid_session(self, tmp_path: Path):
        """Test epochs endpoint with invalid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get("/api/v1/epochs/invalid_session_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_epochs_endpoint_without_session_manager(self):
        """Test epochs endpoint when session manager not initialized."""
        from src.api.routers import epochs as epochs_module

        # Clear session manager
        epochs_module._session_manager = None

        app = create_app()  # No session manager
        client = TestClient(app)

        response = client.get("/api/v1/epochs/some_session_id")

        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]

    def test_epochs_response_structure(self, tmp_path: Path):
        """Test that epochs response has correct structure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(
            config_path=config_file,
            experiment_name="structured_exp",
            description="Test description",
        )
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/epochs/{session_id}")
        data = response.json()

        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "total_epochs" in data
        assert "completed_epochs" in data
        assert "current_epoch" in data
        assert "epochs" in data

        # Verify types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["total_epochs"], int)
        assert isinstance(data["completed_epochs"], int)
        assert isinstance(data["current_epoch"], int)
        assert isinstance(data["epochs"], list)

    def test_epochs_endpoint_multiple_sessions(self, tmp_path: Path):
        """Test epochs endpoint with multiple sessions."""
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
        response1 = client.get(f"/api/v1/epochs/{session_id1}")
        assert response1.status_code == 200
        assert response1.json()["experiment_name"] == "exp1"

        # Query second session
        response2 = client.get(f"/api/v1/epochs/{session_id2}")
        assert response2.status_code == 200
        assert response2.json()["experiment_name"] == "exp2"

    def test_epochs_endpoint_empty_epochs_list(self, tmp_path: Path):
        """Test that new sessions return empty epochs list."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="new_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/epochs/{session_id}")
        data = response.json()

        # New session should have no epochs yet
        assert data["epochs"] == []
        assert data["completed_epochs"] == 0
        assert data["current_epoch"] == 0
