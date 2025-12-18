"""Tests for metrics endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestMetricsEndpoint:
    """Test suite for GET /metrics/{session_id} endpoint."""

    def test_metrics_endpoint_valid_session(self, tmp_path: Path):
        """Test metrics endpoint with valid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_exp"
        assert "train_metrics" in data
        assert "val_metrics" in data
        assert "total_metrics" in data
        assert isinstance(data["train_metrics"], dict)
        assert isinstance(data["val_metrics"], dict)

    def test_metrics_endpoint_invalid_session(self, tmp_path: Path):
        """Test metrics endpoint with invalid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get("/api/v1/metrics/invalid_session_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_metrics_endpoint_without_session_manager(self):
        """Test metrics endpoint when session manager not initialized."""
        from src.api.routers import metrics as metrics_module

        # Clear session manager
        metrics_module._session_manager = None

        app = create_app()  # No session manager
        client = TestClient(app)

        response = client.get("/api/v1/metrics/some_session_id")

        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]

    def test_metrics_response_structure(self, tmp_path: Path):
        """Test that metrics response has correct structure."""
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

        response = client.get(f"/api/v1/metrics/{session_id}")
        data = response.json()

        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "train_metrics" in data
        assert "val_metrics" in data
        assert "total_metrics" in data

        # Verify types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["train_metrics"], dict)
        assert isinstance(data["val_metrics"], dict)
        assert isinstance(data["total_metrics"], int)

    def test_metrics_endpoint_multiple_sessions(self, tmp_path: Path):
        """Test metrics endpoint with multiple sessions."""
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
        response1 = client.get(f"/api/v1/metrics/{session_id1}")
        assert response1.status_code == 200
        assert response1.json()["experiment_name"] == "exp1"

        # Query second session
        response2 = client.get(f"/api/v1/metrics/{session_id2}")
        assert response2.status_code == 200
        assert response2.json()["experiment_name"] == "exp2"

    def test_metrics_endpoint_empty_metrics(self, tmp_path: Path):
        """Test that new sessions return empty metrics."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="new_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}")
        data = response.json()

        # New session should have no metrics yet
        assert data["train_metrics"] == {}
        assert data["val_metrics"] == {}
        assert data["total_metrics"] == 0

    def test_metrics_total_count(self, tmp_path: Path):
        """Test that total_metrics correctly counts train and val metrics."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="count_exp")
        session_id = session_manager.create_session(config)

        response = client.get(f"/api/v1/metrics/{session_id}")
        data = response.json()

        # Verify total_metrics is sum of train and val metrics
        expected_total = len(data["train_metrics"]) + len(data["val_metrics"])
        assert data["total_metrics"] == expected_total
