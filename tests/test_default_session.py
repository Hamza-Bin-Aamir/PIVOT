"""Tests for endpoints with default (latest) session behavior."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestDefaultSessionBehavior:
    """Test suite for endpoint default session_id behavior."""

    def test_epochs_endpoint_defaults_to_latest(self, tmp_path: Path):
        """Test that epochs endpoint uses latest session when no ID provided."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create first session
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_manager.create_session(config1)

        time.sleep(1.1)

        # Create second (latest) session
        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Query without session_id should get the latest (exp2)
        response = client.get("/api/v1/epochs")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id2
        assert data["experiment_name"] == "exp2"

    def test_metrics_endpoint_defaults_to_latest(self, tmp_path: Path):
        """Test that metrics endpoint uses latest session when no ID provided."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create first session
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_manager.create_session(config1)

        time.sleep(1.1)

        # Create second (latest) session
        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Query without session_id should get the latest (exp2)
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id2
        assert data["experiment_name"] == "exp2"

    def test_latest_metrics_endpoint_defaults_to_latest(self, tmp_path: Path):
        """Test that latest metrics endpoint uses latest session when no ID provided."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create first session
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_manager.create_session(config1)

        time.sleep(1.1)

        # Create second (latest) session
        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Query without session_id should get the latest (exp2)
        response = client.get("/api/v1/metrics/latest")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id2
        assert data["experiment_name"] == "exp2"

    def test_history_endpoint_defaults_to_latest(self, tmp_path: Path):
        """Test that history endpoint uses latest session when no ID provided."""
        import time

        config_file1 = tmp_path / "config1.yaml"
        config_file1.write_text("test: config1")
        config_file2 = tmp_path / "config2.yaml"
        config_file2.write_text("test: config2")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create first session
        config1 = SessionConfig(config_path=config_file1, experiment_name="exp1")
        session_manager.create_session(config1)

        time.sleep(1.1)

        # Create second (latest) session
        config2 = SessionConfig(config_path=config_file2, experiment_name="exp2")
        session_id2 = session_manager.create_session(config2)

        # Query without session_id should get the latest (exp2)
        response = client.get("/api/v1/metrics/history/train/loss")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id2
        assert data["experiment_name"] == "exp2"

    def test_default_session_error_when_no_sessions(self, tmp_path: Path):
        """Test that endpoints return 404 when no sessions exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Test epochs endpoint
        response = client.get("/api/v1/epochs")
        assert response.status_code == 404
        assert "no sessions found" in response.json()["detail"].lower()

        # Test metrics endpoint
        response = client.get("/api/v1/metrics")
        assert response.status_code == 404
        assert "no sessions found" in response.json()["detail"].lower()

        # Test latest metrics endpoint
        response = client.get("/api/v1/metrics/latest")
        assert response.status_code == 404
        assert "no sessions found" in response.json()["detail"].lower()

        # Test history endpoint
        response = client.get("/api/v1/metrics/history/train/loss")
        assert response.status_code == 404
        assert "no sessions found" in response.json()["detail"].lower()
