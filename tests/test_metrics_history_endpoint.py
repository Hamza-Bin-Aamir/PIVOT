"""Tests for metrics history endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import SessionConfig, TrainingSessionManager, create_app


class TestMetricsHistoryEndpoint:
    """Test suite for GET /metrics/{session_id}/history/{metric_type}/{metric_name} endpoint."""

    def test_history_endpoint_valid_train_metric(self, tmp_path: Path):
        """Test history endpoint with valid train metric."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        # Create a session
        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/train/loss"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_exp"
        assert data["metric_name"] == "loss"
        assert data["metric_type"] == "train"
        assert "history" in data
        assert "total_points" in data
        assert isinstance(data["history"], list)

    def test_history_endpoint_valid_val_metric(self, tmp_path: Path):
        """Test history endpoint with valid validation metric."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/val/accuracy"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metric_name"] == "accuracy"
        assert data["metric_type"] == "val"

    def test_history_endpoint_invalid_metric_type(self, tmp_path: Path):
        """Test history endpoint with invalid metric type."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="test_exp")
        session_id = session_manager.create_session(config)

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/invalid/loss"
        )

        assert response.status_code == 400
        assert "invalid metric_type" in response.json()["detail"].lower()

    def test_history_endpoint_invalid_session(self, tmp_path: Path):
        """Test history endpoint with invalid session ID."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        response = client.get(
            "/api/v1/metrics/invalid_session_id/history/train/loss"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_history_endpoint_without_session_manager(self):
        """Test history endpoint when session manager not initialized."""
        from src.api.routers import metrics as metrics_module

        # Clear session manager
        metrics_module._session_manager = None

        app = create_app()  # No session manager
        client = TestClient(app)

        response = client.get(
            "/api/v1/metrics/some_session_id/history/train/loss"
        )

        assert response.status_code == 500
        assert "Session manager not initialized" in response.json()["detail"]

    def test_history_response_structure(self, tmp_path: Path):
        """Test that history response has correct structure."""
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

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/train/loss"
        )
        data = response.json()

        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "metric_name" in data
        assert "metric_type" in data
        assert "history" in data
        assert "total_points" in data

        # Verify types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["metric_name"], str)
        assert isinstance(data["metric_type"], str)
        assert isinstance(data["history"], list)
        assert isinstance(data["total_points"], int)

    def test_history_empty_for_new_session(self, tmp_path: Path):
        """Test that new sessions return empty history."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="new_exp")
        session_id = session_manager.create_session(config)

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/train/loss"
        )
        data = response.json()

        # New session should have no history yet
        assert data["history"] == []
        assert data["total_points"] == 0

    def test_history_multiple_metrics(self, tmp_path: Path):
        """Test history endpoint for different metrics."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="multi_exp")
        session_id = session_manager.create_session(config)

        # Query different metrics
        response_loss = client.get(
            f"/api/v1/metrics/{session_id}/history/train/loss"
        )
        response_acc = client.get(
            f"/api/v1/metrics/{session_id}/history/val/accuracy"
        )

        assert response_loss.status_code == 200
        assert response_acc.status_code == 200

        assert response_loss.json()["metric_name"] == "loss"
        assert response_loss.json()["metric_type"] == "train"

        assert response_acc.json()["metric_name"] == "accuracy"
        assert response_acc.json()["metric_type"] == "val"

    def test_history_total_points_matches_history_length(self, tmp_path: Path):
        """Test that total_points matches history list length."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        session_manager = TrainingSessionManager()
        app = create_app(session_manager=session_manager)
        client = TestClient(app)

        config = SessionConfig(config_path=config_file, experiment_name="count_exp")
        session_id = session_manager.create_session(config)

        response = client.get(
            f"/api/v1/metrics/{session_id}/history/train/loss"
        )
        data = response.json()

        assert data["total_points"] == len(data["history"])
