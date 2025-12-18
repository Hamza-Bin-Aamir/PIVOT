"""Tests for GET /graphs/metrics endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestMetricsGraphEndpoint:
    """Test suite for metrics graph endpoints."""

    @pytest.fixture
    def session_manager(self) -> TrainingSessionManager:
        """Create a TrainingSessionManager instance shared between tests and client.

        Returns:
            TrainingSessionManager instance
        """
        return TrainingSessionManager()

    @pytest.fixture
    def client(self, session_manager: TrainingSessionManager) -> TestClient:
        """Create test client with session manager.

        Args:
            session_manager: Shared session manager instance

        Returns:
            TestClient instance for making requests
        """
        app = create_app(session_manager=session_manager)
        return TestClient(app)

    def test_metrics_graph_defaults_to_latest_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that metrics graph defaults to latest session when no ID specified."""
        # Create two sessions
        config1 = tmp_path / "config1.yaml"
        config1.write_text("experiment:\n  name: test1\n")
        session_manager.create_session(
            SessionConfig(config_path=config1, experiment_name="experiment1")
        )

        config2 = tmp_path / "config2.yaml"
        config2.write_text("experiment:\n  name: test2\n")
        session_id2 = session_manager.create_session(
            SessionConfig(config_path=config2, experiment_name="experiment2")
        )

        response = client.get("/api/v1/graphs/metrics/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id2  # Should be latest session
        assert data["experiment_name"] == "experiment2"
        assert data["metric_name"] == "accuracy"

    def test_metrics_graph_by_session_id(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test retrieving metrics for a specific session."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/metrics/dice/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_experiment"
        assert data["metric_name"] == "dice"

    def test_metrics_graph_response_structure(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that response has correct structure for dashboard consumption."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/metrics/iou")
        assert response.status_code == 200

        data = response.json()
        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "metric_name" in data
        assert "train_data" in data
        assert "val_data" in data
        assert "total_points" in data

        # Verify data is list type (ready for plotting)
        assert isinstance(data["train_data"], list)
        assert isinstance(data["val_data"], list)
        assert isinstance(data["total_points"], int)

    def test_metrics_graph_empty_data(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that empty data returns correct structure (before metrics integration)."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/metrics/f1_score")
        assert response.status_code == 200

        data = response.json()
        # Should return empty lists until metrics collection is integrated
        assert data["train_data"] == []
        assert data["val_data"] == []
        assert data["total_points"] == 0

    def test_metrics_graph_invalid_session(
        self, client: TestClient, session_manager: TrainingSessionManager
    ) -> None:
        """Test error handling for non-existent session."""
        response = client.get("/api/v1/graphs/metrics/accuracy/invalid_session_id")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_metrics_graph_no_sessions(self, client: TestClient) -> None:
        """Test error handling when no sessions exist."""
        response = client.get("/api/v1/graphs/metrics/accuracy")
        assert response.status_code == 404
        assert response.json()["detail"] == "No sessions found"

    def test_metrics_graph_data_point_structure(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that DataPoint structure is optimized for dashboard plotting libraries."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/metrics/precision")
        assert response.status_code == 200

        data = response.json()
        # Even with empty data, structure should be ready for plotting
        # When populated, each point should have {x, y} format
        # This is compatible with most charting libraries (Chart.js, Plotly, etc.)
        assert isinstance(data["train_data"], list)
        assert isinstance(data["val_data"], list)

    def test_metrics_graph_different_metric_names(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that different metric names are handled correctly."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Test various common metric names
        metric_names = ["accuracy", "precision", "recall", "f1_score", "dice", "iou"]

        for metric_name in metric_names:
            response = client.get(f"/api/v1/graphs/metrics/{metric_name}")
            assert response.status_code == 200

            data = response.json()
            assert data["metric_name"] == metric_name

    def test_metrics_graph_with_underscores_and_special_chars(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test metric names with underscores and special characters."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Test metric names with underscores (common in ML)
        response = client.get("/api/v1/graphs/metrics/mean_absolute_error")
        assert response.status_code == 200
        assert response.json()["metric_name"] == "mean_absolute_error"
