"""Tests for GET /graphs/learning-rate endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestLearningRateGraphEndpoint:
    """Test suite for learning rate graph endpoints."""

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

    def test_learning_rate_graph_defaults_to_latest_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that LR graph defaults to latest session when no ID specified."""
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

        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id2  # Should be latest session
        assert data["experiment_name"] == "experiment2"

    def test_learning_rate_graph_by_session_id(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test retrieving learning rate for a specific session."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/learning-rate/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_experiment"

    def test_learning_rate_graph_response_structure(
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

        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 200

        data = response.json()
        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "lr_data" in data
        assert "total_points" in data

        # Verify data is list type (ready for plotting)
        assert isinstance(data["lr_data"], list)
        assert isinstance(data["total_points"], int)

    def test_learning_rate_graph_empty_data(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that empty data returns correct structure (before LR tracking integration)."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 200

        data = response.json()
        # Should return empty list until LR tracking is integrated
        assert data["lr_data"] == []
        assert data["total_points"] == 0

    def test_learning_rate_graph_invalid_session(
        self, client: TestClient, session_manager: TrainingSessionManager
    ) -> None:
        """Test error handling for non-existent session."""
        response = client.get("/api/v1/graphs/learning-rate/invalid_session_id")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_learning_rate_graph_no_sessions(self, client: TestClient) -> None:
        """Test error handling when no sessions exist."""
        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 404
        assert response.json()["detail"] == "No sessions found"

    def test_learning_rate_graph_data_point_structure(
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

        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 200

        data = response.json()
        # Even with empty data, structure should be ready for plotting
        # When populated, each point should have {x, y} format
        # x = epoch/step, y = learning rate value
        assert isinstance(data["lr_data"], list)

    def test_learning_rate_visualization_use_case(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that response structure supports LR schedule visualization."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/learning-rate")
        assert response.status_code == 200

        data = response.json()
        # Structure should support visualizing:
        # - Constant LR
        # - Step decay
        # - Exponential decay
        # - Warmup schedules
        # - Cyclical LR
        # All via simple {x: epoch, y: lr_value} points
        assert "lr_data" in data
        assert isinstance(data["lr_data"], list)
