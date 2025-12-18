"""Tests for GET /graphs/loss endpoint."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestLossGraphEndpoint:
    """Test suite for GET /graphs/loss endpoint."""

    @pytest.fixture
    def session_manager(self, tmp_path: Path) -> TrainingSessionManager:
        """Create a shared session manager.

        Args:
            tmp_path: Pytest temporary directory

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
            FastAPI test client
        """
        app = create_app(session_manager=session_manager)
        return TestClient(app)

    def test_loss_graph_defaults_to_latest_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that /graphs/loss defaults to latest session."""
        # Create config files
        config1 = tmp_path / "config1.yaml"
        config1.write_text("experiment:\n  name: exp1\n")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("experiment:\n  name: exp2\n")

        # Create two sessions
        session_manager.create_session(
            SessionConfig(config_path=config1, experiment_name="experiment1")
        )
        time.sleep(1.1)
        session2_id = session_manager.create_session(
            SessionConfig(config_path=config2, experiment_name="experiment2")
        )

        response = client.get("/api/v1/graphs/loss")

        assert response.status_code == 200
        data = response.json()

        # Should return latest session (session2)
        assert data["session_id"] == session2_id
        assert data["experiment_name"] == "experiment2"

    def test_loss_graph_by_session_id(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test getting loss graph for specific session."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")

        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/loss/{session_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_experiment"

    def test_loss_graph_response_structure(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that response has correct structure and types."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")

        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/loss/{session_id}")

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        required_fields = {
            "session_id",
            "experiment_name",
            "train_loss",
            "val_loss",
            "total_points",
        }
        assert set(data.keys()) == required_fields

        # Check types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["experiment_name"], str)
        assert isinstance(data["train_loss"], list)
        assert isinstance(data["val_loss"], list)
        assert isinstance(data["total_points"], int)

    def test_loss_graph_empty_data(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test loss graph returns empty lists when no data available."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")

        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/loss/{session_id}")

        assert response.status_code == 200
        data = response.json()

        # Currently returns empty data (will be populated when metrics integrated)
        assert data["train_loss"] == []
        assert data["val_loss"] == []
        assert data["total_points"] == 0

    def test_loss_graph_invalid_session(self, client: TestClient) -> None:
        """Test loss graph with invalid session ID."""
        response = client.get("/api/v1/graphs/loss/invalid_session")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_loss_graph_no_sessions(self, client: TestClient) -> None:
        """Test loss graph when no sessions exist."""
        response = client.get("/api/v1/graphs/loss")

        assert response.status_code == 404
        data = response.json()
        assert "no sessions found" in data["detail"].lower()

    def test_loss_graph_data_point_structure(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that data points have correct x,y structure."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")

        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/loss/{session_id}")

        assert response.status_code == 200
        data = response.json()

        # Even though empty, structure should be correct
        assert isinstance(data["train_loss"], list)
        assert isinstance(data["val_loss"], list)

        # When populated, each item should have x and y fields
        # This structure is ready for: [{"x": 1, "y": 0.523}, {"x": 2, "y": 0.456}]
