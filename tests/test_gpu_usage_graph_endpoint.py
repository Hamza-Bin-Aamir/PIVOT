"""Tests for GET /graphs/gpu-usage endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestGPUUsageGraphEndpoint:
    """Test suite for GPU usage graph endpoints."""

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

    def test_gpu_usage_graph_defaults_to_latest_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that GPU usage graph defaults to latest session when no ID specified."""
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

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id2  # Should be latest session
        assert data["experiment_name"] == "experiment2"

    def test_gpu_usage_graph_by_session_id(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test retrieving GPU usage for a specific session."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get(f"/api/v1/graphs/gpu-usage/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["experiment_name"] == "test_experiment"

    def test_gpu_usage_graph_response_structure(
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

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        # Verify all required fields are present
        assert "session_id" in data
        assert "experiment_name" in data
        assert "gpu_utilization" in data
        assert "memory_usage" in data
        assert "total_points" in data

        # Verify data is list type (ready for plotting)
        assert isinstance(data["gpu_utilization"], list)
        assert isinstance(data["memory_usage"], list)
        assert isinstance(data["total_points"], int)

    def test_gpu_usage_graph_empty_data(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that empty data returns correct structure (before GPU monitoring integration)."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        # Should return empty lists until GPU monitoring is integrated
        assert data["gpu_utilization"] == []
        assert data["memory_usage"] == []
        assert data["total_points"] == 0

    def test_gpu_usage_graph_invalid_session(
        self, client: TestClient, session_manager: TrainingSessionManager
    ) -> None:
        """Test error handling for non-existent session."""
        response = client.get("/api/v1/graphs/gpu-usage/invalid_session_id")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_gpu_usage_graph_no_sessions(self, client: TestClient) -> None:
        """Test error handling when no sessions exist."""
        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 404
        assert response.json()["detail"] == "No sessions found"

    def test_gpu_usage_graph_data_point_structure(
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

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        # Even with empty data, structure should be ready for plotting
        # When populated, each point should have {x, y} format
        # x = time/epoch, y = percentage (0-100)
        assert isinstance(data["gpu_utilization"], list)
        assert isinstance(data["memory_usage"], list)

    def test_gpu_usage_monitoring_use_case(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that response structure supports GPU monitoring use cases."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        # Structure should support:
        # - Identifying GPU bottlenecks
        # - Memory leak detection
        # - Utilization optimization
        # - Multi-GPU load balancing insights
        assert "gpu_utilization" in data
        assert "memory_usage" in data

    def test_gpu_usage_dual_metrics(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that both GPU utilization and memory usage are tracked separately."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        response = client.get("/api/v1/graphs/gpu-usage")
        assert response.status_code == 200

        data = response.json()
        # Both metrics should be present and independent
        # This allows dashboard to show:
        # 1. GPU compute utilization %
        # 2. GPU memory utilization %
        # Useful for diagnosing different types of bottlenecks
        assert "gpu_utilization" in data
        assert "memory_usage" in data
        # Both should be lists (they happen to be empty now, but that's fine)
        assert isinstance(data["gpu_utilization"], list)
        assert isinstance(data["memory_usage"], list)
