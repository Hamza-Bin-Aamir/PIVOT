"""Tests for POST /training/stop endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.process_manager import TrainingProcessManager


class TestStopTrainingEndpoint:
    """Test suite for POST /training/stop endpoint."""

    @pytest.fixture
    def temp_config(self, tmp_path: Path) -> Path:
        """Create a temporary training configuration file.

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            Path to temporary config file
        """
        config_file = tmp_path / "train.yaml"
        config_file.write_text(
            """
experiment:
  name: test_experiment
  seed: 42
"""
        )
        return config_file

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client with process manager.

        Returns:
            FastAPI test client
        """
        process_manager = TrainingProcessManager()
        app = create_app(process_manager=process_manager)
        return TestClient(app)

    def test_stop_running_training_success(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test successfully stopping a running training process."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        assert start_response.status_code == 200
        process_id = start_response.json()["process_id"]

        # Stop the process
        stop_response = client.post(
            "/api/v1/training/stop",
            json={"process_id": process_id},
        )

        assert stop_response.status_code == 200
        data = stop_response.json()

        # Verify response structure
        assert "process_id" in data
        assert "status" in data
        assert "message" in data

        # Verify values
        assert data["process_id"] == process_id
        assert data["status"] == "stopped"
        assert "stopped successfully" in data["message"].lower()

    def test_stop_paused_training_success(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test successfully stopping a paused training process."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # Pause the process
        client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )

        # Stop the paused process
        stop_response = client.post(
            "/api/v1/training/stop",
            json={"process_id": process_id},
        )

        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["status"] == "stopped"

    def test_stop_training_process_not_found(self, client: TestClient) -> None:
        """Test stopping non-existent process."""
        response = client.post(
            "/api/v1/training/stop",
            json={"process_id": "nonexistent_process"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_stop_already_stopped_training(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test stopping a process that's already stopped."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # Stop it once
        stop_response = client.post(
            "/api/v1/training/stop",
            json={"process_id": process_id},
        )
        assert stop_response.status_code == 200

        # Try to stop again
        stop_again_response = client.post(
            "/api/v1/training/stop",
            json={"process_id": process_id},
        )
        assert stop_again_response.status_code == 400
        assert "cannot stop" in stop_again_response.json()["detail"].lower()

    def test_stop_training_response_structure(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test that response has correct structure and types."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # Stop the process
        response = client.post(
            "/api/v1/training/stop",
            json={"process_id": process_id},
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        required_fields = {"process_id", "status", "message"}
        assert set(data.keys()) == required_fields

        # Check types
        assert isinstance(data["process_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)
