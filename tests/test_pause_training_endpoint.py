"""Tests for POST /training/pause endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.process_manager import TrainingProcessManager


class TestPauseTrainingEndpoint:
    """Test suite for POST /training/pause endpoint."""

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

    def test_pause_training_success(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test successfully pausing a running training process."""
        # Start a process first
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        assert start_response.status_code == 200
        process_id = start_response.json()["process_id"]

        # Pause the process
        pause_response = client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )

        assert pause_response.status_code == 200
        data = pause_response.json()

        # Verify response structure
        assert "process_id" in data
        assert "status" in data
        assert "message" in data

        # Verify values
        assert data["process_id"] == process_id
        assert data["status"] == "paused"
        assert "paused successfully" in data["message"].lower()

    def test_pause_training_process_not_found(self, client: TestClient) -> None:
        """Test pausing non-existent process."""
        response = client.post(
            "/api/v1/training/pause",
            json={"process_id": "nonexistent_process"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_pause_training_invalid_state(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test pausing a process that's not in RUNNING state."""
        # Start and pause a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # First pause should succeed
        pause_response = client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )
        assert pause_response.status_code == 200

        # Second pause should fail (already paused)
        pause_again_response = client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )
        assert pause_again_response.status_code == 400
        assert "cannot pause" in pause_again_response.json()["detail"].lower()

    def test_pause_training_response_structure(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test that response has correct structure and types."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # Pause the process
        response = client.post(
            "/api/v1/training/pause",
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
