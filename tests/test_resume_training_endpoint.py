"""Tests for POST /training/resume endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.process_manager import TrainingProcessManager


class TestResumeTrainingEndpoint:
    """Test suite for POST /training/resume endpoint."""

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

    def test_resume_training_success(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test successfully resuming a paused training process."""
        # Start a process
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

        # Resume the process
        resume_response = client.post(
            "/api/v1/training/resume",
            json={"process_id": process_id},
        )

        assert resume_response.status_code == 200
        data = resume_response.json()

        # Verify response structure
        assert "process_id" in data
        assert "status" in data
        assert "message" in data

        # Verify values
        assert data["process_id"] == process_id
        assert data["status"] == "running"
        assert "resumed successfully" in data["message"].lower()

    def test_resume_training_process_not_found(self, client: TestClient) -> None:
        """Test resuming non-existent process."""
        response = client.post(
            "/api/v1/training/resume",
            json={"process_id": "nonexistent_process"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_resume_training_invalid_state(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test resuming a process that's not in PAUSED state."""
        # Start a process (but don't pause it)
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # Try to resume without pausing first
        resume_response = client.post(
            "/api/v1/training/resume",
            json={"process_id": process_id},
        )
        assert resume_response.status_code == 400
        assert "cannot resume" in resume_response.json()["detail"].lower()

    def test_resume_training_response_structure(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test that response has correct structure and types."""
        # Start and pause a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )

        # Resume the process
        response = client.post(
            "/api/v1/training/resume",
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

    def test_pause_resume_cycle(self, client: TestClient, temp_config: Path) -> None:
        """Test multiple pause/resume cycles on the same process."""
        # Start a process
        start_response = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        process_id = start_response.json()["process_id"]

        # First cycle: pause and resume
        pause1 = client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )
        assert pause1.status_code == 200
        assert pause1.json()["status"] == "paused"

        resume1 = client.post(
            "/api/v1/training/resume",
            json={"process_id": process_id},
        )
        assert resume1.status_code == 200
        assert resume1.json()["status"] == "running"

        # Second cycle: pause and resume again
        pause2 = client.post(
            "/api/v1/training/pause",
            json={"process_id": process_id},
        )
        assert pause2.status_code == 200
        assert pause2.json()["status"] == "paused"

        resume2 = client.post(
            "/api/v1/training/resume",
            json={"process_id": process_id},
        )
        assert resume2.status_code == 200
        assert resume2.json()["status"] == "running"
