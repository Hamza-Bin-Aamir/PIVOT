"""Tests for POST /training/start endpoint."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.process_manager import TrainingProcessManager


class TestStartTrainingEndpoint:
    """Test suite for POST /training/start endpoint."""

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
# Training configuration
experiment:
  name: test_experiment
  seed: 42

training:
  epochs: 10
  batch_size: 4

model:
  architecture: unet
  in_channels: 1
  out_channels: 4
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

    def test_start_training_success(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test successfully starting a training process."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": str(temp_config),
                "use_uv": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "process_id" in data
        assert "status" in data
        assert "pid" in data
        assert "config_path" in data
        assert "message" in data

        # Verify values
        assert data["process_id"].startswith("train_")
        assert data["status"] == "running"
        assert data["pid"] is not None
        assert data["config_path"] == str(temp_config)
        assert "started successfully" in data["message"].lower()

    def test_start_training_config_not_found(self, client: TestClient) -> None:
        """Test starting training with non-existent config file."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": "nonexistent/config.yaml",
                "use_uv": True,
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
        assert "nonexistent/config.yaml" in data["detail"]

    def test_start_training_invalid_config_extension(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Test starting training with non-YAML config file."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("invalid config")

        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": str(config_file),
                "use_uv": True,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "yaml" in data["detail"].lower()
        assert ".txt" in data["detail"]

    def test_start_training_with_environment_vars(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test starting training with custom environment variables."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": str(temp_config),
                "use_uv": True,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "PYTHONPATH": "/custom/path",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["process_id"].startswith("train_")
        assert data["status"] == "running"

    def test_start_training_without_uv(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test starting training without uv run."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": str(temp_config),
                "use_uv": False,
                "python_executable": "python3",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["process_id"].startswith("train_")
        assert data["status"] == "running"

    def test_start_training_response_structure(
        self, client: TestClient, temp_config: Path
    ) -> None:
        """Test that response has correct structure and types."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "config_path": str(temp_config),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        required_fields = {"process_id", "status", "pid", "config_path", "message"}
        assert set(data.keys()) == required_fields

        # Check types
        assert isinstance(data["process_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["pid"], (int, type(None)))
        assert isinstance(data["config_path"], str)
        assert isinstance(data["message"], str)

    def test_start_multiple_training_processes(
        self, client: TestClient, temp_config: Path, tmp_path: Path
    ) -> None:
        """Test starting multiple training processes."""
        # Create second config
        config2 = tmp_path / "train2.yaml"
        config2.write_text(
            """
experiment:
  name: test_experiment_2
  seed: 43
"""
        )

        # Start first process
        response1 = client.post(
            "/api/v1/training/start",
            json={"config_path": str(temp_config)},
        )
        assert response1.status_code == 200
        process_id_1 = response1.json()["process_id"]

        # Sleep to ensure different timestamp
        time.sleep(1.1)

        # Start second process
        response2 = client.post(
            "/api/v1/training/start",
            json={"config_path": str(config2)},
        )
        assert response2.status_code == 200
        process_id_2 = response2.json()["process_id"]

        # Verify they have different IDs
        assert process_id_1 != process_id_2
        assert response2.json()["config_path"] == str(config2)
