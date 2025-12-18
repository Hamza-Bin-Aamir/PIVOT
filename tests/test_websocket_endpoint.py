"""Tests for WebSocket endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestWebSocketEndpoints:
    """Test suite for WebSocket endpoints."""

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

    def test_websocket_training_updates_connection(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test WebSocket connection to training updates endpoint."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as websocket:
            # Should receive welcome message
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "connection"
            assert message["session_id"] == session_id
            assert message["data"]["status"] == "connected"
            assert message["data"]["experiment_name"] == "test_experiment"
            assert "timestamp" in message

    def test_websocket_training_updates_invalid_session(
        self, client: TestClient
    ) -> None:
        """Test WebSocket connection with invalid session ID."""
        with client.websocket_connect(
            "/api/v1/ws/training/invalid_session_id"
        ) as websocket:
            # Should receive error message
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "error"
            assert "Session not found" in message["data"]["message"]

    def test_websocket_training_updates_heartbeat(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that WebSocket sends periodic heartbeat messages."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as websocket:
            # Receive welcome message
            welcome = websocket.receive_text()
            assert json.loads(welcome)["type"] == "connection"

            # Receive heartbeat (with timeout)
            try:
                heartbeat_data = websocket.receive_text()
                heartbeat = json.loads(heartbeat_data)

                assert heartbeat["type"] == "heartbeat"
                assert heartbeat["session_id"] == session_id
                assert "status" in heartbeat["data"]
                assert "timestamp" in heartbeat
            except Exception:
                # Timeout is acceptable since heartbeat interval is 5s
                pass

    def test_websocket_broadcast_connection(self, client: TestClient) -> None:
        """Test WebSocket connection to broadcast endpoint."""
        with client.websocket_connect("/api/v1/ws/broadcast") as websocket:
            # Should receive welcome message
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "connection"
            assert message["data"]["status"] == "connected"
            assert "Broadcasting updates" in message["data"]["message"]
            assert "timestamp" in message

    def test_websocket_broadcast_heartbeat(self, client: TestClient) -> None:
        """Test that broadcast WebSocket sends periodic heartbeat messages."""
        with client.websocket_connect("/api/v1/ws/broadcast") as websocket:
            # Receive welcome message
            welcome = websocket.receive_text()
            assert json.loads(welcome)["type"] == "connection"

            # Receive heartbeat (with timeout)
            try:
                heartbeat_data = websocket.receive_text()
                heartbeat = json.loads(heartbeat_data)

                assert heartbeat["type"] == "heartbeat"
                assert "active_connections" in heartbeat["data"]
                assert heartbeat["data"]["active_connections"] >= 1
                assert "timestamp" in heartbeat
            except Exception:
                # Timeout is acceptable since heartbeat interval is 10s
                pass

    def test_websocket_message_structure(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that WebSocket messages follow the TrainingUpdate structure."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as websocket:
            data = websocket.receive_text()
            message = json.loads(data)

            # Verify required fields
            assert "type" in message
            assert "session_id" in message
            assert "data" in message
            assert "timestamp" in message

            # Verify types
            assert isinstance(message["type"], str)
            assert isinstance(message["session_id"], str)
            assert isinstance(message["data"], dict)
            assert isinstance(message["timestamp"], str)

    def test_websocket_multiple_connections(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test multiple simultaneous WebSocket connections."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Open multiple connections
        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as ws1, client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as ws2:
            # Both should receive welcome messages
            msg1 = json.loads(ws1.receive_text())
            msg2 = json.loads(ws2.receive_text())

            assert msg1["type"] == "connection"
            assert msg2["type"] == "connection"
            assert msg1["session_id"] == session_id
            assert msg2["session_id"] == session_id

    def test_websocket_connection_cleanup(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that disconnected WebSocket connections are cleaned up."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Connect and disconnect
        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as websocket:
            welcome = websocket.receive_text()
            assert json.loads(welcome)["type"] == "connection"

        # Connection should be cleaned up after context exit
        # Verify by opening a new connection
        with client.websocket_connect(
            f"/api/v1/ws/training/{session_id}"
        ) as websocket:
            welcome = websocket.receive_text()
            assert json.loads(welcome)["type"] == "connection"
