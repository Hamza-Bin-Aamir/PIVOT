"""Tests for Server-Sent Events (SSE) endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routers.sse import format_sse_message
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestSSEEndpoints:
    """Test suite for SSE endpoints."""

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

    def test_sse_format_function(self) -> None:
        """Test the SSE message formatting function."""
        result = format_sse_message("test_event", {"key": "value"})

        assert "event: test_event\n" in result
        assert "data: " in result
        assert '"key": "value"' in result
        assert result.endswith("\n\n")

    def test_sse_format_with_complex_data(self) -> None:
        """Test SSE formatting with complex nested data."""
        data = {
            "session_id": "test_123",
            "metrics": {"loss": 0.5, "accuracy": 0.95},
            "status": "running",
        }
        result = format_sse_message("metrics", data)

        assert "event: metrics\n" in result
        assert "data: " in result
        assert '"session_id": "test_123"' in result
        assert result.endswith("\n\n")

    def test_sse_training_endpoint_exists(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test that SSE training endpoint exists and is accessible."""
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Test endpoint exists and can be called
        # We use a mock to avoid infinite stream consumption
        from unittest.mock import patch

        with patch(
            "src.api.routers.sse.training_event_generator"
        ) as mock_generator:
            # Mock returns empty async generator
            async def empty_gen():
                yield format_sse_message("heartbeat", {"status": "alive"})

            mock_generator.return_value = empty_gen()

            response = client.get(f"/api/v1/sse/training/{session_id}")
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_sse_training_endpoint_invalid_session(
        self, client: TestClient
    ) -> None:
        """Test SSE endpoint with invalid session ID works."""
        from unittest.mock import patch

        with patch(
            "src.api.routers.sse.training_event_generator"
        ) as mock_generator:
            # Mock returns empty async generator
            async def empty_gen():
                yield format_sse_message("error", {"message": "Invalid session"})

            mock_generator.return_value = empty_gen()

            response = client.get("/api/v1/sse/training/invalid_session")
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_sse_broadcast_endpoint_exists(self, client: TestClient) -> None:
        """Test that SSE broadcast endpoint exists and is accessible."""
        from unittest.mock import patch

        with patch(
            "src.api.routers.sse.broadcast_event_generator"
        ) as mock_generator:
            # Mock returns empty async generator
            async def empty_gen():
                yield format_sse_message("heartbeat", {"status": "alive"})

            mock_generator.return_value = empty_gen()

            response = client.get("/api/v1/sse/broadcast")
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_sse_endpoints_route_registration(self, client: TestClient) -> None:
        """Test that SSE endpoints are properly registered in the router."""
        # Invalid path should return 404
        response = client.get("/api/v1/sse/nonexistent")
        assert response.status_code == 404

    def test_sse_manager_not_initialized(self, tmp_path: Path) -> None:
        """Test SSE behavior when session manager is not initialized."""
        # Create app without session manager initialization
        app = create_app()
        client = TestClient(app)

        # Endpoint exists but manager is None, which would cause issues
        # when the generator tries to access it
        from unittest.mock import patch

        with patch(
            "src.api.routers.sse.training_event_generator"
        ) as mock_generator:
            # Simulate what happens when manager is None
            async def error_gen():
                yield format_sse_message(
                    "error", {"message": "Session manager not initialized"}
                )

            mock_generator.return_value = error_gen()

            response = client.get("/api/v1/sse/training/test_session")
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )
