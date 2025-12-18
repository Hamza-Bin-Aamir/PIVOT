"""Tests for notification system endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.session_manager import SessionConfig, TrainingSessionManager


class TestNotificationEndpoints:
    """Test suite for notification endpoints."""

    @pytest.fixture
    def client(self, session_manager: TrainingSessionManager) -> TestClient:
        """Create test client with session manager."""
        app = create_app(session_manager=session_manager)
        return TestClient(app)

    @pytest.fixture
    def session_manager(self, tmp_path: Path) -> TrainingSessionManager:
        """Create a session manager for testing."""
        return TrainingSessionManager()

    def test_create_notification_success(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test creating a notification successfully."""
        # Create a session first
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Create notification
        notification_data = {
            "session_id": session_id,
            "type": "info",
            "priority": "medium",
            "title": "Training Started",
            "message": "Training session has started successfully",
            "metadata": {"epoch": 1},
        }

        response = client.post("/api/v1/notifications/", json=notification_data)

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == session_id
        assert data["type"] == "info"
        assert data["priority"] == "medium"
        assert data["title"] == "Training Started"
        assert data["message"] == "Training session has started successfully"
        assert data["read"] is False
        assert "id" in data
        assert "timestamp" in data

    def test_create_notification_without_session(self, client: TestClient) -> None:
        """Test creating a system-wide notification without session ID."""
        notification_data = {
            "type": "warning",
            "priority": "high",
            "title": "System Warning",
            "message": "High memory usage detected",
        }

        response = client.post("/api/v1/notifications/", json=notification_data)

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] is None
        assert data["type"] == "warning"
        assert data["priority"] == "high"

    def test_create_notification_invalid_session(self, client: TestClient) -> None:
        """Test creating notification with invalid session ID."""
        notification_data = {
            "session_id": "invalid_session",
            "type": "info",
            "title": "Test",
            "message": "Test message",
        }

        response = client.post("/api/v1/notifications/", json=notification_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_notification_validation(self, client: TestClient) -> None:
        """Test notification creation with invalid data."""
        # Empty title
        response = client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "",
                "message": "Test message",
            },
        )
        assert response.status_code == 422

        # Empty message
        response = client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "Test",
                "message": "",
            },
        )
        assert response.status_code == 422

    def test_list_notifications(self, client: TestClient) -> None:
        """Test listing all notifications."""
        # Create some notifications
        for i in range(3):
            client.post(
                "/api/v1/notifications/",
                json={
                    "type": "info",
                    "title": f"Notification {i}",
                    "message": f"Message {i}",
                },
            )

        response = client.get("/api/v1/notifications/")

        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data
        assert "total" in data
        assert "unread" in data
        assert data["total"] >= 3
        assert data["unread"] >= 3

    def test_list_notifications_with_filters(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test listing notifications with filters."""
        # Create session
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Create notifications with different types and priorities
        client.post(
            "/api/v1/notifications/",
            json={
                "session_id": session_id,
                "type": "info",
                "priority": "low",
                "title": "Info",
                "message": "Info message",
            },
        )
        client.post(
            "/api/v1/notifications/",
            json={
                "session_id": session_id,
                "type": "warning",
                "priority": "high",
                "title": "Warning",
                "message": "Warning message",
            },
        )
        client.post(
            "/api/v1/notifications/",
            json={
                "type": "error",
                "priority": "critical",
                "title": "Error",
                "message": "Error message",
            },
        )

        # Filter by session
        response = client.get(f"/api/v1/notifications/?session_id={session_id}")
        assert response.status_code == 200
        data = response.json()
        assert all(
            n["session_id"] == session_id for n in data["notifications"]
        )

        # Filter by type
        response = client.get("/api/v1/notifications/?type=warning")
        assert response.status_code == 200
        data = response.json()
        assert all(n["type"] == "warning" for n in data["notifications"])

        # Filter by priority
        response = client.get("/api/v1/notifications/?priority=critical")
        assert response.status_code == 200
        data = response.json()
        assert all(n["priority"] == "critical" for n in data["notifications"])

    def test_get_notification(self, client: TestClient) -> None:
        """Test getting a specific notification."""
        # Create notification
        create_response = client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "Test Notification",
                "message": "Test message",
            },
        )
        notification_id = create_response.json()["id"]

        # Get notification
        response = client.get(f"/api/v1/notifications/{notification_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == notification_id
        assert data["title"] == "Test Notification"

    def test_get_notification_not_found(self, client: TestClient) -> None:
        """Test getting non-existent notification."""
        response = client.get("/api/v1/notifications/nonexistent_id")

        assert response.status_code == 404

    def test_update_notification(self, client: TestClient) -> None:
        """Test updating notification read status."""
        # Create notification
        create_response = client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "Test",
                "message": "Test message",
            },
        )
        notification_id = create_response.json()["id"]

        # Mark as read
        response = client.patch(
            f"/api/v1/notifications/{notification_id}", json={"read": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["read"] is True

        # Mark as unread
        response = client.patch(
            f"/api/v1/notifications/{notification_id}", json={"read": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["read"] is False

    def test_update_notification_not_found(self, client: TestClient) -> None:
        """Test updating non-existent notification."""
        response = client.patch(
            "/api/v1/notifications/nonexistent_id", json={"read": True}
        )

        assert response.status_code == 404

    def test_delete_notification(self, client: TestClient) -> None:
        """Test deleting a notification."""
        # Create notification
        create_response = client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "Test",
                "message": "Test message",
            },
        )
        notification_id = create_response.json()["id"]

        # Delete notification
        response = client.delete(f"/api/v1/notifications/{notification_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/api/v1/notifications/{notification_id}")
        assert get_response.status_code == 404

    def test_delete_notification_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent notification."""
        response = client.delete("/api/v1/notifications/nonexistent_id")

        assert response.status_code == 404

    def test_mark_all_read(self, client: TestClient) -> None:
        """Test marking all notifications as read."""
        # Create some notifications
        for i in range(3):
            client.post(
                "/api/v1/notifications/",
                json={
                    "type": "info",
                    "title": f"Notification {i}",
                    "message": f"Message {i}",
                },
            )

        # Mark all as read
        response = client.post("/api/v1/notifications/mark-all-read")

        assert response.status_code == 200
        data = response.json()
        assert data["marked_read"] >= 3

        # Verify all are read
        list_response = client.get("/api/v1/notifications/")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["unread"] == 0

    def test_mark_all_read_by_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test marking notifications as read for specific session."""
        # Create session
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Create notifications for session
        for i in range(2):
            client.post(
                "/api/v1/notifications/",
                json={
                    "session_id": session_id,
                    "type": "info",
                    "title": f"Session Notification {i}",
                    "message": f"Message {i}",
                },
            )

        # Create system-wide notification
        client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "System Notification",
                "message": "System message",
            },
        )

        # Mark session notifications as read
        response = client.post(
            f"/api/v1/notifications/mark-all-read?session_id={session_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["marked_read"] == 2

    def test_clear_notifications(self, client: TestClient) -> None:
        """Test clearing all notifications."""
        # Create some notifications
        for i in range(3):
            client.post(
                "/api/v1/notifications/",
                json={
                    "type": "info",
                    "title": f"Notification {i}",
                    "message": f"Message {i}",
                },
            )

        # Clear all
        response = client.delete("/api/v1/notifications/")

        assert response.status_code == 204

        # Verify all cleared
        list_response = client.get("/api/v1/notifications/")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["total"] == 0

    def test_clear_notifications_by_session(
        self,
        client: TestClient,
        session_manager: TrainingSessionManager,
        tmp_path: Path,
    ) -> None:
        """Test clearing notifications for specific session."""
        # Create session
        config = tmp_path / "config.yaml"
        config.write_text("experiment:\n  name: test\n")
        session_id = session_manager.create_session(
            SessionConfig(config_path=config, experiment_name="test_experiment")
        )

        # Create notifications
        client.post(
            "/api/v1/notifications/",
            json={
                "session_id": session_id,
                "type": "info",
                "title": "Session Notification",
                "message": "Message",
            },
        )
        client.post(
            "/api/v1/notifications/",
            json={
                "type": "info",
                "title": "System Notification",
                "message": "Message",
            },
        )

        # Clear session notifications
        response = client.delete(f"/api/v1/notifications/?session_id={session_id}")

        assert response.status_code == 204

        # Verify session notifications cleared but system notification remains
        list_response = client.get("/api/v1/notifications/")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["total"] >= 1
        assert all(n["session_id"] != session_id for n in list_data["notifications"])

    def test_unread_only_filter(self, client: TestClient) -> None:
        """Test filtering for unread notifications only."""
        # Create notifications
        for i in range(3):
            response = client.post(
                "/api/v1/notifications/",
                json={
                    "type": "info",
                    "title": f"Notification {i}",
                    "message": f"Message {i}",
                },
            )
            notification_id = response.json()["id"]

            # Mark first one as read
            if i == 0:
                client.patch(
                    f"/api/v1/notifications/{notification_id}", json={"read": True}
                )

        # Get only unread
        response = client.get("/api/v1/notifications/?unread_only=true")

        assert response.status_code == 200
        data = response.json()
        assert all(not n["read"] for n in data["notifications"])
        assert data["total"] >= 2

    def test_notification_limit(self, client: TestClient) -> None:
        """Test limiting number of returned notifications."""
        # Create many notifications
        for i in range(10):
            client.post(
                "/api/v1/notifications/",
                json={
                    "type": "info",
                    "title": f"Notification {i}",
                    "message": f"Message {i}",
                },
            )

        # Request with limit
        response = client.get("/api/v1/notifications/?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["notifications"]) <= 5

    def test_notification_manager_not_initialized(self, tmp_path: Path) -> None:
        """Test notification endpoints when session manager is not initialized."""
        # Create app without session manager
        app = create_app()
        client = TestClient(app)

        # Try to create notification with session_id
        # Manager is None, so session validation will fail with 500
        response = client.post(
            "/api/v1/notifications/",
            json={
                "session_id": "test_session",
                "type": "info",
                "title": "Test",
                "message": "Test message",
            },
        )

        # When manager is not initialized and session_id is provided,
        # it can't validate the session and returns 500 or 404
        assert response.status_code in (404, 500)
