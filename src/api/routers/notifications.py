"""Notification system for training events and alerts.

This module provides endpoints for managing notifications and alerts
during training sessions. Notifications can be sent via WebSocket or SSE
and include events like training start/stop, epoch completion, errors, etc.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.api.session_manager import TrainingSessionManager

router = APIRouter(prefix="/notifications", tags=["notifications"])

# Global session manager reference
_session_manager: TrainingSessionManager | None = None


def set_session_manager(manager: TrainingSessionManager) -> None:
    """Set the global session manager instance."""
    global _session_manager
    _session_manager = manager


class NotificationType(str, Enum):
    """Types of notifications that can be sent."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Notification(BaseModel):
    """Model for a notification event."""

    id: str = Field(description="Unique notification identifier")
    session_id: str | None = Field(
        None, description="Training session ID (None for system-wide)"
    )
    type: NotificationType = Field(description="Type of notification")
    priority: NotificationPriority = Field(
        default=NotificationPriority.MEDIUM, description="Priority level"
    )
    title: str = Field(description="Notification title")
    message: str = Field(description="Notification message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the notification was created",
    )
    read: bool = Field(default=False, description="Whether notification has been read")
    metadata: dict[str, object] | None = Field(
        None, description="Additional notification metadata"
    )


class NotificationCreate(BaseModel):
    """Request model for creating a notification."""

    session_id: str | None = Field(
        None, description="Training session ID (None for system-wide)"
    )
    type: NotificationType = Field(description="Type of notification")
    priority: NotificationPriority = Field(
        default=NotificationPriority.MEDIUM, description="Priority level"
    )
    title: str = Field(description="Notification title", min_length=1, max_length=200)
    message: str = Field(
        description="Notification message", min_length=1, max_length=1000
    )
    metadata: dict[str, object] | None = Field(
        None, description="Additional notification metadata"
    )


class NotificationUpdate(BaseModel):
    """Request model for updating a notification."""

    read: bool = Field(description="Mark notification as read/unread")


class NotificationList(BaseModel):
    """Response model for listing notifications."""

    notifications: list[Notification] = Field(description="List of notifications")
    total: int = Field(description="Total number of notifications")
    unread: int = Field(description="Number of unread notifications")


# In-memory notification storage (in production, use a database)
_notifications: dict[str, Notification] = {}
_notification_counter = 0


def _generate_notification_id() -> str:
    """Generate a unique notification ID."""
    global _notification_counter
    _notification_counter += 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"notif_{timestamp}_{_notification_counter}"


@router.post("/", response_model=Notification, status_code=201)
async def create_notification(notification_data: NotificationCreate) -> Notification:
    """Create a new notification.

    Args:
        notification_data: Notification details

    Returns:
        Created notification

    Raises:
        HTTPException: If session_id is invalid
    """
    if _session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")

    # Validate session_id if provided
    if (
        notification_data.session_id is not None
        and notification_data.session_id not in _session_manager.sessions
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Session {notification_data.session_id} not found",
        )

    # Create notification
    notification = Notification(
        id=_generate_notification_id(),
        session_id=notification_data.session_id,
        type=notification_data.type,
        priority=notification_data.priority,
        title=notification_data.title,
        message=notification_data.message,
        metadata=notification_data.metadata,
    )

    # Store notification
    _notifications[notification.id] = notification

    return notification


@router.get("/", response_model=NotificationList)
async def list_notifications(
    session_id: str | None = None,
    type: NotificationType | None = None,
    priority: NotificationPriority | None = None,
    unread_only: bool = False,
    limit: int = 100,
) -> NotificationList:
    """List notifications with optional filters.

    Args:
        session_id: Filter by session ID (None returns all)
        type: Filter by notification type
        priority: Filter by priority level
        unread_only: Return only unread notifications
        limit: Maximum number of notifications to return

    Returns:
        List of notifications with counts
    """
    # Apply filters
    filtered = list(_notifications.values())

    if session_id is not None:
        filtered = [n for n in filtered if n.session_id == session_id]

    if type is not None:
        filtered = [n for n in filtered if n.type == type]

    if priority is not None:
        filtered = [n for n in filtered if n.priority == priority]

    if unread_only:
        filtered = [n for n in filtered if not n.read]

    # Sort by timestamp (newest first)
    filtered.sort(key=lambda n: n.timestamp, reverse=True)

    # Apply limit
    limited = filtered[:limit]

    # Count unread
    unread_count = sum(1 for n in _notifications.values() if not n.read)

    return NotificationList(
        notifications=limited, total=len(filtered), unread=unread_count
    )


@router.get("/{notification_id}", response_model=Notification)
async def get_notification(notification_id: str) -> Notification:
    """Get a specific notification by ID.

    Args:
        notification_id: Notification ID

    Returns:
        Notification details

    Raises:
        HTTPException: If notification not found
    """
    if notification_id not in _notifications:
        raise HTTPException(status_code=404, detail="Notification not found")

    return _notifications[notification_id]


@router.patch("/{notification_id}", response_model=Notification)
async def update_notification(
    notification_id: str, update_data: NotificationUpdate
) -> Notification:
    """Update a notification (mark as read/unread).

    Args:
        notification_id: Notification ID
        update_data: Update details

    Returns:
        Updated notification

    Raises:
        HTTPException: If notification not found
    """
    if notification_id not in _notifications:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification = _notifications[notification_id]
    notification.read = update_data.read

    return notification


@router.delete("/{notification_id}", status_code=204)
async def delete_notification(notification_id: str) -> None:
    """Delete a notification.

    Args:
        notification_id: Notification ID

    Raises:
        HTTPException: If notification not found
    """
    if notification_id not in _notifications:
        raise HTTPException(status_code=404, detail="Notification not found")

    del _notifications[notification_id]


@router.post("/mark-all-read", response_model=dict[str, int])
async def mark_all_read(session_id: str | None = None) -> dict[str, int]:
    """Mark all notifications as read.

    Args:
        session_id: Optional session ID to filter by

    Returns:
        Count of marked notifications
    """
    count = 0
    for notification in _notifications.values():
        if (
            session_id is None or notification.session_id == session_id
        ) and not notification.read:
            notification.read = True
            count += 1

    return {"marked_read": count}


@router.delete("/", status_code=204)
async def clear_notifications(session_id: str | None = None) -> None:
    """Clear all notifications.

    Args:
        session_id: Optional session ID to filter by
    """
    if session_id is None:
        _notifications.clear()
    else:
        to_delete = [
            nid for nid, n in _notifications.items() if n.session_id == session_id
        ]
        for nid in to_delete:
            del _notifications[nid]
