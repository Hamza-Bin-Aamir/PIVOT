"""WebSocket endpoint for real-time training updates.

This module provides WebSocket connectivity for streaming live training metrics,
logs, and status updates to connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..session_manager import TrainingSessionManager

router = APIRouter(prefix="/ws", tags=["websocket"])

# Global session manager instance
_session_manager: TrainingSessionManager | None = None


def set_session_manager(manager: TrainingSessionManager) -> None:
    """Set the global session manager instance.

    Args:
        manager: TrainingSessionManager instance to use
    """
    global _session_manager
    _session_manager = manager


def get_session_manager() -> TrainingSessionManager | None:
    """Get the global session manager instance.

    Returns:
        TrainingSessionManager instance or None if not initialized
    """
    return _session_manager


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates.

    This class maintains a list of active WebSocket connections and provides
    methods for broadcasting messages to all connected clients.

    Attributes:
        active_connections: List of active WebSocket connections
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from active connections.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket connection.

        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all active WebSocket connections.

        Args:
            message: Message to broadcast to all clients
        """
        import contextlib

        for connection in self.active_connections:
            with contextlib.suppress(Exception):
                # If sending fails, the connection will be cleaned up on disconnect
                await connection.send_text(message)


# Global connection manager
manager = ConnectionManager()


class TrainingUpdate(BaseModel):
    """Model for training update messages.

    Attributes:
        type: Type of update (metrics, status, log, error)
        session_id: Training session ID
        data: Update data (varies by type)
        timestamp: ISO timestamp of the update
    """

    type: str
    session_id: str
    data: dict[str, Any]
    timestamp: str


@router.websocket("/training/{session_id}")
async def websocket_training_updates(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time training updates.

    This endpoint accepts WebSocket connections and streams live updates
    about training progress, metrics, and status changes for a specific session.

    Args:
        websocket: WebSocket connection
        session_id: Training session ID to monitor

    Example:
        JavaScript client:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/training/session_123');

        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            console.log('Update type:', update.type);
            console.log('Data:', update.data);
        };
        ```

        Update message format:
        {
            "type": "metrics",
            "session_id": "session_123",
            "data": {
                "epoch": 5,
                "loss": 0.234,
                "accuracy": 0.892
            },
            "timestamp": "2025-12-18T10:30:45.123456"
        }
    """
    await manager.connect(websocket)

    try:
        # Verify session exists
        session_mgr = get_session_manager()
        if session_mgr:
            try:
                session_info = session_mgr.get_session_info(session_id)

                # Send initial connection confirmation
                welcome_message = TrainingUpdate(
                    type="connection",
                    session_id=session_id,
                    data={
                        "status": "connected",
                        "experiment_name": session_info.experiment_name,
                        "session_status": session_info.status.value,
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                await manager.send_personal_message(
                    welcome_message.model_dump_json(), websocket
                )
            except ValueError:
                # Session not found
                error_message = {
                    "type": "error",
                    "session_id": session_id,
                    "data": {"message": "Session not found"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await manager.send_personal_message(json.dumps(error_message), websocket)
                await websocket.close()
                return

        # Keep connection alive and listen for messages
        while True:
            # In a real implementation, this would:
            # 1. Listen for updates from the training process
            # 2. Stream metrics as they're generated
            # 3. Send status changes
            # For now, we maintain the connection and send periodic heartbeats

            await asyncio.sleep(5)  # Heartbeat every 5 seconds

            # Send heartbeat
            if session_mgr:
                try:
                    session_info = session_mgr.get_session_info(session_id)
                    heartbeat = TrainingUpdate(
                        type="heartbeat",
                        session_id=session_id,
                        data={"status": session_info.status.value},
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    await manager.send_personal_message(
                        heartbeat.model_dump_json(), websocket
                    )
                except ValueError:
                    # Session no longer exists
                    break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
        raise


@router.websocket("/broadcast")
async def websocket_broadcast(websocket: WebSocket) -> None:
    """WebSocket endpoint for receiving updates from all training sessions.

    This endpoint provides a broadcast channel that streams updates from
    all active training sessions, useful for monitoring multiple experiments.

    Args:
        websocket: WebSocket connection

    Example:
        JavaScript client:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/broadcast');

        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            console.log('Session:', update.session_id);
            console.log('Update:', update.data);
        };
        ```
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        welcome = {
            "type": "connection",
            "data": {
                "status": "connected",
                "message": "Broadcasting updates from all sessions",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await manager.send_personal_message(json.dumps(welcome), websocket)

        # Keep connection alive
        while True:
            await asyncio.sleep(10)  # Heartbeat every 10 seconds

            heartbeat = {
                "type": "heartbeat",
                "data": {
                    "active_connections": len(manager.active_connections),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await manager.send_personal_message(json.dumps(heartbeat), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
        raise
