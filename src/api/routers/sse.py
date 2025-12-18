"""Server-Sent Events (SSE) endpoint for real-time training updates.

This module provides SSE connectivity for streaming live training metrics,
logs, and status updates to connected dashboard clients using HTTP streaming.
SSE is a simpler alternative to WebSockets for one-way server-to-client communication.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..session_manager import TrainingSessionManager

router = APIRouter(prefix="/sse", tags=["sse"])

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


def format_sse_message(event_type: str, data: dict[str, Any]) -> str:
    """Format a message for Server-Sent Events protocol.

    Args:
        event_type: Type of event (e.g., 'metrics', 'status', 'heartbeat')
        data: Event data dictionary

    Returns:
        Formatted SSE message string

    Example:
        >>> format_sse_message('metrics', {'loss': 0.5})
        'event: metrics\\ndata: {"loss": 0.5}\\n\\n'
    """
    message = f"event: {event_type}\n"
    message += f"data: {json.dumps(data)}\n\n"
    return message


async def training_event_generator(session_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for a specific training session.

    This generator yields SSE-formatted messages with training updates,
    including metrics, status changes, and heartbeats.

    Args:
        session_id: Training session ID to monitor

    Yields:
        SSE-formatted event strings

    Raises:
        HTTPException: If session not found
    """
    session_mgr = get_session_manager()
    if not session_mgr:
        yield format_sse_message(
            "error",
            {
                "message": "Session manager not initialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    # Verify session exists
    try:
        session_info = session_mgr.get_session_info(session_id)
    except ValueError as e:
        yield format_sse_message(
            "error",
            {
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    # Send initial connection event
    yield format_sse_message(
        "connection",
        {
            "status": "connected",
            "session_id": session_id,
            "experiment_name": session_info.experiment_name,
            "session_status": session_info.status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Stream updates
    while True:
        try:
            # Check if session still exists
            session_info = session_mgr.get_session_info(session_id)

            # Send heartbeat
            yield format_sse_message(
                "heartbeat",
                {
                    "session_id": session_id,
                    "status": session_info.status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # In a real implementation, this would stream actual training metrics
            # For now, we send periodic heartbeats
            await asyncio.sleep(5)

        except ValueError:
            # Session no longer exists
            yield format_sse_message(
                "error",
                {
                    "message": "Session no longer exists",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            break
        except asyncio.CancelledError:
            # Client disconnected
            break


async def broadcast_event_generator() -> AsyncGenerator[str, None]:
    """Generate SSE events for all training sessions.

    This generator yields SSE-formatted messages with updates from all
    active training sessions, useful for monitoring multiple experiments.

    Yields:
        SSE-formatted event strings
    """
    # Send initial connection event
    yield format_sse_message(
        "connection",
        {
            "status": "connected",
            "message": "Broadcasting updates from all sessions",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Stream updates
    while True:
        try:
            session_mgr = get_session_manager()

            # Send heartbeat
            active_sessions = 0
            if session_mgr:
                active_sessions = len(session_mgr.list_sessions())

            yield format_sse_message(
                "heartbeat",
                {
                    "active_sessions": active_sessions,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # In a real implementation, this would stream updates from all sessions
            await asyncio.sleep(10)

        except asyncio.CancelledError:
            # Client disconnected
            break


@router.get("/training/{session_id}")
async def sse_training_updates(session_id: str) -> StreamingResponse:
    """SSE endpoint for real-time training updates.

    This endpoint provides Server-Sent Events streaming for a specific
    training session. Unlike WebSockets, SSE uses standard HTTP and is
    one-way (server to client only).

    Args:
        session_id: Training session ID to monitor

    Returns:
        StreamingResponse with SSE events

    Raises:
        HTTPException: If session manager not initialized

    Example:
        JavaScript client:
        ```javascript
        const eventSource = new EventSource('/api/v1/sse/training/session_123');

        eventSource.addEventListener('connection', (e) => {
            const data = JSON.parse(e.data);
            console.log('Connected:', data);
        });

        eventSource.addEventListener('metrics', (e) => {
            const data = JSON.parse(e.data);
            console.log('Metrics:', data);
        });

        eventSource.addEventListener('heartbeat', (e) => {
            const data = JSON.parse(e.data);
            console.log('Heartbeat:', data);
        });
        ```

        Event format:
        ```
        event: metrics
        data: {"epoch": 5, "loss": 0.234, "accuracy": 0.892}

        event: heartbeat
        data: {"session_id": "session_123", "status": "running"}
        ```
    """
    if not get_session_manager():
        raise HTTPException(
            status_code=500, detail="Session manager not initialized"
        )

    return StreamingResponse(
        training_event_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/broadcast")
async def sse_broadcast() -> StreamingResponse:
    """SSE endpoint for receiving updates from all training sessions.

    This endpoint provides a broadcast channel that streams updates from
    all active training sessions using Server-Sent Events.

    Returns:
        StreamingResponse with SSE events

    Example:
        JavaScript client:
        ```javascript
        const eventSource = new EventSource('/api/v1/sse/broadcast');

        eventSource.addEventListener('connection', (e) => {
            const data = JSON.parse(e.data);
            console.log('Connected:', data);
        });

        eventSource.addEventListener('heartbeat', (e) => {
            const data = JSON.parse(e.data);
            console.log('Active sessions:', data.active_sessions);
        });
        ```
    """
    return StreamingResponse(
        broadcast_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
