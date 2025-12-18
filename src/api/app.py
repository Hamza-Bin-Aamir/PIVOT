"""FastAPI application factory.

This module provides the main application factory for creating and configuring
the FastAPI server with middleware, routers, and exception handlers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import APIConfig
from .database import init_database
from .logging_config import setup_logging
from .process_manager import TrainingProcessManager
from .routers import (
    alerts,
    checkpoints,
    epochs,
    graphs,
    health,
    metrics,
    monitor,
    notifications,
    sse,
    status,
    training,
    webhook,
    websocket,
)
from .session_manager import TrainingSessionManager


def create_app(
    config: APIConfig | None = None,
    session_manager: TrainingSessionManager | None = None,
    process_manager: TrainingProcessManager | None = None,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: API configuration. If None, uses default configuration
        session_manager: Training session manager
        process_manager: Training process manager

    Returns:
        Configured FastAPI application

    Example:
        >>> from api import create_app, APIConfig
        >>> config = APIConfig(debug=True)
        >>> app = create_app(config)
        >>> # Run with: uvicorn api.app:app --reload
    """
    if config is None:
        config = APIConfig()

    # Setup logging
    logger = setup_logging(config)
    logger.info('Initializing PIVOT API', extra={'extra_data': {
        'version': config.version,
        'debug': config.debug,
    }})

    # Initialize database
    init_database(config)
    logger.info('Database initialized', extra={'extra_data': {
        'database_url': config.database_url,
    }})

    # Create FastAPI app
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        debug=config.debug,
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
    )

    logger.info('FastAPI application created')

    # Add CORS middleware
    if config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors.allow_origins,
            allow_credentials=config.cors.allow_credentials,
            allow_methods=config.cors.allow_methods,
            allow_headers=config.cors.allow_headers,
        )

    # Initialize session manager
    if session_manager is not None:
        status.set_session_manager(session_manager)
        epochs.set_session_manager(session_manager)
        metrics.set_session_manager(session_manager)
        graphs.set_session_manager(session_manager)
        websocket.set_session_manager(session_manager)
        sse.set_session_manager(session_manager)
        notifications.set_session_manager(session_manager)

    # Initialize process manager
    if process_manager is not None:
        training.initialize_process_manager(process_manager)

    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(status.router, prefix="/api/v1", tags=["status"])
    app.include_router(epochs.router, prefix="/api/v1", tags=["epochs"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
    app.include_router(training.router, prefix="/api/v1", tags=["training"])
    app.include_router(graphs.router, prefix="/api/v1", tags=["graphs"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
    app.include_router(sse.router, prefix="/api/v1", tags=["sse"])
    app.include_router(
        notifications.router, prefix="/api/v1", tags=["notifications"]
    )
    app.include_router(checkpoints.router, prefix="/api/v1", tags=["checkpoints"])
    app.include_router(monitor.router, prefix="/api/v1", tags=["monitoring"])
    app.include_router(alerts.router, prefix="/api/v1", tags=["alerting"])
    app.include_router(webhook.router, prefix="/api/v1", tags=["webhooks"])

    # Root endpoint
    @app.get("/")
    async def root() -> JSONResponse:
        """Root endpoint returning API information."""
        return JSONResponse(
            {
                "name": config.title,
                "version": config.version,
                "status": "running",
                "docs": "/docs" if config.debug else None,
            }
        )

    return app
