"""FastAPI application factory.

This module provides the main application factory for creating and configuring
the FastAPI server with middleware, routers, and exception handlers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import APIConfig
from .process_manager import TrainingProcessManager
from .routers import epochs, graphs, health, metrics, status, training
from .session_manager import TrainingSessionManager


def create_app(
    config: APIConfig | None = None,
    session_manager: TrainingSessionManager | None = None,
    process_manager: TrainingProcessManager | None = None,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: API configuration. If None, uses default configuration

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

    # Create FastAPI app
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        debug=config.debug,
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
    )

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
