"""FastAPI application factory.

This module provides the main application factory for creating and configuring
the FastAPI server with middleware, routers, and exception handlers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import APIConfig
from .routers import health


def create_app(config: APIConfig | None = None) -> FastAPI:
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

    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

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
