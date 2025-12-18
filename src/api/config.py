"""API configuration for FastAPI server.

This module defines configuration dataclasses for the FastAPI server,
including server settings, CORS configuration, and API versioning.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CORSConfig:
    """CORS (Cross-Origin Resource Sharing) configuration.

    Attributes:
        enabled: Whether to enable CORS middleware
        allow_origins: List of allowed origins. Use ["*"] to allow all
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed HTTP headers
        allow_credentials: Whether to allow credentials (cookies, auth headers)
    """

    enabled: bool = True
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False

    def __post_init__(self) -> None:
        """Validate CORS configuration."""
        if self.allow_credentials and "*" in self.allow_origins:
            raise ValueError(
                "Cannot allow credentials when allow_origins contains '*'. "
                "Specify explicit origins instead."
            )


@dataclass
class ServerConfig:
    """Server configuration.

    Attributes:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Enable auto-reload on code changes (development only)
        workers: Number of worker processes (production only)
        log_level: Logging level (debug, info, warning, error, critical)
    """

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"

    def __post_init__(self) -> None:
        """Validate server configuration."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be in range 1-65535, got {self.port}")

        if self.workers < 1:
            raise ValueError(f"Workers must be >= 1, got {self.workers}")

        valid_log_levels = {"debug", "info", "warning", "error", "critical"}
        if self.log_level.lower() not in valid_log_levels:
            raise ValueError(
                f"Invalid log level '{self.log_level}'. "
                f"Must be one of {valid_log_levels}"
            )


@dataclass
class APIConfig:
    """Complete API configuration.

    Attributes:
        title: API title
        description: API description
        version: API version
        debug: Enable debug mode
        log_file: Path to log file (None to disable file logging)
        error_log_file: Path to error log file (None to disable)
        database_url: Database connection URL (SQLite or PostgreSQL)
        server: Server configuration
        cors: CORS configuration
    """

    title: str = "PIVOT Training API"
    description: str = "REST API for managing lung nodule detection model training"
    version: str = "0.1.0"
    debug: bool = False
    log_file: Path | None = field(default_factory=lambda: Path("logs/api.log"))
    error_log_file: Path | None = field(
        default_factory=lambda: Path("logs/error.log")
    )
    database_url: str = "sqlite:///./pivot_metrics.db"
    server: ServerConfig = field(default_factory=ServerConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)

    def __post_init__(self) -> None:
        """Validate API configuration."""
        if self.debug and self.server.workers > 1:
            raise ValueError("Debug mode requires workers=1")
