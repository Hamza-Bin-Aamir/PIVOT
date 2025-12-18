"""Tests for FastAPI server configuration."""

import pytest

from src.api.config import APIConfig, CORSConfig, ServerConfig


class TestCORSConfig:
    """Test suite for CORS configuration."""

    def test_default_cors_config(self):
        """Test default CORS configuration."""
        config = CORSConfig()

        assert config.enabled is True
        assert config.allow_origins == ["*"]
        assert config.allow_methods == ["*"]
        assert config.allow_headers == ["*"]
        assert config.allow_credentials is False

    def test_custom_cors_config(self):
        """Test custom CORS configuration."""
        config = CORSConfig(
            enabled=False,
            allow_origins=["http://localhost:3000"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
        )

        assert config.enabled is False
        assert config.allow_origins == ["http://localhost:3000"]
        assert config.allow_methods == ["GET", "POST"]
        assert config.allow_headers == ["Content-Type"]
        assert config.allow_credentials is False

    def test_credentials_with_wildcard_origin_raises_error(self):
        """Test that allowing credentials with wildcard origin raises error."""
        with pytest.raises(ValueError, match="Cannot allow credentials"):
            CORSConfig(allow_credentials=True, allow_origins=["*"])

    def test_credentials_with_specific_origins(self):
        """Test allowing credentials with specific origins."""
        config = CORSConfig(
            allow_credentials=True,
            allow_origins=["http://localhost:3000", "https://example.com"],
        )

        assert config.allow_credentials is True
        assert len(config.allow_origins) == 2


class TestServerConfig:
    """Test suite for server configuration."""

    def test_default_server_config(self):
        """Test default server configuration."""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.reload is False
        assert config.workers == 1
        assert config.log_level == "info"

    def test_custom_server_config(self):
        """Test custom server configuration."""
        config = ServerConfig(
            host="127.0.0.1",
            port=5000,
            reload=True,
            workers=4,
            log_level="debug",
        )

        assert config.host == "127.0.0.1"
        assert config.port == 5000
        assert config.reload is True
        assert config.workers == 4
        assert config.log_level == "debug"

    def test_invalid_port_too_low(self):
        """Test that port below 1 raises error."""
        with pytest.raises(ValueError, match="Port must be in range 1-65535"):
            ServerConfig(port=0)

    def test_invalid_port_too_high(self):
        """Test that port above 65535 raises error."""
        with pytest.raises(ValueError, match="Port must be in range 1-65535"):
            ServerConfig(port=65536)

    def test_valid_port_boundaries(self):
        """Test valid port boundaries."""
        config_low = ServerConfig(port=1)
        config_high = ServerConfig(port=65535)

        assert config_low.port == 1
        assert config_high.port == 65535

    def test_invalid_workers(self):
        """Test that workers < 1 raises error."""
        with pytest.raises(ValueError, match="Workers must be >= 1"):
            ServerConfig(workers=0)

    def test_invalid_log_level(self):
        """Test that invalid log level raises error."""
        with pytest.raises(ValueError, match="Invalid log level"):
            ServerConfig(log_level="invalid")

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        valid_levels = ["debug", "info", "warning", "error", "critical"]

        for level in valid_levels:
            config = ServerConfig(log_level=level)
            assert config.log_level == level

    def test_log_level_case_insensitive(self):
        """Test that log level validation is case-insensitive."""
        config = ServerConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"


class TestAPIConfig:
    """Test suite for API configuration."""

    def test_default_api_config(self):
        """Test default API configuration."""
        config = APIConfig()

        assert config.title == "PIVOT Training API"
        assert "lung nodule" in config.description.lower()
        assert config.version == "0.1.0"
        assert config.debug is False
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.cors, CORSConfig)

    def test_custom_api_config(self):
        """Test custom API configuration."""
        config = APIConfig(
            title="Custom API",
            description="Custom description",
            version="1.0.0",
            debug=True,
        )

        assert config.title == "Custom API"
        assert config.description == "Custom description"
        assert config.version == "1.0.0"
        assert config.debug is True

    def test_api_config_with_custom_server(self):
        """Test API config with custom server configuration."""
        server_config = ServerConfig(port=9000, reload=True)
        config = APIConfig(server=server_config)

        assert config.server.port == 9000
        assert config.server.reload is True

    def test_api_config_with_custom_cors(self):
        """Test API config with custom CORS configuration."""
        cors_config = CORSConfig(
            enabled=False,
            allow_origins=["http://localhost:3000"],
        )
        config = APIConfig(cors=cors_config)

        assert config.cors.enabled is False
        assert config.cors.allow_origins == ["http://localhost:3000"]

    def test_debug_with_multiple_workers_raises_error(self):
        """Test that debug mode with multiple workers raises error."""
        server_config = ServerConfig(workers=4)

        with pytest.raises(ValueError, match="Debug mode requires workers=1"):
            APIConfig(debug=True, server=server_config)

    def test_debug_with_single_worker(self):
        """Test that debug mode works with single worker."""
        server_config = ServerConfig(workers=1)
        config = APIConfig(debug=True, server=server_config)

        assert config.debug is True
        assert config.server.workers == 1
