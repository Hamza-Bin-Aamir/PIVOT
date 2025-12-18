"""Tests for FastAPI application factory."""

from fastapi.testclient import TestClient

from src.api import APIConfig, CORSConfig, create_app


class TestCreateApp:
    """Test suite for FastAPI app factory."""

    def test_create_app_default_config(self):
        """Test creating app with default configuration."""
        app = create_app()

        assert app.title == "PIVOT Training API"
        assert app.version == "0.1.0"
        assert app.debug is False

    def test_create_app_custom_config(self):
        """Test creating app with custom configuration."""
        config = APIConfig(
            title="Test API",
            description="Test description",
            version="2.0.0",
            debug=True,
        )
        app = create_app(config)

        assert app.title == "Test API"
        assert app.version == "2.0.0"
        assert app.debug is True

    def test_app_docs_enabled_in_debug(self):
        """Test that docs are enabled in debug mode."""
        config = APIConfig(debug=True)
        app = create_app(config)

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_app_docs_disabled_in_production(self):
        """Test that docs are disabled in production mode."""
        config = APIConfig(debug=False)
        app = create_app(config)

        assert app.docs_url is None
        assert app.redoc_url is None

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "PIVOT Training API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"

    def test_root_endpoint_debug_mode(self):
        """Test root endpoint includes docs link in debug mode."""
        config = APIConfig(debug=True)
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/")
        data = response.json()

        assert data["docs"] == "/docs"

    def test_root_endpoint_production_mode(self):
        """Test root endpoint excludes docs link in production mode."""
        config = APIConfig(debug=False)
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/")
        data = response.json()

        assert data["docs"] is None

    def test_health_router_included(self):
        """Test that health router is included."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_cors_middleware_enabled(self):
        """Test CORS middleware is added when enabled."""
        cors_config = CORSConfig(
            enabled=True,
            allow_origins=["http://localhost:3000"],
            allow_credentials=False,
        )
        config = APIConfig(cors=cors_config)
        app = create_app(config)

        # CORS middleware should be in the middleware stack
        # Check by making a request with Origin header
        client = TestClient(app)
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200

    def test_multiple_app_instances(self):
        """Test creating multiple app instances with different configs."""
        config1 = APIConfig(title="API 1", version="1.0.0")
        config2 = APIConfig(title="API 2", version="2.0.0")

        app1 = create_app(config1)
        app2 = create_app(config2)

        assert app1.title == "API 1"
        assert app2.title == "API 2"
        assert app1.version == "1.0.0"
        assert app2.version == "2.0.0"


class TestHealthEndpoints:
    """Test suite for health check endpoints."""

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0.0

    def test_health_check_response_schema(self):
        """Test health check response matches schema."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/health")
        data = response.json()

        # Verify all required fields
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["uptime_seconds"], (int, float))

    def test_ping_endpoint(self):
        """Test ping endpoint."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pong"

    def test_multiple_health_checks(self):
        """Test multiple consecutive health checks."""
        app = create_app()
        client = TestClient(app)

        for _ in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_health_check_different_timestamps(self):
        """Test that consecutive health checks have different timestamps."""
        app = create_app()
        client = TestClient(app)

        response1 = client.get("/api/v1/health")
        timestamp1 = response1.json()["timestamp"]

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        response2 = client.get("/api/v1/health")
        timestamp2 = response2.json()["timestamp"]

        assert timestamp1 != timestamp2
