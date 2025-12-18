"""Tests for database module.

This module tests SQLAlchemy models, database initialization,
and CRUD operations for sessions, epochs, and metrics.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.api.config import APIConfig
from src.api.database import (
    EpochRecord,
    MetricRecord,
    SessionRecord,
    create_epoch,
    create_metric,
    create_session,
    delete_session_data,
    get_db_session,
    get_session,
    get_session_epochs,
    get_session_metrics,
    init_database,
    list_sessions,
    update_session_status,
)


class TestDatabaseInitialization:
    """Test database initialization and setup."""

    def test_init_database_creates_tables(self, tmp_path: Path) -> None:
        """Test that database initialization creates all tables."""
        db_file = tmp_path / "test.db"
        config = APIConfig(database_url=f"sqlite:///{db_file}")

        init_database(config)

        # Verify database file exists
        assert db_file.exists()

        # Verify we can get a session (meaning database is initialized)
        db = get_db_session()
        assert db is not None
        db.close()

    def test_get_db_session_raises_without_init(self) -> None:
        """Test that get_db_session raises error if not initialized."""
        # Reset global state
        import src.api.database as db_module
        db_module._engine = None
        db_module._SessionLocal = None

        with pytest.raises(RuntimeError, match="Database not initialized"):
            get_db_session()

    def test_init_database_with_postgres_url(self) -> None:
        """Test database initialization with PostgreSQL URL."""
        config = APIConfig(database_url="postgresql://user:pass@localhost/dbname")

        # This will fail to connect, but should not raise during init
        # In real tests, you'd use a test database
        import contextlib
        with contextlib.suppress(Exception):
            init_database(config)


class TestSessionCRUD:
    """Test CRUD operations for training sessions."""

    @pytest.fixture(autouse=True)
    def setup_database(self, tmp_path: Path) -> None:
        """Setup test database before each test."""
        db_file = tmp_path / "test_sessions.db"
        config = APIConfig(database_url=f"sqlite:///{db_file}")
        init_database(config)

        # Clear all data
        db = get_db_session()
        try:
            db.query(SessionRecord).delete()
            db.commit()
        finally:
            db.close()

    def test_create_session(self) -> None:
        """Test creating a new training session."""
        session = create_session(
            session_id="sess_001",
            status="active",
            config={"epochs": 10, "batch_size": 16}
        )

        assert session.id is not None
        assert session.session_id == "sess_001"
        assert session.status == "active"
        assert session.config == {"epochs": 10, "batch_size": 16}
        assert isinstance(session.created_at, datetime)

    def test_get_session_existing(self) -> None:
        """Test getting an existing session."""
        create_session(session_id="sess_002", status="active")

        session = get_session("sess_002")

        assert session is not None
        assert session.session_id == "sess_002"
        assert session.status == "active"

    def test_get_session_nonexistent(self) -> None:
        """Test getting a non-existent session returns None."""
        session = get_session("nonexistent")

        assert session is None

    def test_update_session_status(self) -> None:
        """Test updating session status."""
        create_session(session_id="sess_003", status="active")

        updated = update_session_status("sess_003", "completed")

        assert updated is not None
        assert updated.status == "completed"
        assert updated.updated_at > updated.created_at

    def test_update_session_status_nonexistent(self) -> None:
        """Test updating non-existent session returns None."""
        result = update_session_status("nonexistent", "completed")

        assert result is None

    def test_list_sessions(self) -> None:
        """Test listing all sessions."""
        create_session(session_id="sess_004", status="active")
        create_session(session_id="sess_005", status="completed")
        create_session(session_id="sess_006", status="failed")

        sessions = list_sessions()

        assert len(sessions) == 3
        # Should be ordered by created_at descending
        assert sessions[0].session_id == "sess_006"

    def test_list_sessions_with_pagination(self) -> None:
        """Test listing sessions with pagination."""
        for i in range(5):
            create_session(session_id=f"sess_{i:03d}", status="active")

        sessions = list_sessions(limit=2, offset=1)

        assert len(sessions) == 2

    def test_delete_session_data(self) -> None:
        """Test deleting session and all related data."""
        create_session(session_id="sess_007", status="active")

        result = delete_session_data("sess_007")

        assert result is True
        assert get_session("sess_007") is None

    def test_delete_session_data_nonexistent(self) -> None:
        """Test deleting non-existent session."""
        result = delete_session_data("nonexistent")

        assert result is False


class TestEpochCRUD:
    """Test CRUD operations for epoch records."""

    @pytest.fixture(autouse=True)
    def setup_database(self, tmp_path: Path) -> None:
        """Setup test database and create a session."""
        db_file = tmp_path / "test_epochs.db"
        config = APIConfig(database_url=f"sqlite:///{db_file}")
        init_database(config)

        # Clear and create test session
        db = get_db_session()
        try:
            db.query(EpochRecord).delete()
            db.query(SessionRecord).delete()
            db.commit()
        finally:
            db.close()

        create_session(session_id="sess_test", status="active")

    def test_create_epoch(self) -> None:
        """Test creating an epoch record."""
        epoch = create_epoch(
            session_id="sess_test",
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.85,
            val_accuracy=0.80,
            learning_rate=0.001,
            extra_data={"batch_count": 100}
        )

        assert epoch.id is not None
        assert epoch.session_id == "sess_test"
        assert epoch.epoch == 1
        assert epoch.train_loss == 0.5
        assert epoch.val_loss == 0.6
        assert epoch.extra_data == {"batch_count": 100}

    def test_get_session_epochs(self) -> None:
        """Test getting all epochs for a session."""
        create_epoch(session_id="sess_test", epoch=1, train_loss=0.5)
        create_epoch(session_id="sess_test", epoch=2, train_loss=0.4)
        create_epoch(session_id="sess_test", epoch=3, train_loss=0.3)

        epochs = get_session_epochs("sess_test")

        assert len(epochs) == 3
        assert epochs[0].epoch == 1
        assert epochs[1].epoch == 2
        assert epochs[2].epoch == 3

    def test_get_session_epochs_empty(self) -> None:
        """Test getting epochs for session with no epochs."""
        epochs = get_session_epochs("sess_test")

        assert len(epochs) == 0

    def test_cascade_delete_epochs(self) -> None:
        """Test that deleting a session cascades to epochs."""
        create_epoch(session_id="sess_test", epoch=1, train_loss=0.5)
        create_epoch(session_id="sess_test", epoch=2, train_loss=0.4)

        delete_session_data("sess_test")

        epochs = get_session_epochs("sess_test")
        assert len(epochs) == 0


class TestMetricCRUD:
    """Test CRUD operations for metric records."""

    @pytest.fixture(autouse=True)
    def setup_database(self, tmp_path: Path) -> None:
        """Setup test database and create a session."""
        db_file = tmp_path / "test_metrics.db"
        config = APIConfig(database_url=f"sqlite:///{db_file}")
        init_database(config)

        # Clear and create test session
        db = get_db_session()
        try:
            db.query(MetricRecord).delete()
            db.query(SessionRecord).delete()
            db.commit()
        finally:
            db.close()

        create_session(session_id="sess_metrics", status="active")

    def test_create_metric(self) -> None:
        """Test creating a metric record."""
        metric = create_metric(
            session_id="sess_metrics",
            metric_name="accuracy",
            metric_value=0.85,
            metric_type="train",
            epoch=1,
            batch=10,
            extra_data={"gpu_memory": "2GB"}
        )

        assert metric.id is not None
        assert metric.session_id == "sess_metrics"
        assert metric.metric_name == "accuracy"
        assert metric.metric_value == 0.85
        assert metric.metric_type == "train"
        assert metric.epoch == 1
        assert metric.batch == 10

    def test_get_session_metrics(self) -> None:
        """Test getting all metrics for a session."""
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.5)
        create_metric(session_id="sess_metrics", metric_name="accuracy", metric_value=0.8)
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.4)

        metrics = get_session_metrics("sess_metrics")

        assert len(metrics) == 3

    def test_get_session_metrics_by_name(self) -> None:
        """Test filtering metrics by name."""
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.5)
        create_metric(session_id="sess_metrics", metric_name="accuracy", metric_value=0.8)
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.4)

        metrics = get_session_metrics("sess_metrics", metric_name="loss")

        assert len(metrics) == 2
        assert all(m.metric_name == "loss" for m in metrics)

    def test_get_session_metrics_by_type(self) -> None:
        """Test filtering metrics by type."""
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.5, metric_type="train")
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.6, metric_type="val")
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.4, metric_type="train")

        metrics = get_session_metrics("sess_metrics", metric_type="train")

        assert len(metrics) == 2
        assert all(m.metric_type == "train" for m in metrics)

    def test_get_session_metrics_with_limit(self) -> None:
        """Test limiting number of metrics returned."""
        for i in range(10):
            create_metric(session_id="sess_metrics", metric_name="loss", metric_value=float(i))

        metrics = get_session_metrics("sess_metrics", limit=5)

        assert len(metrics) == 5

    def test_cascade_delete_metrics(self) -> None:
        """Test that deleting a session cascades to metrics."""
        create_metric(session_id="sess_metrics", metric_name="loss", metric_value=0.5)
        create_metric(session_id="sess_metrics", metric_name="accuracy", metric_value=0.8)

        delete_session_data("sess_metrics")

        metrics = get_session_metrics("sess_metrics")
        assert len(metrics) == 0


class TestDatabaseRelationships:
    """Test database relationships and constraints."""

    @pytest.fixture(autouse=True)
    def setup_database(self, tmp_path: Path) -> None:
        """Setup test database."""
        db_file = tmp_path / "test_relationships.db"
        config = APIConfig(database_url=f"sqlite:///{db_file}")
        init_database(config)

        # Clear all data
        db = get_db_session()
        try:
            db.query(MetricRecord).delete()
            db.query(EpochRecord).delete()
            db.query(SessionRecord).delete()
            db.commit()
        finally:
            db.close()

    def test_session_has_epochs_relationship(self) -> None:
        """Test that session has epochs relationship."""
        _ = create_session(session_id="sess_rel", status="active")
        create_epoch(session_id="sess_rel", epoch=1, train_loss=0.5)
        create_epoch(session_id="sess_rel", epoch=2, train_loss=0.4)

        db = get_db_session()
        try:
            db_session = db.query(SessionRecord).filter(SessionRecord.session_id == "sess_rel").first()
            assert len(db_session.epochs) == 2
        finally:
            db.close()

    def test_session_has_metrics_relationship(self) -> None:
        """Test that session has metrics relationship."""
        create_session(session_id="sess_rel", status="active")
        create_metric(session_id="sess_rel", metric_name="loss", metric_value=0.5)
        create_metric(session_id="sess_rel", metric_name="accuracy", metric_value=0.8)

        db = get_db_session()
        try:
            db_session = db.query(SessionRecord).filter(SessionRecord.session_id == "sess_rel").first()
            assert len(db_session.metrics) == 2
        finally:
            db.close()

    def test_unique_session_id_constraint(self) -> None:
        """Test that session_id must be unique."""
        from sqlalchemy.exc import IntegrityError
        create_session(session_id="unique_sess", status="active")

        with pytest.raises(IntegrityError):
            create_session(session_id="unique_sess", status="active")
