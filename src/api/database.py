"""Database models and setup for metrics persistence.

This module provides SQLAlchemy models for storing training metrics,
epoch data, and session information. Supports both SQLite (development)
and PostgreSQL (production).
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import APIConfig


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class SessionRecord(Base):
    """Training session record."""

    __tablename__ = 'training_sessions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    epochs: Mapped[list['EpochRecord']] = relationship(
        'EpochRecord',
        back_populates='session',
        cascade='all, delete-orphan',
    )
    metrics: Mapped[list['MetricRecord']] = relationship(
        'MetricRecord',
        back_populates='session',
        cascade='all, delete-orphan',
    )


class EpochRecord(Base):
    """Training epoch record."""

    __tablename__ = 'epochs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), ForeignKey('training_sessions.session_id'), nullable=False, index=True)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Metrics
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Additional data
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    session: Mapped['SessionRecord'] = relationship('SessionRecord', back_populates='epochs')


class MetricRecord(Base):
    """Individual metric record."""

    __tablename__ = 'metrics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), ForeignKey('training_sessions.session_id'), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Metric data
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=True)  # 'train', 'val', 'test'

    # Additional context
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    session: Mapped['SessionRecord'] = relationship('SessionRecord', back_populates='metrics')


class CheckpointMetadata(Base):
    """Checkpoint metadata model."""

    __tablename__ = 'checkpoints'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(512), nullable=False)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)


# Database connection and session management
_engine = None
_SessionLocal = None


def init_database(config: APIConfig) -> None:
    """Initialize database connection and create tables.

    Args:
        config: API configuration with database settings
    """
    global _engine, _SessionLocal

    # Default to SQLite for development
    database_url = getattr(config, 'database_url', 'sqlite:///./pivot_metrics.db')

    # Create engine
    connect_args = {}
    if database_url.startswith('sqlite'):
        connect_args = {'check_same_thread': False}

    _engine = create_engine(database_url, connect_args=connect_args, echo=config.debug)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Create all tables
    Base.metadata.create_all(bind=_engine)


def get_db_session() -> Session:
    """Get a database session.

    Returns:
        Database session

    Raises:
        RuntimeError: If database not initialized
    """
    if _SessionLocal is None:
        raise RuntimeError('Database not initialized. Call init_database first.')

    return _SessionLocal()


# CRUD operations for sessions
def create_session(session_id: str, status: str = 'active', config: dict[str, Any] | None = None) -> SessionRecord:
    """Create a new training session record.

    Args:
        session_id: Unique session identifier
        status: Session status
        config: Optional session configuration

    Returns:
        Created session record
    """
    db = get_db_session()
    try:
        session = SessionRecord(session_id=session_id, status=status, config=config)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def get_session(session_id: str) -> SessionRecord | None:
    """Get a training session by ID.

    Args:
        session_id: Session identifier

    Returns:
        Session record or None if not found
    """
    db = get_db_session()
    try:
        return db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
    finally:
        db.close()


def update_session_status(session_id: str, status: str) -> SessionRecord | None:
    """Update session status.

    Args:
        session_id: Session identifier
        status: New status

    Returns:
        Updated session record or None if not found
    """
    db = get_db_session()
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if session:
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()


def list_sessions(limit: int = 100, offset: int = 0) -> list[SessionRecord]:
    """List all training sessions.

    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip

    Returns:
        List of session records
    """
    db = get_db_session()
    try:
        return db.query(SessionRecord).order_by(SessionRecord.created_at.desc()).limit(limit).offset(offset).all()
    finally:
        db.close()


# CRUD operations for epochs
def create_epoch(
    session_id: str,
    epoch: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
    train_accuracy: float | None = None,
    val_accuracy: float | None = None,
    learning_rate: float | None = None,
    extra_data: dict[str, Any] | None = None,
) -> EpochRecord:
    """Create a new epoch record.

    Args:
        session_id: Session identifier
        epoch: Epoch number
        train_loss: Training loss
        val_loss: Validation loss
        train_accuracy: Training accuracy
        val_accuracy: Validation accuracy
        learning_rate: Learning rate
        extra_data: Additional metadata

    Returns:
        Created epoch record
    """
    db = get_db_session()
    try:
        epoch_record = EpochRecord(
            session_id=session_id,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            extra_data=extra_data,
        )
        db.add(epoch_record)
        db.commit()
        db.refresh(epoch_record)
        return epoch_record
    finally:
        db.close()


def get_session_epochs(session_id: str) -> list[EpochRecord]:
    """Get all epochs for a session.

    Args:
        session_id: Session identifier

    Returns:
        List of epoch records
    """
    db = get_db_session()
    try:
        return db.query(EpochRecord).filter(EpochRecord.session_id == session_id).order_by(EpochRecord.epoch).all()
    finally:
        db.close()


# CRUD operations for metrics
def create_metric(
    session_id: str,
    metric_name: str,
    metric_value: float,
    metric_type: str | None = None,
    epoch: int | None = None,
    batch: int | None = None,
    extra_data: dict[str, Any] | None = None,
) -> MetricRecord:
    """Create a new metric record.

    Args:
        session_id: Session identifier
        metric_name: Name of the metric
        metric_value: Metric value
        metric_type: Type of metric (train/val/test)
        epoch: Epoch number
        batch: Batch number
        extra_data: Additional metadata

    Returns:
        Created metric record
    """
    db = get_db_session()
    try:
        metric = MetricRecord(
            session_id=session_id,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            epoch=epoch,
            batch=batch,
            extra_data=extra_data,
        )
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric
    finally:
        db.close()


def get_session_metrics(
    session_id: str,
    metric_name: str | None = None,
    metric_type: str | None = None,
    limit: int = 1000,
) -> list[MetricRecord]:
    """Get metrics for a session.

    Args:
        session_id: Session identifier
        metric_name: Filter by metric name
        metric_type: Filter by metric type
        limit: Maximum number of records

    Returns:
        List of metric records
    """
    db = get_db_session()
    try:
        query = db.query(MetricRecord).filter(MetricRecord.session_id == session_id)

        if metric_name:
            query = query.filter(MetricRecord.metric_name == metric_name)
        if metric_type:
            query = query.filter(MetricRecord.metric_type == metric_type)

        return query.order_by(MetricRecord.timestamp).limit(limit).all()
    finally:
        db.close()


def delete_session_data(session_id: str) -> bool:
    """Delete all data for a session.

    Args:
        session_id: Session identifier

    Returns:
        True if session was deleted, False if not found
    """
    db = get_db_session()
    try:
        session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    finally:
        db.close()
