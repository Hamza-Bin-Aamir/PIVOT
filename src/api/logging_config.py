"""Structured logging configuration for PIVOT API.

This module provides centralized logging configuration with structured
output, log rotation, and different handlers for development and production.
"""

import logging
import sys
from collections.abc import MutableMapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import APIConfig


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Add custom fields
        record.service = getattr(record, 'service', 'pivot-api')
        record.request_id = getattr(record, 'request_id', '-')

        # Format the base message
        message = super().format(record)

        # Add extra context if available
        if hasattr(record, 'extra_data') and record.extra_data:
            extra_str = ' '.join(f'{k}={v}' for k, v in record.extra_data.items())
            message = f'{message} | {extra_str}'

        return message


def setup_logging(config: APIConfig | None = None) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        config: API configuration. If None, uses default configuration

    Returns:
        Configured root logger

    Example:
        >>> from api.logging_config import setup_logging
        >>> logger = setup_logging()
        >>> logger.info('Application started')
    """
    if config is None:
        config = APIConfig()

    # Get root logger
    logger = logging.getLogger('pivot')
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = StructuredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(service)s | %(request_id)s | '
            '%(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    console_handler.setFormatter(
        simple_formatter if config.debug else detailed_formatter
    )
    logger.addHandler(console_handler)

    # File handler (rotating)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Error file handler
    if config.error_log_file:
        error_path = Path(config.error_log_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)

        error_handler = RotatingFileHandler(
            filename=error_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    logger.info('Logging configured', extra={'extra_data': {
        'debug': config.debug,
        'log_file': str(config.log_file) if config.log_file else None,
    }})

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Processing request')
    """
    return logging.getLogger(f'pivot.{name}')


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter to add contextual information to logs."""

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Add context to log messages.

        Args:
            msg: Log message
            kwargs: Log keyword arguments

        Returns:
            Processed message and kwargs
        """
        # Add context from extra dict
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_context_logger(name: str, **context: object) -> LoggerAdapter:
    """Get a logger with contextual information.

    Args:
        name: Logger name
        **context: Context to add to all log messages

    Returns:
        Logger adapter with context

    Example:
        >>> logger = get_context_logger(__name__, session_id='sess_123')
        >>> logger.info('Training started')
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
