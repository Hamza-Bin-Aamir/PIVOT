"""Tests for structured logging configuration."""

import logging
from pathlib import Path

from src.api.config import APIConfig
from src.api.logging_config import (
    LoggerAdapter,
    StructuredFormatter,
    get_context_logger,
    get_logger,
    setup_logging,
)


class TestStructuredFormatter:
    """Test suite for StructuredFormatter."""

    def test_formatter_basic(self) -> None:
        """Test basic log formatting."""
        formatter = StructuredFormatter(
            fmt='%(levelname)s | %(service)s | %(message)s'
        )
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert 'INFO' in formatted
        assert 'pivot-api' in formatted
        assert 'Test message' in formatted

    def test_formatter_with_extra_data(self) -> None:
        """Test formatting with extra data."""
        formatter = StructuredFormatter(
            fmt='%(levelname)s | %(message)s'
        )
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None,
        )
        record.extra_data = {'session_id': 'sess_123', 'epoch': 5}

        formatted = formatter.format(record)
        assert 'Test message' in formatted
        assert 'session_id=sess_123' in formatted
        assert 'epoch=5' in formatted


class TestLoggingSetup:
    """Test suite for logging setup."""

    def test_setup_logging_default(self, tmp_path: Path) -> None:
        """Test setting up logging with default config."""
        config = APIConfig(
            log_file=tmp_path / "test.log",
            error_log_file=tmp_path / "error.log",
        )
        logger = setup_logging(config)

        assert logger is not None
        assert logger.name == 'pivot'
        assert len(logger.handlers) >= 1

    def test_setup_logging_debug_mode(self, tmp_path: Path) -> None:
        """Test logging setup in debug mode."""
        config = APIConfig(
            debug=True,
            log_file=tmp_path / "test.log",
        )
        logger = setup_logging(config)

        assert logger.level == logging.DEBUG

    def test_setup_logging_no_file(self) -> None:
        """Test logging setup without file logging."""
        config = APIConfig(log_file=None, error_log_file=None)
        logger = setup_logging(config)

        # Should only have console handler
        assert len(logger.handlers) == 1

    def test_get_logger(self) -> None:
        """Test getting a named logger."""
        logger = get_logger('test_module')

        assert logger.name == 'pivot.test_module'

    def test_get_context_logger(self) -> None:
        """Test getting a logger with context."""
        logger = get_context_logger('test_module', session_id='sess_123')

        assert isinstance(logger, LoggerAdapter)
        assert logger.extra is not None
        assert logger.extra['session_id'] == 'sess_123'

    def test_log_file_creation(self, tmp_path: Path) -> None:
        """Test that log files are created."""
        log_file = tmp_path / "logs" / "test.log"
        config = APIConfig(log_file=log_file, error_log_file=None)

        _ = setup_logging(config)
        test_logger = get_logger('test')
        test_logger.info('Test message')

        assert log_file.exists()
        content = log_file.read_text()
        assert 'Test message' in content

    def test_error_log_separation(self, tmp_path: Path) -> None:
        """Test that errors go to separate file."""
        log_file = tmp_path / "test.log"
        error_file = tmp_path / "error.log"
        config = APIConfig(log_file=log_file, error_log_file=error_file)

        _ = setup_logging(config)
        test_logger = get_logger('test')

        test_logger.info('Info message')
        test_logger.error('Error message')

        # Both should be in main log
        main_content = log_file.read_text()
        assert 'Info message' in main_content
        assert 'Error message' in main_content

        # Only error should be in error log
        error_content = error_file.read_text()
        assert 'Info message' not in error_content
        assert 'Error message' in error_content


class TestLoggerAdapter:
    """Test suite for LoggerAdapter."""

    def test_adapter_adds_context(self) -> None:
        """Test that adapter adds context to logs."""
        base_logger = get_logger('test')
        adapter = LoggerAdapter(base_logger, {'session_id': 'sess_123'})

        msg, kwargs = adapter.process('Test message', {})

        assert 'session_id' in kwargs['extra']
        assert kwargs['extra']['session_id'] == 'sess_123'

    def test_adapter_merges_extra(self) -> None:
        """Test that adapter merges with existing extra."""
        base_logger = get_logger('test')
        adapter = LoggerAdapter(base_logger, {'session_id': 'sess_123'})

        msg, kwargs = adapter.process(
            'Test message',
            {'extra': {'epoch': 5}}
        )

        assert kwargs['extra']['session_id'] == 'sess_123'
        assert kwargs['extra']['epoch'] == 5
