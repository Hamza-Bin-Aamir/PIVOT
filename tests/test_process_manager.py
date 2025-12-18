"""Tests for training process manager."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.process_manager import (
    ProcessConfig,
    ProcessInfo,
    ProcessStatus,
    TrainingProcessManager,
)


class TestProcessConfig:
    """Test suite for ProcessConfig."""

    def test_default_config(self, tmp_path: Path):
        """Test default process configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        config = ProcessConfig(config_path=config_file)

        assert config.config_path == config_file
        assert config.python_executable == "python"
        assert config.training_script == "src/train/main.py"
        assert config.use_uv is True
        assert config.environment == {}

    def test_custom_config(self, tmp_path: Path):
        """Test custom process configuration."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("custom: config")

        config = ProcessConfig(
            config_path=config_file,
            python_executable="/usr/bin/python3",
            training_script="custom/train.py",
            use_uv=False,
            environment={"KEY": "value"},
        )

        assert config.python_executable == "/usr/bin/python3"
        assert config.training_script == "custom/train.py"
        assert config.use_uv is False
        assert config.environment == {"KEY": "value"}

    def test_nonexistent_config_raises_error(self, tmp_path: Path):
        """Test that nonexistent config file raises error."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ProcessConfig(config_path=config_file)

    def test_invalid_extension_raises_error(self, tmp_path: Path):
        """Test that non-YAML extension raises error."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("test")

        with pytest.raises(ValueError, match="must be YAML"):
            ProcessConfig(config_path=config_file)

    def test_yml_extension_accepted(self, tmp_path: Path):
        """Test that .yml extension is accepted."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: value")

        config = ProcessConfig(config_path=config_file)
        assert config.config_path == config_file


class TestProcessInfo:
    """Test suite for ProcessInfo."""

    def test_default_process_info(self):
        """Test default process info."""
        info = ProcessInfo(process_id="test_123")

        assert info.process_id == "test_123"
        assert info.pid is None
        assert info.status == ProcessStatus.IDLE
        assert info.config_path is None
        assert info.start_time is None
        assert info.end_time is None
        assert info.exit_code is None
        assert info.error_message is None

    def test_to_dict(self):
        """Test converting process info to dictionary."""
        from datetime import datetime

        start_time = datetime.now()
        info = ProcessInfo(
            process_id="test_456",
            pid=12345,
            status=ProcessStatus.RUNNING,
            config_path="/path/to/config.yaml",
            start_time=start_time,
        )

        result = info.to_dict()

        assert result["process_id"] == "test_456"
        assert result["pid"] == 12345
        assert result["status"] == "running"
        assert result["config_path"] == "/path/to/config.yaml"
        assert result["start_time"] == start_time.isoformat()
        assert result["end_time"] is None
        assert result["exit_code"] is None
        assert result["error_message"] is None


class TestTrainingProcessManager:
    """Test suite for TrainingProcessManager."""

    def test_init(self):
        """Test manager initialization."""
        manager = TrainingProcessManager()

        assert manager.processes == {}
        assert manager._process_handles == {}
        assert manager._lock is not None

    @patch("subprocess.Popen")
    def test_start_process_with_uv(self, mock_popen: MagicMock, tmp_path: Path):
        """Test starting process with uv."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Mock subprocess
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file, use_uv=True)

        process_id = manager.start_process(config)

        assert process_id.startswith("train_")
        assert process_id in manager.processes
        assert manager.processes[process_id].pid == 12345
        assert manager.processes[process_id].status == ProcessStatus.RUNNING

        # Verify command
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "uv"
        assert cmd[1] == "run"
        assert cmd[2] == "python"
        assert "--config" in cmd

    @patch("subprocess.Popen")
    def test_start_process_without_uv(self, mock_popen: MagicMock, tmp_path: Path):
        """Test starting process without uv."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12346
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file, use_uv=False)

        process_id = manager.start_process(config)

        assert manager.processes[process_id].pid == 12346

        # Verify command doesn't include uv
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "uv" not in cmd
        assert cmd[0] == "python"

    @patch("subprocess.Popen")
    def test_start_process_failure(self, mock_popen: MagicMock, tmp_path: Path):
        """Test handling process start failure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_popen.side_effect = Exception("Failed to start")

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)

        with pytest.raises(RuntimeError, match="Failed to start process"):
            manager.start_process(config)

        # Process should still be tracked as failed
        assert len(manager.processes) == 1
        info = list(manager.processes.values())[0]
        assert info.status == ProcessStatus.FAILED
        assert "Failed to start" in info.error_message  # type: ignore[operator]

    @patch("subprocess.Popen")
    def test_get_process_info(self, mock_popen: MagicMock, tmp_path: Path):
        """Test getting process information."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12347
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)
        process_id = manager.start_process(config)

        info = manager.get_process_info(process_id)

        assert info.process_id == process_id
        assert info.pid == 12347
        assert info.status == ProcessStatus.RUNNING

    def test_get_process_info_not_found(self):
        """Test getting info for nonexistent process."""
        manager = TrainingProcessManager()

        with pytest.raises(ValueError, match="Process not found"):
            manager.get_process_info("nonexistent")

    @patch("subprocess.Popen")
    def test_list_processes(self, mock_popen: MagicMock, tmp_path: Path):
        """Test listing all processes."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        manager = TrainingProcessManager()

        assert manager.list_processes() == []

        # Start first process
        mock_process1 = MagicMock()
        mock_process1.pid = 12348
        mock_popen.return_value = mock_process1
        config = ProcessConfig(config_path=config_file)
        process_id1 = manager.start_process(config)

        # Start second process with different pid
        time.sleep(1.1)  # Ensure different timestamp (format is seconds resolution)
        mock_process2 = MagicMock()
        mock_process2.pid = 12349
        mock_popen.return_value = mock_process2
        process_id2 = manager.start_process(config)

        processes = manager.list_processes()
        assert len(processes) == 2
        assert any(p.process_id == process_id1 for p in processes)
        assert any(p.process_id == process_id2 for p in processes)

    @patch("subprocess.Popen")
    def test_stop_process(self, mock_popen: MagicMock, tmp_path: Path):
        """Test stopping a running process."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12349
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)
        process_id = manager.start_process(config)

        manager.stop_process(process_id)

        info = manager.get_process_info(process_id)
        assert info.status == ProcessStatus.STOPPED
        assert info.end_time is not None
        assert info.exit_code == 0

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    def test_stop_process_not_found(self):
        """Test stopping nonexistent process."""
        manager = TrainingProcessManager()

        with pytest.raises(ValueError, match="Process not found"):
            manager.stop_process("nonexistent")

    @patch("subprocess.Popen")
    def test_stop_process_invalid_state(self, mock_popen: MagicMock, tmp_path: Path):
        """Test stopping process in invalid state."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12350
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)
        process_id = manager.start_process(config)

        # Manually set to stopped
        manager.processes[process_id].status = ProcessStatus.STOPPED

        with pytest.raises(RuntimeError, match="Cannot stop process"):
            manager.stop_process(process_id)

    @patch("subprocess.Popen")
    def test_update_process_status_completed(
        self, mock_popen: MagicMock, tmp_path: Path
    ):
        """Test updating status for completed process."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12351
        mock_process.poll.return_value = 0  # Process completed successfully
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)
        process_id = manager.start_process(config)

        status = manager.update_process_status(process_id)

        assert status == ProcessStatus.COMPLETED
        info = manager.get_process_info(process_id)
        assert info.exit_code == 0

    @patch("subprocess.Popen")
    def test_update_process_status_failed(self, mock_popen: MagicMock, tmp_path: Path):
        """Test updating status for failed process."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12352
        mock_process.poll.return_value = 1  # Process failed
        mock_process.communicate.return_value = (b"", b"Error message")
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)
        process_id = manager.start_process(config)

        status = manager.update_process_status(process_id)

        assert status == ProcessStatus.FAILED
        info = manager.get_process_info(process_id)
        assert info.exit_code == 1

    @patch("subprocess.Popen")
    def test_cleanup_completed(self, mock_popen: MagicMock, tmp_path: Path):
        """Test cleaning up completed processes."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        mock_process = MagicMock()
        mock_process.pid = 12353
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        manager = TrainingProcessManager()
        config = ProcessConfig(config_path=config_file)

        # Start and stop a process
        process_id = manager.start_process(config)
        manager.stop_process(process_id)

        assert len(manager.processes) == 1

        count = manager.cleanup_completed()

        assert count == 1
        assert len(manager.processes) == 0

    def test_update_status_not_found(self):
        """Test updating status for nonexistent process."""
        manager = TrainingProcessManager()

        with pytest.raises(ValueError, match="Process not found"):
            manager.update_process_status("nonexistent")
