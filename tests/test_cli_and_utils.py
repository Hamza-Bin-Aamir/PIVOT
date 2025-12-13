"""Tests covering CLI entry points and shared utilities."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

from src.inference import main as inference_main
from src.train import main as train_main
from src.utils.logger import setup_logger


def _cleanup_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_package_inits_are_importable() -> None:
    for module_name in ["src.inference", "src.model", "src.train", "src.utils"]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "__all__")


def test_inference_main_emits_messages(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference",
            "--input",
            "scan.nii.gz",
            "--model",
            "checkpoint.pt",
            "--output",
            "predictions.nii.gz",
        ],
    )

    inference_main.main()

    captured = capsys.readouterr()
    assert "scan.nii.gz" in captured.out
    assert "checkpoint.pt" in captured.out
    assert "predictions.nii.gz" in captured.out


def test_train_main_emits_messages(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/train.yaml",
            "--data_dir",
            "data/processed",
            "--output_dir",
            "checkpoints",
        ],
    )

    train_main.main()

    captured = capsys.readouterr()
    assert "configs/train.yaml" in captured.out
    assert "data/processed" in captured.out
    assert "checkpoints" in captured.out


def test_setup_logger_console_only() -> None:
    logger = setup_logger("test_console_logger")
    try:
        assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    finally:
        _cleanup_logger(logger)


def test_setup_logger_with_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "run.log"
    logger = setup_logger("test_file_logger", log_file=log_path)
    try:
        assert log_path.exists()
        assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    finally:
        _cleanup_logger(logger)
