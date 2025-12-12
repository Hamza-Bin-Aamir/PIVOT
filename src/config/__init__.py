"""Configuration management."""

from .base import (
    AugmentationConfig,
    CheckpointConfig,
    ConfigLoader,
    DataConfig,
    HardwareConfig,
    InferenceConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    SchedulerConfig,
    ValidationConfig,
    WandBConfig,
)
from .config import Config, TrainConfig
from .defaults import (
    get_amd_rocm_config,
    get_default_train_config,
    get_fast_dev_config,
    get_high_performance_config,
    get_intel_xpu_config,
    get_multi_gpu_config,
)

__all__ = [
    # Main config classes
    "Config",
    "TrainConfig",
    # Component configs
    "ModelConfig",
    "DataConfig",
    "PreprocessingConfig",
    "AugmentationConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "WandBConfig",
    "ValidationConfig",
    "InferenceConfig",
    "HardwareConfig",
    # Utilities
    "ConfigLoader",
    # Default configs
    "get_default_train_config",
    "get_fast_dev_config",
    "get_high_performance_config",
    "get_multi_gpu_config",
    "get_amd_rocm_config",
    "get_intel_xpu_config",
]
