"""Base configuration classes and utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    type: str = "unet3d"
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 32
    depth: int = 4
    dropout: float = 0.0
    batch_norm: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if self.init_features < 1:
            raise ValueError("init_features must be >= 1")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adam"
    lr: float = 0.0001
    weight_decay: float = 0.00001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9  # For SGD
    nesterov: bool = False  # For SGD

    def __post_init__(self):
        """Validate configuration."""
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-6
    step_size: int = 30  # For StepLR
    gamma: float = 0.1  # For StepLR/MultiStepLR
    milestones: list[int] = field(default_factory=lambda: [30, 60, 90])  # For MultiStepLR

    def __post_init__(self):
        """Validate configuration."""
        if self.min_lr <= 0:
            raise ValueError("min_lr must be > 0")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")


@dataclass
class LossConfig:
    """Loss function configuration."""

    type: str = "dice"
    weight: list[float] | None = None
    smooth: float = 1e-5
    focal_alpha: float = 0.25  # For focal loss
    focal_gamma: float = 2.0  # For focal loss
    reduction: str = "mean"

    def __post_init__(self):
        """Validate configuration."""
        if self.smooth < 0:
            raise ValueError("smooth must be >= 0")
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data/processed"
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True
    cache_rate: float = 0.0  # MONAI cache rate (0=no cache, 1=full cache)

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if not 0 <= self.cache_rate <= 1:
            raise ValueError("cache_rate must be in [0, 1]")


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""

    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    window_center: int = -600
    window_width: int = 1500
    patch_size: tuple[int, int, int] = (96, 96, 96)
    normalize: bool = True
    clip_values: tuple[float, float] | None = None

    def __post_init__(self):
        """Validate configuration."""
        if len(self.target_spacing) != 3:
            raise ValueError("target_spacing must have 3 values")
        if len(self.patch_size) != 3:
            raise ValueError("patch_size must have 3 values")


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    enabled: bool = True
    random_flip_prob: float = 0.5
    random_rotate_prob: float = 0.5
    random_scale_prob: float = 0.5
    random_intensity_shift_prob: float = 0.5
    random_intensity_scale_prob: float = 0.5
    elastic_deform_prob: float = 0.0
    gaussian_noise_prob: float = 0.0
    gaussian_smooth_prob: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        probs = [
            self.random_flip_prob,
            self.random_rotate_prob,
            self.random_scale_prob,
            self.random_intensity_shift_prob,
            self.random_intensity_scale_prob,
            self.elastic_deform_prob,
            self.gaussian_noise_prob,
            self.gaussian_smooth_prob,
        ]
        for prob in probs:
            if not 0 <= prob <= 1:
                raise ValueError("All probabilities must be in [0, 1]")


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    save_best: bool = True
    metric_to_track: str = "val_dice"
    mode: str = "max"  # 'max' or 'min'
    save_last: bool = True
    max_checkpoints: int | None = 5

    def __post_init__(self):
        """Validate configuration."""
        if self.save_every < 1:
            raise ValueError("save_every must be >= 1")
        if self.mode not in ["max", "min"]:
            raise ValueError("mode must be 'max' or 'min'")
        if self.max_checkpoints is not None and self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be >= 1 or None")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "logs"
    log_every: int = 10
    tensorboard: bool = True
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    def __post_init__(self):
        """Validate configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.console_log_level not in valid_levels:
            raise ValueError(f"console_log_level must be one of {valid_levels}")
        if self.file_log_level not in valid_levels:
            raise ValueError(f"file_log_level must be one of {valid_levels}")


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "pivot"
    entity: str = ""
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    resume: str = "auto"  # 'auto', 'allow', 'never', 'must'

    def __post_init__(self):
        """Validate configuration."""
        if self.resume not in ["auto", "allow", "never", "must"]:
            raise ValueError("resume must be 'auto', 'allow', 'never', or 'must'")


@dataclass
class ValidationConfig:
    """Validation configuration."""

    val_every: int = 1
    val_batch_size: int | None = None  # If None, uses train batch_size
    compute_metrics: bool = True
    save_predictions: bool = False
    prediction_save_dir: str = "predictions"

    def __post_init__(self):
        """Validate configuration."""
        if self.val_every < 1:
            raise ValueError("val_every must be >= 1")


@dataclass
class InferenceConfig:
    """Inference configuration."""

    batch_size: int = 1
    overlap: float = 0.5
    threshold: float = 0.5
    sliding_window: bool = True
    save_probabilities: bool = False
    tta: bool = False  # Test-time augmentation

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    device: str = "cuda"  # 'cuda', 'rocm', 'xpu', 'cpu'
    gpu_ids: list[int] | None = None
    mixed_precision: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    seed: int | None = 42

    def __post_init__(self):
        """Validate configuration."""
        valid_devices = ["cuda", "rocm", "xpu", "cpu"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")


class ConfigLoader:
    """Utility class for loading and merging configurations."""

    @staticmethod
    def load_yaml(config_path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary containing configuration
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config or {}

    @staticmethod
    def save_yaml(config: dict[str, Any], save_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override config into base config.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged
