"""Main configuration class for PIVOT."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class TrainConfig:
    """Complete training configuration."""

    # Core settings
    epochs: int = 100
    gradient_clip: float | None = None
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int | None = None

    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0 or None")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    def save(self, save_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            save_path: Path to save configuration
        """
        config_dict = self.to_dict()
        ConfigLoader.save_yaml(config_dict, save_path)


@dataclass
class Config:
    """Main configuration class combining train and inference configs."""

    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment_name: str = "default"
    output_dir: str = "outputs"
    resume_from: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Create configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config object
        """
        config_dict = ConfigLoader.load_yaml(config_path)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        # Extract top-level fields
        experiment_name = config_dict.get("experiment_name", "default")
        output_dir = config_dict.get("output_dir", "outputs")
        resume_from = config_dict.get("resume_from")

        # Parse train configuration
        train_dict = config_dict.get("train", {})
        train_config = cls._parse_train_config(train_dict)

        # Parse inference configuration
        inference_dict = config_dict.get("inference", {})
        inference_config = InferenceConfig(**inference_dict)

        # Parse preprocessing if at top level (for backward compatibility)
        if "preprocessing" in config_dict and "preprocessing" not in train_dict:
            train_config.preprocessing = PreprocessingConfig(**config_dict["preprocessing"])

        return cls(
            train=train_config,
            inference=inference_config,
            experiment_name=experiment_name,
            output_dir=output_dir,
            resume_from=resume_from,
        )

    @staticmethod
    def _parse_train_config(train_dict: dict[str, Any]) -> TrainConfig:
        """Parse training configuration from dictionary.

        Args:
            train_dict: Training configuration dictionary

        Returns:
            TrainConfig object
        """
        # Extract core training settings
        epochs = train_dict.get("epochs", 100)
        gradient_clip = train_dict.get("gradient_clip")
        gradient_accumulation_steps = train_dict.get("gradient_accumulation_steps", 1)
        early_stopping_patience = train_dict.get("early_stopping_patience")

        # Parse model config
        model_dict = train_dict.get("model", {})
        model_config = ModelConfig(**model_dict)

        # Parse data config
        data_config = DataConfig(
            data_dir=train_dict.get("data_dir", "data/processed"),
            batch_size=train_dict.get("batch_size", 2),
            num_workers=train_dict.get("num_workers", 4),
        )

        # Parse preprocessing config
        preprocessing_dict = train_dict.get("preprocessing", {})
        preprocessing_config = PreprocessingConfig(**preprocessing_dict)

        # Parse augmentation config
        augmentation_dict = train_dict.get("augmentation", {})
        augmentation_config = AugmentationConfig(**augmentation_dict)

        # Parse optimizer config
        optimizer_dict = train_dict.get("optimizer", {})
        # Handle learning_rate -> lr mapping
        if "learning_rate" in train_dict:
            optimizer_dict["lr"] = train_dict["learning_rate"]
        optimizer_config = OptimizerConfig(**optimizer_dict)

        # Parse scheduler config
        scheduler_dict = train_dict.get("scheduler", {})
        scheduler_config = SchedulerConfig(**scheduler_dict)

        # Parse loss config
        loss_dict = train_dict.get("loss", {})
        loss_config = LossConfig(**loss_dict)

        # Parse checkpoint config
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=train_dict.get("checkpoint_dir", "checkpoints"),
            save_every=train_dict.get("save_every", 10),
        )

        # Parse logging config
        logging_config = LoggingConfig(
            log_dir=train_dict.get("log_dir", "logs"),
            log_every=train_dict.get("log_every", 10),
        )

        # Parse wandb config
        wandb_dict = train_dict.get("wandb", {})
        wandb_config = WandBConfig(**wandb_dict)

        # Parse validation config
        validation_dict = train_dict.get("validation", {})
        validation_config = ValidationConfig(**validation_dict)

        # Parse hardware config
        hardware_dict = train_dict.get("hardware", {})
        hardware_config = HardwareConfig(**hardware_dict)

        return TrainConfig(
            epochs=epochs,
            gradient_clip=gradient_clip,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            model=model_config,
            data=data_config,
            preprocessing=preprocessing_config,
            augmentation=augmentation_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            loss=loss_config,
            checkpoint=checkpoint_config,
            logging=logging_config,
            wandb=wandb_config,
            validation=validation_config,
            hardware=hardware_config,
        )

    def merge_from_file(self, override_path: str | Path) -> "Config":
        """Merge configuration with overrides from another file.

        Args:
            override_path: Path to override configuration file

        Returns:
            New Config object with merged configuration
        """
        base_dict = asdict(self)
        override_dict = ConfigLoader.load_yaml(override_path)
        merged_dict = ConfigLoader.merge_configs(base_dict, override_dict)
        return Config.from_dict(merged_dict)

    def merge_from_dict(self, override_dict: dict[str, Any]) -> "Config":
        """Merge configuration with overrides from dictionary.

        Args:
            override_dict: Override configuration dictionary

        Returns:
            New Config object with merged configuration
        """
        base_dict = asdict(self)
        merged_dict = ConfigLoader.merge_configs(base_dict, override_dict)
        return Config.from_dict(merged_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    def save(self, save_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            save_path: Path to save configuration
        """
        config_dict = self.to_dict()
        ConfigLoader.save_yaml(config_dict, save_path)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(experiment_name='{self.experiment_name}')"
