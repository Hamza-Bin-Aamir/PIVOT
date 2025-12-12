"""Tests for configuration management system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AugmentationConfig,
    Config,
    ConfigLoader,
    DataConfig,
    HardwareConfig,
    InferenceConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessingConfig,
    get_default_train_config,
    get_fast_dev_config,
    get_high_performance_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.type == "unet3d"
        assert config.in_channels == 1
        assert config.out_channels == 1
        assert config.init_features == 32
        assert config.depth == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            type="custom",
            in_channels=3,
            out_channels=2,
            init_features=64,
            depth=5,
        )
        assert config.type == "custom"
        assert config.in_channels == 3
        assert config.out_channels == 2
        assert config.init_features == 64
        assert config.depth == 5

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            ModelConfig(in_channels=0)
        with pytest.raises(ValueError):
            ModelConfig(out_channels=-1)
        with pytest.raises(ValueError):
            ModelConfig(depth=0)


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizerConfig()
        assert config.type == "adam"
        assert config.lr == 0.0001
        assert config.weight_decay == 0.00001

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            OptimizerConfig(lr=0)
        with pytest.raises(ValueError):
            OptimizerConfig(lr=-0.001)
        with pytest.raises(ValueError):
            OptimizerConfig(weight_decay=-0.1)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.data_dir == "data/processed"
        assert config.batch_size == 2
        assert config.num_workers == 4

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            DataConfig(batch_size=0)
        with pytest.raises(ValueError):
            DataConfig(num_workers=-1)
        with pytest.raises(ValueError):
            DataConfig(cache_rate=1.5)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.target_spacing == (1.0, 1.0, 1.0)
        assert config.window_center == -600
        assert config.window_width == 1500
        assert config.patch_size == (96, 96, 96)

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            PreprocessingConfig(target_spacing=(1.0, 1.0))  # Must be 3D
        with pytest.raises(ValueError):
            PreprocessingConfig(patch_size=(96, 96))  # Must be 3D


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_yaml(self):
        """Test loading YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value", "nested": {"key2": 123}}, f)
            temp_path = f.name

        try:
            config = ConfigLoader.load_yaml(temp_path)
            assert config["key"] == "value"
            assert config["nested"]["key2"] == 123
        finally:
            Path(temp_path).unlink()

    def test_load_yaml_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_yaml("nonexistent.yaml")

    def test_save_yaml(self):
        """Test saving YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_config.yaml"
            config = {"key": "value", "nested": {"key2": 123}}

            ConfigLoader.save_yaml(config, save_path)

            assert save_path.exists()
            loaded = ConfigLoader.load_yaml(save_path)
            assert loaded == config

    def test_merge_configs(self):
        """Test merging configurations."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": 4,
        }
        override = {
            "b": {"c": 10},
            "e": 5,
            "f": 6,
        }

        merged = ConfigLoader.merge_configs(base, override)

        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5
        assert merged["f"] == 6


class TestConfig:
    """Tests for main Config class."""

    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        config_dict = {
            "experiment_name": "test_exp",
            "train": {
                "epochs": 50,
                "batch_size": 4,
                "model": {"type": "unet3d", "in_channels": 1},
            },
            "inference": {"batch_size": 1},
        }

        config = Config.from_dict(config_dict)

        assert config.experiment_name == "test_exp"
        assert config.train.epochs == 50
        assert config.train.data.batch_size == 4
        assert config.train.model.type == "unet3d"
        assert config.inference.batch_size == 1

    def test_from_yaml(self):
        """Test creating config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_dict = {
                "experiment_name": "yaml_test",
                "train": {
                    "epochs": 100,
                    "model": {"type": "unet3d"},
                },
                "inference": {"batch_size": 1},
            }
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            assert config.experiment_name == "yaml_test"
            assert config.train.epochs == 100
        finally:
            Path(temp_path).unlink()

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "train" in config_dict
        assert "inference" in config_dict
        assert "experiment_name" in config_dict

    def test_save(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config.yaml"
            config = Config(experiment_name="save_test")
            config.save(save_path)

            assert save_path.exists()
            loaded = Config.from_yaml(save_path)
            assert loaded.experiment_name == "save_test"

    def test_merge_from_dict(self):
        """Test merging config with dictionary."""
        config = Config(experiment_name="base")
        override = {
            "experiment_name": "merged",
            "train": {"epochs": 200},
        }

        merged = config.merge_from_dict(override)

        assert merged.experiment_name == "merged"
        assert merged.train.epochs == 200

    def test_backward_compatibility_learning_rate(self):
        """Test backward compatibility with learning_rate field."""
        config_dict = {
            "train": {
                "learning_rate": 0.001,  # Old field name
                "model": {"type": "unet3d"},
            }
        }

        config = Config.from_dict(config_dict)
        assert config.train.optimizer.lr == 0.001


class TestDefaultConfigs:
    """Tests for default configuration templates."""

    def test_default_train_config(self):
        """Test default training config."""
        config_dict = get_default_train_config()

        assert "train" in config_dict
        assert "inference" in config_dict
        assert config_dict["train"]["epochs"] == 100
        assert config_dict["train"]["model"]["type"] == "unet3d"

    def test_fast_dev_config(self):
        """Test fast dev config."""
        config_dict = get_fast_dev_config()

        assert config_dict["experiment_name"] == "fast_dev"
        assert config_dict["train"]["epochs"] == 2
        assert config_dict["train"]["batch_size"] == 1
        assert config_dict["train"]["augmentation"]["enabled"] is False

    def test_high_performance_config(self):
        """Test high performance config."""
        config_dict = get_high_performance_config()

        assert config_dict["experiment_name"] == "high_performance"
        assert config_dict["train"]["epochs"] == 300
        assert config_dict["train"]["batch_size"] == 4
        assert config_dict["train"]["data"]["cache_rate"] == 1.0


class TestAugmentationConfig:
    """Tests for AugmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        assert config.enabled is True
        assert config.random_flip_prob == 0.5

    def test_validation(self):
        """Test probability validation."""
        with pytest.raises(ValueError):
            AugmentationConfig(random_flip_prob=1.5)
        with pytest.raises(ValueError):
            AugmentationConfig(random_rotate_prob=-0.1)


class TestHardwareConfig:
    """Tests for HardwareConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HardwareConfig()
        assert config.device == "cuda"
        assert config.mixed_precision is True
        assert config.seed == 42

    def test_validation(self):
        """Test device validation."""
        with pytest.raises(ValueError):
            HardwareConfig(device="invalid")

        # Valid devices
        HardwareConfig(device="cuda")
        HardwareConfig(device="rocm")
        HardwareConfig(device="xpu")
        HardwareConfig(device="cpu")


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceConfig()
        assert config.batch_size == 1
        assert config.overlap == 0.5
        assert config.threshold == 0.5

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            InferenceConfig(overlap=1.5)
        with pytest.raises(ValueError):
            InferenceConfig(threshold=-0.1)
