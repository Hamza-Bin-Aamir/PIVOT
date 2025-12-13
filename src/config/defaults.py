"""Default configuration templates."""

from typing import Any


def get_default_train_config() -> dict[str, Any]:
    """Get default training configuration.

    Returns:
        Default training configuration dictionary
    """
    return {
        "experiment_name": "default_experiment",
        "output_dir": "outputs",
        "train": {
            "epochs": 100,
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 1,
            "early_stopping_patience": None,
            "data_dir": "data/processed",
            "batch_size": 2,
            "num_workers": 4,
            "data": {
                "pin_memory": True,
                "shuffle_train": True,
                "drop_last": True,
                "cache_rate": 0.0,
            },
            "model": {
                "type": "unet3d",
                "in_channels": 1,
                "out_channels": 1,
                "init_features": 32,
                "depth": 4,
                "dropout": 0.0,
                "batch_norm": True,
            },
            "preprocessing": {
                "target_spacing": [1.0, 1.0, 1.0],
                "window_center": -600,
                "window_width": 1500,
                "patch_size": [96, 96, 96],
                "normalize": True,
                "clip_values": None,
            },
            "augmentation": {
                "enabled": True,
                "random_flip_prob": 0.5,
                "random_rotate_prob": 0.5,
                "random_scale_prob": 0.5,
                "random_intensity_shift_prob": 0.5,
                "random_intensity_scale_prob": 0.5,
                "elastic_deform_prob": 0.0,
                "gaussian_noise_prob": 0.0,
                "gaussian_smooth_prob": 0.0,
            },
            "optimizer": {
                "type": "adam",
                "lr": 0.0001,
                "weight_decay": 0.00001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {
                "type": "cosine",
                "min_lr": 1e-6,
                "warmup_epochs": 0,
                "warmup_start_lr": 1e-6,
            },
            "loss": {
                "type": "dice",
                "smooth": 1e-5,
                "reduction": "mean",
            },
            "checkpoint": {
                "checkpoint_dir": "checkpoints",
                "save_every": 10,
                "save_best": True,
                "metric_to_track": "val_dice",
                "mode": "max",
                "save_last": True,
                "max_checkpoints": 5,
            },
            "logging": {
                "log_dir": "logs",
                "log_every": 10,
                "tensorboard": True,
                "console_log_level": "INFO",
                "file_log_level": "DEBUG",
            },
            "wandb": {
                "enabled": False,
                "project": "pivot",
                "entity": "",
                "tags": [],
                "notes": "",
                "resume": "auto",
            },
            "validation": {
                "val_every": 1,
                "val_batch_size": None,
                "compute_metrics": True,
                "save_predictions": False,
                "prediction_save_dir": "predictions",
            },
            "hardware": {
                "device": "cuda",
                "gpu_ids": None,
                "mixed_precision": True,
                "cudnn_benchmark": True,
                "deterministic": False,
                "seed": 42,
            },
        },
        "inference": {
            "batch_size": 1,
            "overlap": 0.5,
            "threshold": 0.5,
            "sliding_window": True,
            "save_probabilities": False,
            "tta": False,
        },
    }


def get_fast_dev_config() -> dict[str, Any]:
    """Get fast development configuration for quick testing.

    Returns:
        Fast development configuration dictionary
    """
    config = get_default_train_config()
    config["experiment_name"] = "fast_dev"
    config["train"]["epochs"] = 2
    config["train"]["batch_size"] = 1
    config["train"]["num_workers"] = 0
    config["train"]["save_every"] = 1
    config["train"]["log_every"] = 1
    config["train"]["validation"]["val_every"] = 1
    config["train"]["augmentation"]["enabled"] = False
    config["train"].setdefault("data", {})
    config["train"]["data"]["cache_rate"] = 0.0
    return config


def get_high_performance_config() -> dict[str, Any]:
    """Get high-performance configuration for production training.

    Returns:
        High-performance configuration dictionary
    """
    config = get_default_train_config()
    config["experiment_name"] = "high_performance"
    config["train"]["batch_size"] = 4
    config["train"]["num_workers"] = 8
    config["train"]["epochs"] = 300
    config["train"]["gradient_accumulation_steps"] = 2
    config["train"]["early_stopping_patience"] = 30
    config["train"].setdefault("data", {})
    config["train"]["data"]["cache_rate"] = 1.0  # Full caching
    config["train"]["data"]["pin_memory"] = True
    config["train"]["hardware"]["mixed_precision"] = True
    config["train"]["hardware"]["cudnn_benchmark"] = True
    config["train"]["checkpoint"]["max_checkpoints"] = 10
    config["train"]["validation"]["val_every"] = 5
    return config


def get_multi_gpu_config() -> dict[str, Any]:
    """Get multi-GPU training configuration.

    Returns:
        Multi-GPU configuration dictionary
    """
    config = get_high_performance_config()
    config["experiment_name"] = "multi_gpu"
    config["train"]["batch_size"] = 2  # Per GPU
    config["train"]["hardware"]["gpu_ids"] = [0, 1, 2, 3]
    return config


def get_amd_rocm_config() -> dict[str, Any]:
    """Get AMD ROCm optimized configuration.

    Returns:
        AMD ROCm configuration dictionary
    """
    config = get_default_train_config()
    config["experiment_name"] = "amd_rocm"
    config["train"]["hardware"]["device"] = "cuda"  # ROCm uses same backend
    config["train"]["hardware"]["mixed_precision"] = True
    config["train"]["batch_size"] = 2
    return config


def get_intel_xpu_config() -> dict[str, Any]:
    """Get Intel XPU optimized configuration.

    Returns:
        Intel XPU configuration dictionary
    """
    config = get_default_train_config()
    config["experiment_name"] = "intel_xpu"
    config["train"]["hardware"]["device"] = "xpu"
    config["train"]["hardware"]["mixed_precision"] = False  # XPU has different AMP
    config["train"]["batch_size"] = 2
    return config
