#!/usr/bin/env python3
"""Configuration management CLI tool."""

import argparse
import sys
from pathlib import Path

from src.config import Config, ConfigLoader, get_default_train_config


def validate_config(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    try:
        config = Config.from_yaml(args.config)
        print(f"✓ Configuration is valid: {args.config}")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Epochs: {config.train.epochs}")
        print(f"  Batch size: {config.train.data.batch_size}")
        print(f"  Model: {config.train.model.type}")
        print(f"  Device: {config.train.hardware.device}")
        return 0
    except Exception as e:
        print(f"✗ Configuration is invalid: {args.config}")
        print(f"  Error: {str(e)}")
        return 1


def create_config(args: argparse.Namespace) -> int:
    """Create a new configuration file from template."""
    templates = {
        "default": get_default_train_config,
        "fast_dev": lambda: ConfigLoader.load_yaml("configs/fast_dev.yaml"),
        "high_performance": lambda: ConfigLoader.load_yaml("configs/high_performance.yaml"),
        "amd_rocm": lambda: ConfigLoader.load_yaml("configs/amd_rocm.yaml"),
        "intel_xpu": lambda: ConfigLoader.load_yaml("configs/intel_xpu.yaml"),
    }

    if args.template not in templates:
        print(f"✗ Unknown template: {args.template}")
        print(f"  Available templates: {', '.join(templates.keys())}")
        return 1

    try:
        config_dict = templates[args.template]()
        ConfigLoader.save_yaml(config_dict, args.output)
        print(f"✓ Created configuration: {args.output}")
        print(f"  Template: {args.template}")
        return 0
    except Exception as e:
        print("✗ Failed to create configuration")
        print(f"  Error: {str(e)}")
        return 1


def merge_configs(args: argparse.Namespace) -> int:
    """Merge multiple configuration files."""
    try:
        base = ConfigLoader.load_yaml(args.base)
        print(f"Loading base config: {args.base}")

        for override_path in args.overrides:
            override = ConfigLoader.load_yaml(override_path)
            base = ConfigLoader.merge_configs(base, override)
            print(f"Merging: {override_path}")

        ConfigLoader.save_yaml(base, args.output)
        print(f"✓ Merged configuration saved: {args.output}")
        return 0
    except Exception as e:
        print("✗ Failed to merge configurations")
        print(f"  Error: {str(e)}")
        return 1


def show_config(args: argparse.Namespace) -> int:
    """Display configuration in human-readable format."""
    try:
        config = Config.from_yaml(args.config)
        print(f"\n{'=' * 60}")
        print(f"Configuration: {args.config}")
        print(f"{'=' * 60}\n")

        print(f"Experiment: {config.experiment_name}")
        print(f"Output Directory: {config.output_dir}")
        if config.resume_from:
            print(f"Resume From: {config.resume_from}")

        print(f"\n{'-' * 60}")
        print("Training Configuration")
        print(f"{'-' * 60}")
        print(f"  Epochs: {config.train.epochs}")
        print(f"  Batch Size: {config.train.data.batch_size}")
        print(f"  Learning Rate: {config.train.optimizer.lr}")
        print(f"  Device: {config.train.hardware.device}")
        print(f"  Mixed Precision: {config.train.hardware.mixed_precision}")

        print(f"\n{'-' * 60}")
        print("Model Configuration")
        print(f"{'-' * 60}")
        print(f"  Type: {config.train.model.type}")
        print(f"  Input Channels: {config.train.model.in_channels}")
        print(f"  Output Channels: {config.train.model.out_channels}")
        print(f"  Features: {config.train.model.init_features}")
        print(f"  Depth: {config.train.model.depth}")

        print(f"\n{'-' * 60}")
        print("Data Configuration")
        print(f"{'-' * 60}")
        print(f"  Data Directory: {config.train.data.data_dir}")
        print(f"  Num Workers: {config.train.data.num_workers}")
        print(f"  Patch Size: {config.train.preprocessing.patch_size}")
        print(f"  Augmentation: {config.train.augmentation.enabled}")

        print(f"\n{'-' * 60}")
        print("Checkpointing")
        print(f"{'-' * 60}")
        print(f"  Directory: {config.train.checkpoint.checkpoint_dir}")
        print(f"  Save Every: {config.train.checkpoint.save_every} epochs")
        print(
            f"  Track Metric: {config.train.checkpoint.metric_to_track} ({config.train.checkpoint.mode})"
        )

        if config.train.wandb.enabled:
            print(f"\n{'-' * 60}")
            print("Weights & Biases")
            print(f"{'-' * 60}")
            print(f"  Project: {config.train.wandb.project}")
            if config.train.wandb.entity:
                print(f"  Entity: {config.train.wandb.entity}")

        print(f"\n{'-' * 60}")
        print("Inference Configuration")
        print(f"{'-' * 60}")
        print(f"  Batch Size: {config.inference.batch_size}")
        print(f"  Overlap: {config.inference.overlap}")
        print(f"  Threshold: {config.inference.threshold}")
        print(f"  Test-Time Augmentation: {config.inference.tta}")

        print(f"\n{'=' * 60}\n")
        return 0
    except Exception as e:
        print("✗ Failed to load configuration")
        print(f"  Error: {str(e)}")
        return 1


def list_templates(_args: argparse.Namespace) -> int:
    """List available configuration templates."""
    configs_dir = Path("configs")
    templates = list(configs_dir.glob("*.yaml"))

    print(f"\n{'=' * 60}")
    print("Available Configuration Templates")
    print(f"{'=' * 60}\n")

    for template in sorted(templates):
        print(f"  • {template.stem}")
        try:
            config = Config.from_yaml(template)
            print(f"    - Epochs: {config.train.epochs}")
            print(f"    - Batch Size: {config.train.data.batch_size}")
            print(f"    - Device: {config.train.hardware.device}")
        except Exception:
            print("    - (Unable to parse)")
        print()

    print(f"{'=' * 60}\n")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration management for PIVOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config", type=str, help="Path to configuration file")
    validate_parser.set_defaults(func=validate_config)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration file")
    create_parser.add_argument(
        "--template",
        type=str,
        default="default",
        choices=["default", "fast_dev", "high_performance", "amd_rocm", "intel_xpu"],
        help="Template to use",
    )
    create_parser.add_argument("--output", type=str, required=True, help="Output file path")
    create_parser.set_defaults(func=create_config)

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple configuration files")
    merge_parser.add_argument("--base", type=str, required=True, help="Base configuration file")
    merge_parser.add_argument(
        "--overrides",
        type=str,
        nargs="+",
        required=True,
        help="Override configuration files",
    )
    merge_parser.add_argument("--output", type=str, required=True, help="Output file path")
    merge_parser.set_defaults(func=merge_configs)

    # Show command
    show_parser = subparsers.add_parser("show", help="Display configuration")
    show_parser.add_argument("config", type=str, help="Path to configuration file")
    show_parser.set_defaults(func=show_config)

    # List command
    list_parser = subparsers.add_parser("list", help="List available templates")
    list_parser.set_defaults(func=list_templates)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
