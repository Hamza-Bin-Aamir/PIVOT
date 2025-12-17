"""Main training script."""

import argparse


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train PIVOT model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    print(f"Starting training with config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # TODO: Implement training loop
    print("Training not yet implemented")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
