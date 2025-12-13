"""Main inference script."""

import argparse


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run PIVOT inference")
    parser.add_argument("--input", type=str, required=True, help="Input CT scan path")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output path")

    args = parser.parse_args()

    print(f"Running inference on: {args.input}")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to: {args.output}")

    # TODO: Implement inference pipeline
    print("Inference not yet implemented")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
