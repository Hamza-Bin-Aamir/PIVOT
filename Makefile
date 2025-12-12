# Makefile for PIVOT project

.PHONY: help install install-dev test lint format clean setup docker-detect-gpu docker-build docker-build-cuda docker-build-rocm docker-build-intel docker-train docker-inference docker-dev docker-clean

help:
	@echo "PIVOT Project Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  setup           - Run full environment setup"
	@echo "  test            - Run tests with pytest"
	@echo "  lint            - Run linting checks (ruff, mypy)"
	@echo "  format          - Format code with ruff"
	@echo "  clean           - Clean up build artifacts and cache files"
	@echo ""
	@echo "Docker targets (set GPU_BACKEND=cuda|rocm|intel, default: cuda):"
	@echo "  docker-detect-gpu  - Detect available GPU backend"
	@echo "  docker-build       - Build Docker images for specified backend"
	@echo "  docker-build-cuda  - Build Docker images for NVIDIA GPUs"
	@echo "  docker-build-rocm  - Build Docker images for AMD GPUs"
	@echo "  docker-build-intel - Build Docker images for Intel GPUs"
	@echo "  docker-train       - Run training in Docker"
	@echo "  docker-inference   - Run inference in Docker (requires INPUT and MODEL)"
	@echo "  docker-dev         - Start Jupyter development environment"
	@echo "  docker-clean       - Stop and remove Docker containers"
	@echo ""
	@echo "Examples:"
	@echo "  make docker-detect-gpu"
	@echo "  make docker-build GPU_BACKEND=rocm"
	@echo "  make docker-train GPU_BACKEND=intel"
	@echo "  make docker-inference INPUT=./data/test.mhd MODEL=./model.pth GPU_BACKEND=cuda"
	@echo ""

install:
	uv sync

install-dev:
	uv sync --all-extras

setup:
	@echo "Setting up PIVOT development environment with uv..."
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync --all-extras
	uv run pre-commit install --hook-type commit-msg
	uv run pre-commit install
	@echo "Setup complete!"

test:
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .coverage htmlcov/
	@echo "Cleaned up build artifacts and cache files"

# Docker targets
GPU_BACKEND ?= cuda

docker-detect-gpu:
	@bash scripts/detect_gpu.sh

docker-build:
	@echo "Building Docker images for $(GPU_BACKEND) backend..."
	bash scripts/docker_build.sh --all --backend $(GPU_BACKEND)

docker-build-cuda:
	bash scripts/docker_build.sh --all --backend cuda

docker-build-rocm:
	bash scripts/docker_build.sh --all --backend rocm

docker-build-intel:
	bash scripts/docker_build.sh --all --backend intel

docker-train:
	bash scripts/docker_train.sh --backend $(GPU_BACKEND) --config configs/train.yaml

docker-inference:
	@echo "Usage: make docker-inference INPUT=<path> MODEL=<path> [OUTPUT=<path>] [GPU_BACKEND=cuda|rocm|intel]"
	@if [ -z "$(INPUT)" ] || [ -z "$(MODEL)" ]; then \
		echo "Error: INPUT and MODEL are required"; \
		echo "Example: make docker-inference INPUT=./data/raw/scan.mhd MODEL=./checkpoints/model.pth GPU_BACKEND=cuda"; \
		exit 1; \
	fi
	bash scripts/docker_inference.sh --backend $(GPU_BACKEND) --input $(INPUT) --model $(MODEL) --output $(or $(OUTPUT),./output)

docker-dev:
	bash scripts/docker_dev.sh 8888 $(GPU_BACKEND)

docker-clean:
	docker-compose -f docker/docker-compose.yml down
	@echo "Docker containers stopped and removed"
