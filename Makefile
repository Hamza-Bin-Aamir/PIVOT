# Makefile for PIVOT project

.PHONY: help install install-dev test lint format clean setup

help:
	@echo "PIVOT Project Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Run full environment setup"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Run linting checks (flake8, mypy)"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean up build artifacts and cache files"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

setup:
	bash scripts/setup_env.sh

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .coverage htmlcov/
	@echo "Cleaned up build artifacts and cache files"
