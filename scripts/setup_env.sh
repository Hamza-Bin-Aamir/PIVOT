#!/usr/bin/env bash
# shellcheck shell=bash
# Script to set up the development environment
set -euo pipefail

echo "Setting up PIVOT development environment..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install development requirements
echo "Installing development requirements..."
pip install -r requirements-dev.txt

# Install package in editable mode
echo "Installing PIVOT package in editable mode..."
pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install --hook-type commit-msg
pre-commit install

# Make hook executable
chmod +x .githooks/check_imports.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run pre-commit checks manually:"
echo "  pre-commit run --all-files"
