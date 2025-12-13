#!/usr/bin/env bash
# shellcheck shell=bash
# Start development environment with Jupyter and multi-GPU backend support

set -euo pipefail

PORT=${1:-8888}
GPU_BACKEND=${2:-cuda}

# Validate GPU backend
if [[ "$GPU_BACKEND" != "cuda" && "$GPU_BACKEND" != "rocm" && "$GPU_BACKEND" != "intel" ]]; then
    echo "Error: Invalid GPU backend '$GPU_BACKEND'"
    echo "Usage: $0 [PORT] [BACKEND]"
    echo ""
    echo "GPU Backends:"
    echo "  cuda  - NVIDIA GPUs (default)"
    echo "  rocm  - AMD GPUs"
    echo "  intel - Intel GPUs"
    exit 1
fi

echo "Starting PIVOT development environment..."
echo "GPU Backend: $GPU_BACKEND"
echo "Jupyter will be available at: http://localhost:$PORT"
echo ""

if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker-compose)
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD=(docker compose)
else
    echo "Error: Docker Compose is not installed. Install docker-compose or Docker Compose v2." >&2
    exit 1
fi

DOCKER_COMPOSE_STR=${DOCKER_COMPOSE_CMD[*]}

# Build image if it doesn't exist
if [[ "$(docker images -q pivot-train-$GPU_BACKEND:latest 2> /dev/null)" == "" ]]; then
    echo "Training image not found. Building..."
    ./scripts/docker_build.sh --train --backend "$GPU_BACKEND"
fi

# Update docker-compose to use the right backend
DEV_SERVICE="dev"

case "$GPU_BACKEND" in
    cuda)
        DEV_SERVICE="dev"
        ;;
    rocm)
        DEV_SERVICE="dev-rocm"
        ;;
    intel)
        DEV_SERVICE="dev-intel"
        ;;
esac

if [[ "$GPU_BACKEND" != "cuda" ]]; then
    echo "Launching $DEV_SERVICE service for backend '$GPU_BACKEND'"
fi

"${DOCKER_COMPOSE_CMD[@]}" -f docker/docker-compose.yml up "$DEV_SERVICE"
