#!/usr/bin/env bash
# shellcheck shell=bash
# Run PIVOT training in Docker with multi-GPU backend support

set -euo pipefail

# Default values
CONFIG="configs/train.yaml"
GPU_BACKEND="cuda"
DETACH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --backend)
            GPU_BACKEND="$2"
            shift 2
            ;;
        --detach|-d)
            DETACH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG] [--backend cuda|rocm|intel] [--detach]"
            echo ""
            echo "GPU Backends:"
            echo "  cuda  - NVIDIA GPUs (default)"
            echo "  rocm  - AMD GPUs"
            echo "  intel - Intel GPUs"
            exit 1
            ;;
    esac
done

# Validate GPU backend
if [[ "$GPU_BACKEND" != "cuda" && "$GPU_BACKEND" != "rocm" && "$GPU_BACKEND" != "intel" ]]; then
    echo "Error: Invalid GPU backend '$GPU_BACKEND'"
    echo "Valid options: cuda, rocm, intel"
    exit 1
fi

echo "Starting PIVOT training in Docker..."
echo "Config: $CONFIG"
echo "GPU Backend: $GPU_BACKEND"
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

# Run training
if [[ "$DETACH" == true ]]; then
    "${DOCKER_COMPOSE_CMD[@]}" -f docker/docker-compose.yml up -d train-$GPU_BACKEND
    echo ""
    echo "Training container started in detached mode"
    echo "To view logs: $DOCKER_COMPOSE_STR -f docker/docker-compose.yml logs -f train-$GPU_BACKEND"
    echo "To attach: docker attach pivot-train-$GPU_BACKEND"
else
    "${DOCKER_COMPOSE_CMD[@]}" -f docker/docker-compose.yml run --rm train-$GPU_BACKEND python -m train.main --config "$CONFIG"
fi
