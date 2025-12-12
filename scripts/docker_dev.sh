#!/bin/bash
# Start development environment with Jupyter and multi-GPU backend support

set -e

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

# Build image if it doesn't exist
if [[ "$(docker images -q pivot-train-$GPU_BACKEND:latest 2> /dev/null)" == "" ]]; then
    echo "Training image not found. Building..."
    ./scripts/docker_build.sh --train --backend $GPU_BACKEND
fi

# Update docker-compose to use the right backend
# Note: We'll use the default dev service which uses CUDA
# For other backends, users should use docker-compose directly
if [[ "$GPU_BACKEND" != "cuda" ]]; then
    echo "Note: For $GPU_BACKEND backend, use: docker-compose -f docker/docker-compose.yml up train-$GPU_BACKEND"
    echo "Then run Jupyter manually inside the container:"
    echo "  docker-compose -f docker/docker-compose.yml exec train-$GPU_BACKEND jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root"
else
    # Start dev container
    docker-compose -f docker/docker-compose.yml up dev
fi
