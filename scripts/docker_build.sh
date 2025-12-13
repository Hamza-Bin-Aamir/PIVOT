#!/usr/bin/env bash
# shellcheck shell=bash
# Build Docker images for PIVOT with multi-GPU backend support

set -euo pipefail

echo "Building PIVOT Docker Images"
echo "============================"
echo ""

# Parse arguments
BUILD_TRAIN=false
BUILD_INFERENCE=false
BUILD_ALL=false
GPU_BACKEND="cuda"  # Default to NVIDIA CUDA

while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            BUILD_TRAIN=true
            shift
            ;;
        --inference)
            BUILD_INFERENCE=true
            shift
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        --backend)
            GPU_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--train] [--inference] [--all] [--backend cuda|rocm|intel]"
            echo ""
            echo "GPU Backends:"
            echo "  cuda  - NVIDIA GPUs (default)"
            echo "  rocm  - AMD GPUs"
            echo "  intel - Intel GPUs (integrated and dedicated)"
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

# If no arguments, build all
if [[ "$BUILD_TRAIN" == false && "$BUILD_INFERENCE" == false && "$BUILD_ALL" == false ]]; then
    BUILD_ALL=true
fi

echo "GPU Backend: $GPU_BACKEND"
echo ""

if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_STR="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_STR="docker compose"
else
    DOCKER_COMPOSE_STR="docker compose"
fi

# Build training image
if [[ "$BUILD_TRAIN" == true || "$BUILD_ALL" == true ]]; then
    echo "Building training image for $GPU_BACKEND..."
    docker build -f docker/Dockerfile.train.$GPU_BACKEND -t pivot-train-$GPU_BACKEND:latest .
    echo "✅ Training image built successfully"
    echo ""
fi

# Build inference image
if [[ "$BUILD_INFERENCE" == true || "$BUILD_ALL" == true ]]; then
    echo "Building inference image for $GPU_BACKEND..."
    docker build -f docker/Dockerfile.inference.$GPU_BACKEND -t pivot-inference-$GPU_BACKEND:latest .
    echo "✅ Inference image built successfully"
    echo ""
fi

echo "Docker images built successfully!"
echo ""
echo "To view images:"
echo "  docker images | grep pivot"
echo ""
echo "To run with docker-compose:"
echo "  $DOCKER_COMPOSE_STR up -d train-$GPU_BACKEND      # Start training container"
echo "  $DOCKER_COMPOSE_STR up -d inference-$GPU_BACKEND  # Start inference container"
echo ""
echo "GPU Backend Information:"
case $GPU_BACKEND in
    cuda)
        echo "  NVIDIA CUDA - Requires nvidia-docker runtime"
        echo "  Test GPU: docker run --rm --gpus all pivot-train-cuda:latest nvidia-smi"
        ;;
    rocm)
        echo "  AMD ROCm - Requires /dev/kfd and /dev/dri devices"
        echo "  Test GPU: docker run --rm --device=/dev/kfd --device=/dev/dri pivot-train-rocm:latest rocm-smi"
        ;;
    intel)
        echo "  Intel OneAPI - Requires /dev/dri device"
        echo "  Test GPU: docker run --rm --device=/dev/dri pivot-train-intel:latest bash -c 'source /opt/intel/oneapi/setvars.sh && sycl-ls'"
        ;;
esac
