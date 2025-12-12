#!/bin/bash
# Run PIVOT inference in Docker with multi-GPU backend support

set -e

# Default values
INPUT=""
MODEL=""
OUTPUT="./output"
GPU_BACKEND="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --backend)
            GPU_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input INPUT --model MODEL [--output OUTPUT] [--backend cuda|rocm|intel]"
            echo ""
            echo "GPU Backends:"
            echo "  cuda  - NVIDIA GPUs (default)"
            echo "  rocm  - AMD GPUs"
            echo "  intel - Intel GPUs"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT" || -z "$MODEL" ]]; then
    echo "Error: --input and --model are required"
    echo "Usage: $0 --input INPUT --model MODEL [--output OUTPUT] [--backend cuda|rocm|intel]"
    exit 1
fi

# Validate GPU backend
if [[ "$GPU_BACKEND" != "cuda" && "$GPU_BACKEND" != "rocm" && "$GPU_BACKEND" != "intel" ]]; then
    echo "Error: Invalid GPU backend '$GPU_BACKEND'"
    echo "Valid options: cuda, rocm, intel"
    exit 1
fi

echo "Running PIVOT inference in Docker..."
echo "Input: $INPUT"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "GPU Backend: $GPU_BACKEND"
echo ""

# Build image if it doesn't exist
if [[ "$(docker images -q pivot-inference-$GPU_BACKEND:latest 2> /dev/null)" == "" ]]; then
    echo "Inference image not found. Building..."
    ./scripts/docker_build.sh --inference --backend $GPU_BACKEND
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT"

# Set device arguments based on backend
case $GPU_BACKEND in
    cuda)
        DEVICE_ARGS="--gpus all"
        ;;
    rocm)
        DEVICE_ARGS="--device=/dev/kfd --device=/dev/dri --group-add video"
        ;;
    intel)
        DEVICE_ARGS="--device=/dev/dri"
        ;;
esac

# Run inference
docker run --rm $DEVICE_ARGS \
    -v "$(realpath "$INPUT"):/app/input:ro" \
    -v "$(realpath "$MODEL"):/app/models/model.pth:ro" \
    -v "$(realpath "$OUTPUT"):/app/output" \
    pivot-inference-$GPU_BACKEND:latest \
    python -m inference.main \
    --input /app/input \
    --model /app/models/model.pth \
    --output /app/output

echo ""
echo "✅ Inference completed successfully"
echo "Results saved to: $OUTPUT"
