#!/bin/bash
# Detect available GPU backend on the system

set -e

echo "PIVOT GPU Backend Detection"
echo "==========================="
echo ""

DETECTED_BACKEND=""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    DETECTED_BACKEND="cuda"
    echo ""
fi

# Check for AMD GPU
if [ -e /dev/kfd ] && command -v rocm-smi &> /dev/null; then
    echo "✅ AMD GPU (ROCm) detected"
    rocm-smi --showproductname || echo "ROCm installed"
    DETECTED_BACKEND="rocm"
    echo ""
elif [ -e /dev/kfd ]; then
    echo "⚠️  AMD GPU device detected but ROCm not installed"
    echo "   Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    DETECTED_BACKEND="rocm"
    echo ""
fi

# Check for Intel GPU
if [ -e /dev/dri ]; then
    if lspci | grep -i "vga.*intel" &> /dev/null; then
        echo "✅ Intel GPU detected"
        lspci | grep -i "vga.*intel"
        if [ -z "$DETECTED_BACKEND" ]; then
            DETECTED_BACKEND="intel"
        fi
        echo ""
    fi
fi

# Provide recommendation
echo "Recommendation:"
echo "==============="
if [ -n "$DETECTED_BACKEND" ]; then
    echo "Use GPU backend: $DETECTED_BACKEND"
    echo ""
    echo "Build command:"
    echo "  bash scripts/docker_build.sh --all --backend $DETECTED_BACKEND"
    echo ""
    echo "Training command:"
    echo "  bash scripts/docker_train.sh --backend $DETECTED_BACKEND"
    echo ""
    echo "Inference command:"
    echo "  bash scripts/docker_inference.sh --backend $DETECTED_BACKEND --input <input> --model <model>"
else
    echo "No GPU detected. PIVOT can run on CPU but training will be very slow."
    echo "For CPU-only deployment, use the CUDA images (they support CPU fallback)."
fi

echo ""
echo "Supported backends:"
echo "  cuda  - NVIDIA GPUs (requires nvidia-docker)"
echo "  rocm  - AMD GPUs (requires ROCm and /dev/kfd, /dev/dri)"
echo "  intel - Intel GPUs (requires Intel OneAPI and /dev/dri)"
