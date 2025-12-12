# Docker Guide for PIVOT

This guide provides detailed information about using Docker with PIVOT for training and inference across multiple GPU backends.

## Multi-GPU Backend Support

PIVOT supports training and inference on:

1. **NVIDIA GPUs** (CUDA) - Consumer and datacenter GPUs
2. **AMD GPUs** (ROCm) - Integrated and dedicated GPUs
3. **Intel GPUs** (OneAPI) - Integrated GPUs (11th gen+) and Arc series

## Prerequisites

### Common Requirements
- Docker Engine 20.10+
- Docker Compose 1.29+

### GPU-Specific Requirements

#### NVIDIA (CUDA)
- NVIDIA GPU with CUDA 12.1+ support
- NVIDIA Docker runtime

Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### AMD (ROCm)
- AMD GPU (Vega, RDNA, or newer architecture)
- ROCm 5.7+ installed on host

Install ROCm:
```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
sudo usermod -a -G render,video $USER
```

Verify:
```bash
rocm-smi
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

#### Intel (OneAPI)
- Intel integrated GPU (11th gen or newer) or Arc GPU
- Intel GPU drivers installed

Install Intel GPU drivers:
```bash
# Ubuntu 22.04
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
  sudo tee /etc/apt/sources.list.d/intel-graphics.list
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
```

Verify:
```bash
ls -la /dev/dri
```

## Detecting Your GPU

Use the built-in detection script to identify your GPU:

```bash
bash scripts/detect_gpu.sh
```

This will detect available GPUs and recommend the appropriate backend.

## Docker Images

PIVOT provides separate images for each GPU backend:

### Training Images
| Backend | Image Name | Base Image | Size |
|---------|-----------|------------|------|
| NVIDIA  | `pivot-train-cuda` | nvidia/cuda:12.1.0-cudnn8-runtime | ~8GB |
| AMD     | `pivot-train-rocm` | rocm/pytorch:rocm5.7 | ~10GB |
| Intel   | `pivot-train-intel` | intel/oneapi-basekit:2024.0.0 | ~12GB |

### Inference Images
| Backend | Image Name | Base Image | Size |
|---------|-----------|------------|------|
| NVIDIA  | `pivot-inference-cuda` | nvidia/cuda:12.1.0-cudnn8-runtime | ~4GB |
| AMD     | `pivot-inference-rocm` | rocm/pytorch:rocm5.7 | ~6GB |
| Intel   | `pivot-inference-intel` | intel/oneapi-basekit:2024.0.0-runtime | ~7GB |

All images include:
- PyTorch with appropriate GPU support
- MONAI for medical imaging
- All PIVOT dependencies

## Building Images

### Auto-detect and Build
```bash
# Detect GPU and get recommendations
bash scripts/detect_gpu.sh

# Build for detected backend (replace 'cuda' with your backend)
bash scripts/docker_build.sh --all --backend cuda
```

### Build All Images for Specific Backend
```bash
bash scripts/docker_build.sh --all --backend cuda   # NVIDIA
bash scripts/docker_build.sh --all --backend rocm   # AMD
bash scripts/docker_build.sh --all --backend intel  # Intel
```

### Build Specific Images
```bash
# Training image
bash scripts/docker_build.sh --train --backend cuda
bash scripts/docker_build.sh --train --backend rocm
bash scripts/docker_build.sh --train --backend intel

# Inference image
bash scripts/docker_build.sh --inference --backend cuda
bash scripts/docker_build.sh --inference --backend rocm
bash scripts/docker_build.sh --inference --backend intel
```

### Using Makefile
```bash
make docker-build  # Builds CUDA images by default
```

## Running Containers

### Training

#### NVIDIA CUDA
```bash
# Interactive training session
docker-compose -f docker/docker-compose.yml run --rm train-cuda bash

# Inside the container:
python -m train.main --config configs/train.yaml

# Using helper script
bash scripts/docker_train.sh --backend cuda --config configs/train.yaml

# Background training
docker-compose -f docker/docker-compose.yml up -d train-cuda
docker attach pivot-train-cuda
```

#### AMD ROCm
```bash
# Interactive
docker-compose -f docker/docker-compose.yml run --rm train-rocm bash

# Helper script
bash scripts/docker_train.sh --backend rocm --config configs/train.yaml

# Background
docker-compose -f docker/docker-compose.yml up -d train-rocm
```

#### Intel OneAPI
```bash
# Interactive (OneAPI env auto-loaded)
docker-compose -f docker/docker-compose.yml run --rm train-intel

# Helper script
bash scripts/docker_train.sh --backend intel --config configs/train.yaml

# Background
docker-compose -f docker/docker-compose.yml up -d train-intel
```

### Inference

#### Using Helper Scripts
```bash
# NVIDIA
bash scripts/docker_inference.sh \
  --backend cuda \
  --input ./data/raw/patient001.mhd \
  --model ./checkpoints/best_model.pth \
  --output ./output

# AMD
bash scripts/docker_inference.sh \
  --backend rocm \
  --input ./data/raw/patient001.mhd \
  --model ./checkpoints/best_model.pth \
  --output ./output

# Intel
bash scripts/docker_inference.sh \
  --backend intel \
  --input ./data/raw/patient001.mhd \
  --model ./checkpoints/best_model.pth \
  --output ./output
```

#### Manual Docker Run

**NVIDIA:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/data/raw:/app/input:ro \
  -v $(pwd)/checkpoints:/app/models:ro \
  -v $(pwd)/output:/app/output \
  pivot-inference-cuda:latest \
  python -m inference.main \
  --input /app/input/scan.mhd \
  --model /app/models/model.pth \
  --output /app/output
```

**AMD:**
```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v $(pwd)/data/raw:/app/input:ro \
  -v $(pwd)/checkpoints:/app/models:ro \
  -v $(pwd)/output:/app/output \
  pivot-inference-rocm:latest \
  python -m inference.main \
  --input /app/input/scan.mhd \
  --model /app/models/model.pth \
  --output /app/output
```

**Intel:**
```bash
docker run --rm \
  --device=/dev/dri \
  -v $(pwd)/data/raw:/app/input:ro \
  -v $(pwd)/checkpoints:/app/models:ro \
  -v $(pwd)/output:/app/output \
  pivot-inference-intel:latest \
  bash -c "source /opt/intel/oneapi/setvars.sh --force && \
    python -m inference.main \
    --input /app/input/scan.mhd \
    --model /app/models/model.pth \
    --output /app/output"
```
# Training image
bash scripts/docker_build.sh --train

# Inference image
bash scripts/docker_build.sh --inference
```

### Using Makefile
```bash
make docker-build
```

## Running Containers

### Training

#### Interactive Training Session
```bash
docker-compose -f docker/docker-compose.yml run --rm train bash
```

Inside the container:
```bash
python -m train.main --config configs/train.yaml
```

#### Using Helper Script
```bash
bash scripts/docker_train.sh --config configs/train.yaml
```

#### Background Training
```bash
docker-compose -f docker/docker-compose.yml up -d train
docker attach pivot-train
```

### Inference

#### Using Helper Script
```bash
bash scripts/docker_inference.sh \
  --input ./data/raw/patient001.mhd \
  --model ./checkpoints/best_model.pth \
  --output ./output
```

#### Manual Docker Run
```bash
docker run --rm --gpus all \
  -v $(pwd)/data/raw:/app/input:ro \
  -v $(pwd)/checkpoints:/app/models:ro \
  -v $(pwd)/output:/app/output \
  pivot-inference:latest \
  python -m inference.main \
  --input /app/input/scan.mhd \
  --model /app/models/model.pth \
  --output /app/output
```

### Development with Jupyter

#### Start Jupyter Lab
```bash
bash scripts/docker_dev.sh
```

Or:
```bash
docker-compose -f docker/docker-compose.yml up dev
```

Access Jupyter at `http://localhost:8888`

Default token is displayed in the logs:
```bash
docker-compose logs dev | grep token
```

## Docker Compose Services

### Available Services

1. **train** - Training container with GPU support
2. **inference** - Inference container
3. **dev** - Development container with Jupyter

### Service Configuration

#### Environment Variables
Set in `.env` file or pass to docker-compose:
```bash
WANDB_API_KEY=your_key_here
WANDB_PROJECT=pivot
CUDA_VISIBLE_DEVICES=0
```

#### GPU Configuration
By default, all GPUs are available. To limit GPUs:

```yaml
# In docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use only GPUs 0 and 1
```

Or when running:
```bash
CUDA_VISIBLE_DEVICES=0 docker-compose -f docker/docker-compose.yml up train
```

### Volume Mounts

The following directories are mounted by default:

- `./data` → `/workspace/data` (training) or `/app/input` (inference)
- `./checkpoints` → `/workspace/checkpoints` or `/app/models`
- `./logs` → `/workspace/logs`
- `./configs` → `/workspace/configs` or `/app/configs`

## Common Tasks

### View Running Containers
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f train
```

### Execute Command in Running Container
```bash
docker-compose exec train bash
docker-compose exec train python -m train.main --help
```

### Stop Containers
```bash
# Stop all
docker-compose down

# Stop specific service
docker-compose stop train
```

### Remove All PIVOT Containers and Images
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi pivot-train:latest pivot-inference:latest
```

## TensorBoard

TensorBoard is exposed on port 6006:

```bash
# Inside training container
tensorboard --logdir /workspace/logs --bind_all

# Access at http://localhost:6006
```

Or start with docker-compose:
```bash
docker-compose exec train tensorboard --logdir /workspace/logs --bind_all
```

## Troubleshooting

### GPU Not Available

Check NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails, ensure nvidia-container-toolkit is installed and Docker daemon is restarted.

### Permission Denied Errors

Files created by Docker are owned by root. To fix:
```bash
sudo chown -R $USER:$USER ./data ./checkpoints ./logs
```

Or run containers with your UID:
```bash
docker-compose -f docker/docker-compose.yml run --rm -u $(id -u):$(id -g) train bash
```

### Out of Memory

Increase shared memory size in `docker-compose.yml`:
```yaml
shm_size: '16gb'  # Increase from 8gb
```

### Container Won't Start

Check logs:
```bash
docker-compose logs train
```

Rebuild image:
```bash
docker-compose build --no-cache train
```

## Production Deployment

### Using Inference Image in Production

1. **Build optimized image:**
```bash
docker build -f docker/Dockerfile.inference.cuda -t pivot-inference:v1.0 .
```

2. **Tag for registry:**
```bash
docker tag pivot-inference:v1.0 your-registry.com/pivot-inference:v1.0
```

3. **Push to registry:**
```bash
docker push your-registry.com/pivot-inference:v1.0
```

4. **Deploy:**
```bash
docker run -d --gpus all \
  --name pivot-inference-prod \
  -v /data/models:/app/models:ro \
  -v /data/input:/app/input:ro \
  -v /data/output:/app/output \
  -p 8000:8000 \
  your-registry.com/pivot-inference:v1.0
```

### Health Checks

The inference image includes a health check:
```bash
docker inspect --format='{{.State.Health.Status}}' pivot-inference-prod
```

## Best Practices

1. **Use .dockerignore** - Already configured to exclude unnecessary files
2. **Mount volumes** - Don't copy large datasets into images
3. **Tag images** - Use version tags for production images
4. **Multi-stage builds** - Consider for even smaller images
5. **Security** - Run as non-root user in production
6. **Logging** - Use Docker logging drivers for production
7. **Resource limits** - Set memory and CPU limits in production

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Reference](https://docs.docker.com/compose/)
