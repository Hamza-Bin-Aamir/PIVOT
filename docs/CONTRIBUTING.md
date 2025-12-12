# Contributing to PIVOT

## Commit Message Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Scopes

Common scopes for this project:
- **data**: Data preprocessing, loading, augmentation
- **model**: Model architecture, layers, heads
- **train**: Training pipeline, loops, callbacks
- **loss**: Loss functions
- **inference**: Inference pipeline, post-processing
- **api**: FastAPI server, endpoints
- **viz**: Visualization, graphs, dashboards
- **eval**: Evaluation metrics, validation
- **deploy**: Deployment, Docker, infrastructure
- **monitor**: Monitoring, logging, alerts

### Examples

```
feat[data]: add LUNA16 DICOM loader
fix[train]: resolve NaN loss issue in multi-task learning
docs[api]: update API endpoint documentation
refactor[model]: simplify U-Net decoder architecture
perf[inference]: optimize sliding window overlap computation
test[loss]: add unit tests for focal loss
chore[deps]: update PyTorch to 2.1.0
```

### Breaking Changes

Breaking changes should be indicated by a `!` after the type/scope and explained in the footer:

```
feat[api]!: change response format for /metrics endpoint

BREAKING CHANGE: The /metrics endpoint now returns data in a different JSON structure.
Clients need to update their parsing logic.
```

## Setting Up Pre-commit

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks:
   ```bash
   pre-commit install --hook-type commit-msg
   pre-commit install  # for other hooks
   ```

3. (Optional) Run against all files:
   ```bash
   pre-commit run --all-files
   ```

## Commit Message Validation

The pre-commit hook will automatically validate your commit messages. If your commit message doesn't follow the conventional format, the commit will be rejected with an error message.

### Valid Examples
✅ `feat[data]: implement isotropic resampling`
✅ `fix: resolve GPU memory overflow`
✅ `docs: update README with setup instructions`
✅ `refactor[model]!: redesign detection head architecture`

### Invalid Examples
❌ `added new feature` (missing type)
❌ `feat add feature` (missing colon)
❌ `FIX[data]: bug fix` (type must be lowercase)
❌ `feat[data] : fix` (space before colon)

## Bypassing the Hook (Not Recommended)

In exceptional cases, you can bypass the hook with:
```bash
git commit --no-verify -m "your message"
```

However, this should be avoided as it breaks the commit convention.

## Hardware-Agnostic Design

PIVOT is designed to run on **any** GPU hardware, not just NVIDIA. We support three GPU backends:

### Supported Hardware

#### 1. NVIDIA GPUs (CUDA)
- **Technology**: NVIDIA CUDA 12.1
- **Compatible GPUs**: Any NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- **Use Cases**: GTX/RTX series, Tesla, A100, H100, etc.
- **PyTorch Backend**: `torch+cu121` (CUDA 12.1)

#### 2. AMD GPUs (ROCm)
- **Technology**: AMD ROCm 5.7
- **Compatible GPUs**:
  - AMD Radeon RX 6000/7000 series (dedicated)
  - AMD Radeon RX Vega series (dedicated)
  - AMD Ryzen APUs with Radeon Graphics (integrated)
- **Use Cases**: Desktop workstations, gaming PCs, AMD-based laptops
- **PyTorch Backend**: `torch+rocm5.7`

#### 3. Intel GPUs (OneAPI)
- **Technology**: Intel OneAPI 2024.0.0
- **Compatible GPUs**:
  - Intel Arc series (dedicated)
  - Intel Iris Xe Graphics (integrated)
  - Intel UHD Graphics (integrated)
- **Use Cases**: Modern Intel laptops, Intel Arc desktop cards
- **PyTorch Backend**: `torch+xpu` via Intel Extension for PyTorch (IPEX)

### Automatic GPU Detection

The project includes an automatic GPU detection script that identifies your hardware:

```bash
# Detect your GPU backend
bash scripts/detect_gpu.sh

# Output examples:
# "cuda"  - NVIDIA GPU detected
# "rocm"  - AMD GPU detected
# "intel" - Intel GPU detected
# "cpu"   - No compatible GPU found, will use CPU
```

### Building for Your Hardware

#### Option 1: Automatic Detection (Recommended)

```bash
# Auto-detect and build for your GPU
make docker-detect-gpu
make docker-build

# Or use the helper script
bash scripts/docker_build.sh
```

#### Option 2: Manual Backend Selection

```bash
# For NVIDIA GPUs
make docker-build-cuda

# For AMD GPUs
make docker-build-rocm

# For Intel GPUs
make docker-build-intel

# Or specify backend explicitly
GPU_BACKEND=cuda make docker-build
GPU_BACKEND=rocm make docker-build
GPU_BACKEND=intel make docker-build
```

### Running Training on Different Hardware

The training command is **identical** across all hardware:

```bash
# Auto-detect backend
bash scripts/docker_train.sh

# Or specify backend
bash scripts/docker_train.sh cuda   # NVIDIA
bash scripts/docker_train.sh rocm   # AMD
bash scripts/docker_train.sh intel  # Intel

# Using docker-compose directly
docker-compose -f docker/docker-compose.yml run --rm train-cuda
docker-compose -f docker/docker-compose.yml run --rm train-rocm
docker-compose -f docker/docker-compose.yml run --rm train-intel
```

### Running Inference on Different Hardware

```bash
# Auto-detect backend
bash scripts/docker_inference.sh path/to/scan.nii.gz

# Or specify backend
bash scripts/docker_inference.sh path/to/scan.nii.gz cuda
bash scripts/docker_inference.sh path/to/scan.nii.gz rocm
bash scripts/docker_inference.sh path/to/scan.nii.gz intel
```

### Development Guidelines for Hardware Compatibility

When contributing code, ensure it remains hardware-agnostic:

#### ✅ DO:
```python
# Use device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Use PyTorch's automatic device selection
loss.backward()
optimizer.step()

# Use standard PyTorch operations (work on all backends)
output = torch.softmax(logits, dim=1)
```

#### ❌ DON'T:
```python
# Don't hardcode CUDA-specific code
torch.cuda.synchronize()  # Use only if absolutely necessary
torch.cuda.set_device(0)  # Let PyTorch handle device management

# Don't use NVIDIA-specific features
from apex import amp  # Use torch.amp instead

# Don't assume CUDA is always available
model.cuda()  # Use .to(device) instead
```

#### Mixed Precision Training (All Backends)

```python
# Use PyTorch's native AMP - works on CUDA, ROCm, and XPU
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Testing on Multiple Backends

If you have access to multiple GPU types, test your changes on all of them:

```bash
# Test on NVIDIA
GPU_BACKEND=cuda make docker-build
docker-compose -f docker/docker-compose.yml run --rm train-cuda

# Test on AMD
GPU_BACKEND=rocm make docker-build
docker-compose -f docker/docker-compose.yml run --rm train-rocm

# Test on Intel
GPU_BACKEND=intel make docker-build
docker-compose -f docker/docker-compose.yml run --rm train-intel
```

### Performance Expectations

- **NVIDIA (CUDA)**: Best performance, most mature ecosystem
- **AMD (ROCm)**: 80-95% of CUDA performance on compatible hardware
- **Intel (XPU)**: 60-80% of CUDA performance, improving rapidly

### Troubleshooting Hardware Issues

#### NVIDIA: No GPU Detected
```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### AMD: ROCm Not Working
```bash
# Check ROCm installation
rocm-smi

# Verify GPU is supported
rocminfo | grep "Name:"
```

#### Intel: XPU Not Available
```bash
# Check Intel GPU
sudo lspci | grep -i vga

# Verify OneAPI installation
sycl-ls
```

### Architecture Documentation

All Dockerfiles use the same structure:
- Base image with GPU support (cuda/rocm/oneapi)
- Python 3.10+
- PyTorch with appropriate backend
- MONAI and medical imaging libraries
- Application code from `src/`

See `docker/requirements-gpu.txt` for detailed installation instructions per backend.
