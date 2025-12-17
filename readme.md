# Pulmonary Imaging for Volume Oncology Triage

This repository contains resources and tools for the Pulmonary Imaging for Volume Oncology Triage (PIVOT) project. The project uses Pulmonary Imaging to assist in the triage of oncology patients.

## Quick Start

### Installation

1. **Install uv** (fast Python package manager):
   ```bash
   pip install uv
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Hamza-Bin-Aamir/PIVOT.git
   cd PIVOT
   ```

3. Install dependencies with uv:
   ```bash
   # For CPU-only (local development without GPU)
   uv sync

   # Or if you have NVIDIA GPU
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   uv sync

   # Or if you have AMD GPU
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
   uv sync
   ```

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install --hook-type commit-msg
   uv run pre-commit install
   ```

5. Copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Using Make Commands

```bash
make install      # Install dependencies with uv
make test         # Run tests
make format       # Format code with ruff
make lint         # Run linting checks
make clean        # Clean build artifacts
```

### Docker Setup (Recommended for GPU Training)

PIVOT supports multiple GPU backends through Docker:
- **NVIDIA GPUs** (CUDA) - Consumer and datacenter GPUs
- **AMD GPUs** (ROCm) - Integrated and dedicated GPUs
- **Intel GPUs** (OneAPI) - Integrated and dedicated GPUs (Arc series)

#### Detect Your GPU

```bash
# Auto-detect available GPU backend
bash scripts/detect_gpu.sh
```

> **Note**: GPU detection inspects Linux device nodes. On Windows, run this
> command inside Git Bash (with WSL) or on a Linux host.

#### Build Docker Images

```bash
# Detect GPU and build appropriate images
bash scripts/detect_gpu.sh  # See recommended backend

# Build for specific GPU backend
bash scripts/docker_build.sh --all --backend cuda   # NVIDIA
bash scripts/docker_build.sh --all --backend rocm   # AMD
bash scripts/docker_build.sh --all --backend intel  # Intel

# Build specific image types
bash scripts/docker_build.sh --train --backend cuda      # Training only
bash scripts/docker_build.sh --inference --backend rocm  # Inference only
```

#### Run Training in Docker

```bash
# Using auto-detected or specified backend
bash scripts/docker_train.sh --backend cuda --config configs/train.yaml

# Start training container in background
docker-compose up -d train-cuda    # NVIDIA
docker-compose up -d train-rocm    # AMD
docker-compose up -d train-intel   # Intel
```

#### Run Inference in Docker

```bash
# With specified GPU backend
bash scripts/docker_inference.sh \
  --backend cuda \
  --input ./data/raw/scan.mhd \
  --model ./checkpoints/model.pth \
  --output ./output

# AMD GPU
bash scripts/docker_inference.sh --backend rocm --input <input> --model <model>

# Intel GPU
bash scripts/docker_inference.sh --backend intel --input <input> --model <model>
```

#### Development with Jupyter

```bash
# Start Jupyter Lab (NVIDIA by default)
bash scripts/docker_dev.sh

# With specific backend and custom port
bash scripts/docker_dev.sh 8888 rocm
bash scripts/docker_dev.sh 9000 intel

# Windows PowerShell users: prefix with bash
bash scripts/docker_dev.sh 8888 cuda
```

Access Jupyter at `http://localhost:8888`

#### Docker Compose Services

```bash
# Start specific service
docker-compose -f docker/docker-compose.yml up -d train-cuda       # Training with NVIDIA GPU
docker-compose -f docker/docker-compose.yml up -d train-rocm       # Training with AMD GPU
docker-compose -f docker/docker-compose.yml up -d train-intel      # Training with Intel GPU
docker-compose -f docker/docker-compose.yml up -d inference-cuda   # Inference with NVIDIA GPU

# View logs
docker-compose -f docker/docker-compose.yml logs -f train-cuda

# Stop services
docker-compose -f docker/docker-compose.yml down
```

#### GPU Backend Requirements

**NVIDIA (CUDA)**
- NVIDIA GPU with CUDA 12.1+ support
- nvidia-docker2 runtime
- Install: `sudo apt-get install nvidia-docker2`

**AMD (ROCm)**
- AMD GPU (Vega, RDNA, or newer)
- ROCm 5.7+ installed
- Install: [ROCm Installation Guide](https://rocm.docs.amd.com/)

**Intel (OneAPI)**
- Intel integrated GPU (11th gen or newer) or Arc GPU
- Intel GPU drivers
- Install: [Intel GPU Drivers](https://dgpu-docs.intel.com/)

## Configuration Management

PIVOT uses a comprehensive type-safe configuration system with built-in validation and hardware-agnostic settings.

### Quick Start

```bash
# List available configuration templates
python scripts/manage_config.py list

# Use a pre-configured template
python -m train.main --config configs/fast_dev.yaml        # Fast development
python -m train.main --config configs/high_performance.yaml # Production
python -m train.main --config configs/amd_rocm.yaml        # AMD GPUs
python -m train.main --config configs/intel_xpu.yaml       # Intel GPUs

# Validate your configuration
python scripts/manage_config.py validate configs/train.yaml

# Show detailed configuration
python scripts/manage_config.py show configs/train.yaml

# Create custom configuration from template
python scripts/manage_config.py create \
  --template high_performance \
  --output configs/my_config.yaml
```

### Available Templates

- **`train.yaml`** - Default training configuration
- **`fast_dev.yaml`** - Quick testing (2 epochs, small batch, no augmentation)
- **`high_performance.yaml`** - Production training (300 epochs, large batch, full augmentation)
- **`amd_rocm.yaml`** - Optimized for AMD GPUs
- **`intel_xpu.yaml`** - Optimized for Intel GPUs
- **`inference.yaml`** - Inference-only configuration

### Configuration Structure

```yaml
experiment_name: my_experiment
output_dir: outputs

train:
  epochs: 100
  batch_size: 2
  precision: "32"  # Options: "32" (FP32), "16-mixed" (FP16), "bf16-mixed" (BF16)

  model:
    type: unet3d
    depth: 4
    init_features: 32

  optimizer:
    type: adam
    lr: 0.0001

  hardware:
    device: cuda  # cuda, rocm, xpu, cpu
    seed: 42

inference:
  batch_size: 1
  overlap: 0.5
  threshold: 0.5
```

#### Mixed Precision Training

PIVOT supports mixed precision training for faster training and reduced memory usage:

- **`"32"`** (default): Full FP32 precision - most stable, slowest
- **`"16-mixed"`**: Mixed precision FP16 - 2-3x faster on NVIDIA GPUs
- **`"bf16-mixed"`**: Mixed precision BF16 - recommended for AMD/Intel GPUs, more stable than FP16

Example configuration:
```yaml
train:
  precision: "16-mixed"  # For NVIDIA GPUs
```

Or in code:
```python
from src.train import LitNoduleDetection
import lightning as L

model = LitNoduleDetection(precision="16-mixed")
trainer = L.Trainer(max_epochs=100, precision="16-mixed")
trainer.fit(model, train_dataloader, val_dataloader)
```

See **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** for complete documentation.

## Dataset

This project employs an open-source data strategy to ensure both generalization against academic benchmarks and reproducibility. The model is pre-trained on the public LUNA16 dataset (a subset of LIDC-IDRI).

### Public Benchmark: LUNA16 / LIDC-IDRI

The primary training and validation are conducted using the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is a curated subset of the larger **LIDC-IDRI** database.

- **Source:** [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/)
- **Composition:** 888 CT scans with slice thickness ≤ 2.5mm.
- **Annotations:** 1,186 nodules marked as "positive" (accepted by at least 3 out of 4 radiologists).
- **Inclusion Criteria:** Nodules with a diameter ≥ 3mm. Non-nodules, nodules < 3mm, and nodules with ambiguous consensus are excluded from the positive class during training but may be used for hard negative mining.

### Annotation Metadata Utilities

Use the helpers in `src/data/annotations.py` to transform both the LUNA16 CSV
and the richer LIDC-IDRI XML reports into immutable dataclasses ready for data
prep and ground-truth generation. See [docs/ANNOTATIONS.md](docs/ANNOTATIONS.md)
for usage examples, filtering options, and integration tips.

## Preprocessing Pipeline

Raw CT data (DICOM or MHD) undergoes a rigorous 3D preprocessing pipeline before entering the network. The pipeline is implemented in `src/data/preprocess.py`.

For an end-to-end roadmap that mirrors the active GitHub issues—from raw data intake through training, inference, and serving—see **[docs/PIPELINE.md](docs/PIPELINE.md)**.

1. **Resampling:** All scans are resampled to an isotropic resolution of `1mm x 1mm x 1mm` to handle varying slice thicknesses across scanners.
2. **HU Windowing:** Pixel intensities are clipped to the standard lung window:
   - **Level:** -600 HU
   - **Width:** 1500 HU (Range: -1350 to +150)
3. **Intensity Normalization:** `normalize_intensity` clips to the configured HU window, scales values into `[0, 1]`, and can optionally apply global or adaptive histogram equalisation. The function returns descriptive statistics so downstream steps can log or monitor preprocessing drift.
4. **Patch Generation:** The volume is cropped into 3D patches of size `(96, 96, 96)` for training.

## Data Augmentation

Training-time augmentation is implemented in `src/data/augment.py`. The default pipeline provided by `build_default_augmentation_pipeline` composes:

- Random flips across each spatial axis.
- 90° rotations with axis reordering so tensors remain in `(C, D, H, W)` format.
- Optional Gaussian noise injection controlled via probability and standard deviation.
- Random intensity scaling and shifting to simulate scanner-specific contrast changes.

Configure the probabilities and ranges through `AugmentationConfig`; disable any step by setting its probability to `0.0`. The `LUNADataset` accepts a transform callable, so custom augmentation stacks can be swapped in as needed.

## Directory Structure

The project follows this structure:

```
PIVOT/
├── configs/                # Configuration files
│   └── train.yaml         # Training configuration
├── data/                  # Data directory (gitignored)
│   ├── raw/               # Raw datasets
│   │   └── luna16/       # LUNA16 dataset
│   └── processed/        # Preprocessed data
│       ├── train/        # Training data
│       └── val/          # Validation data
├── docker/                # Docker configuration
│   ├── Dockerfile.train.cuda       # NVIDIA training image
│   ├── Dockerfile.train.rocm       # AMD training image
│   ├── Dockerfile.train.intel      # Intel training image
│   ├── Dockerfile.inference.cuda   # NVIDIA inference image
│   ├── Dockerfile.inference.rocm   # AMD inference image
│   ├── Dockerfile.inference.intel  # Intel inference image
│   ├── docker-compose.yml          # Docker orchestration
│   ├── .dockerignore               # Docker ignore patterns
│   └── requirements-gpu.txt        # GPU backend info
├── docs/                  # Documentation
│   ├── DOCKER.md         # Docker guide
│   └── CONTRIBUTING.md   # Contribution guidelines
├── scripts/               # Utility scripts
│   ├── setup_env.sh      # Environment setup
│   ├── download_luna16.sh # Dataset download helper
│   ├── docker_build.sh   # Build Docker images
│   ├── docker_train.sh   # Run training in Docker
│   ├── docker_inference.sh # Run inference in Docker
│   ├── docker_dev.sh     # Development environment
│   └── detect_gpu.sh     # GPU backend detection
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   │   ├── dataset.py    # Dataset classes
│   │   └── preprocess.py # Preprocessing pipeline
│   ├── model/            # Model architectures
│   │   └── unet.py       # 3D U-Net implementation
│   ├── train/            # Training pipeline
│   │   └── main.py       # Training script
│   ├── inference/        # Inference pipeline
│   │   └── main.py       # Inference script
│   ├── utils/            # Utility functions
│   │   └── logger.py     # Logging utilities
│   └── config/           # Configuration management
├── tests/                 # Test suite
├── .editorconfig         # Editor configuration
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks
├── .python-version       # Python version for uv
├── pyproject.toml        # Project metadata and dependencies (uv)
├── uv.lock               # Locked dependencies (generated by uv)
├── Makefile              # Build automation
├── MANIFEST.in           # Package distribution rules
├── setup.py              # Legacy setup file (for compatibility)
├── requirements.txt      # Legacy requirements (Docker only)
└── requirements-dev.txt  # Legacy dev requirements (Docker only)
```

**Note**: This project uses `uv` for package management. The `requirements*.txt` files are maintained for Docker builds only. For local development, use `uv sync`.

## Data Directory Structure

To run the training scripts, organize your data as follows:

```
data/
├── raw/
│   ├── luna16/             # Extracted .mhd/.raw files from LUNA16
│   └── private_dicom/      # Your local DICOM folders
├── processed/
│   ├── train/              # Pre-computed numpy arrays (.npy) for training
│   └── val/                # Pre-computed numpy arrays (.npy) for validation
└── splits/
    ├── train_ids.json
    └── val_ids.json
```

## Download Instructions

To replicate the benchmark results:

1. Download the LUNA16 dataset (approx. 60GB compressed) from the [official source](https://zenodo.org/record/3723295).
2. Extract all subsets (subset0 - subset9) into `data/raw/luna16`.
3. Run the conversion script to generate the MONAI-compatible dataset:

   ```bash
   python src/data/prepare_luna.py --input_dir data/raw/luna16 --output_dir data/processed
   ```
