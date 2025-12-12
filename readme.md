# Pulmonary Imaging for Volume Oncology Triage

This repository contains resources and tools for the Pulmonary Imaging for Volume Oncology Triage (PIVOT) project. The project uses Pulmonary Imaging to assist in the triage of oncology patients.

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hamza-Bin-Aamir/PIVOT.git
   cd PIVOT
   ```

2. Set up the development environment (recommended):
   ```bash
   bash scripts/setup_env.sh
   ```

   Or manually install dependencies:
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -e .

   # Install pre-commit hooks
   pre-commit install --hook-type commit-msg
   pre-commit install
   ```

3. Copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Using Make Commands

```bash
make install      # Install dependencies
make test         # Run tests
make format       # Format code with black and isort
make lint         # Run linting checks
make clean        # Clean build artifacts
```

## Dataset

This project employs an open-source data strategy to ensure both generalization against academic benchmarks and reproducibility. The model is pre-trained on the public LUNA16 dataset (a subset of LIDC-IDRI).

### Public Benchmark: LUNA16 / LIDC-IDRI

The primary training and validation are conducted using the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is a curated subset of the larger **LIDC-IDRI** database.

- **Source:** [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/)
- **Composition:** 888 CT scans with slice thickness ≤ 2.5mm.
- **Annotations:** 1,186 nodules marked as "positive" (accepted by at least 3 out of 4 radiologists).
- **Inclusion Criteria:** Nodules with a diameter ≥ 3mm. Non-nodules, nodules < 3mm, and nodules with ambiguous consensus are excluded from the positive class during training but may be used for hard negative mining.

## Preprocessing Pipeline

Raw CT data (DICOM or MHD) undergoes a rigorous 3D preprocessing pipeline before entering the network. The pipeline is implemented in `src/data/preprocess.py`.

1. **Resampling:** All scans are resampled to an isotropic resolution of `1mm x 1mm x 1mm` to handle varying slice thicknesses across scanners.
2. **HU Windowing:** Pixel intensities are clipped to the standard lung window:
   - **Level:** -600 HU
   - **Width:** 1500 HU (Range: -1350 to +150)
3. **Normalization:** Windowed values are linearly scaled to the range `[0, 1]`.
4. **Patch Generation:** The volume is cropped into 3D patches of size `(96, 96, 96)` for training.

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
├── docs/                  # Documentation
├── scripts/               # Utility scripts
│   ├── setup_env.sh      # Environment setup
│   └── download_luna16.sh # Dataset download helper
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
├── .githooks/            # Custom git hooks
│   └── check_imports.py  # Import validation hook
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── setup.py              # Package installation
├── pyproject.toml        # Tool configuration
├── Makefile              # Build automation
└── .pre-commit-config.yaml # Pre-commit hooks

```

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
