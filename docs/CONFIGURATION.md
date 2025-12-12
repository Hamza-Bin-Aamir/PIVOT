# Configuration System

PIVOT uses a comprehensive configuration management system that supports:

- **Type-safe configurations** with dataclasses and validation
- **Multiple configuration templates** for different use cases
- **Hardware-agnostic settings** (NVIDIA, AMD, Intel)
- **Easy config merging and overrides**
- **CLI tools** for config management

## Quick Start

### Using Existing Templates

```bash
# Fast development (2 epochs, minimal settings)
python -m train.main --config configs/fast_dev.yaml

# High performance (production training)
python -m train.main --config configs/high_performance.yaml

# Hardware-specific configs
python -m train.main --config configs/amd_rocm.yaml
python -m train.main --config configs/intel_xpu.yaml
```

### Creating Custom Configurations

```bash
# Create from template
python scripts/manage_config.py create --template high_performance --output configs/my_config.yaml

# Validate your config
python scripts/manage_config.py validate configs/my_config.yaml

# Show detailed config info
python scripts/manage_config.py show configs/my_config.yaml

# List all available templates
python scripts/manage_config.py list
```

## Configuration Structure

### Main Config File

```yaml
experiment_name: my_experiment
output_dir: outputs/my_experiment

train:
  epochs: 100
  batch_size: 2

  model:
    type: unet3d
    in_channels: 1
    out_channels: 1
    init_features: 32
    depth: 4

  optimizer:
    type: adam
    lr: 0.0001
    weight_decay: 0.00001

  hardware:
    device: cuda
    mixed_precision: true
    seed: 42

inference:
  batch_size: 1
  overlap: 0.5
  threshold: 0.5
```

### Component Configurations

#### Model Configuration
```yaml
model:
  type: unet3d           # Model architecture
  in_channels: 1         # Input channels (1 for CT)
  out_channels: 1        # Output channels (1 for binary seg)
  init_features: 32      # Initial feature maps
  depth: 4               # U-Net depth
  dropout: 0.0           # Dropout rate
  batch_norm: true       # Use batch normalization
```

#### Data Configuration
```yaml
data_dir: data/processed
batch_size: 2
num_workers: 4
pin_memory: true
shuffle_train: true
cache_rate: 0.0          # MONAI cache (0=none, 1=full)
```

#### Preprocessing Configuration
```yaml
preprocessing:
  target_spacing: [1.0, 1.0, 1.0]  # Resample to 1mm isotropic
  window_center: -600               # HU window center
  window_width: 1500                # HU window width
  patch_size: [96, 96, 96]          # 3D patch size
  normalize: true                   # Normalize intensities
```

#### Augmentation Configuration
```yaml
augmentation:
  enabled: true
  random_flip_prob: 0.5
  random_rotate_prob: 0.5
  random_scale_prob: 0.5
  random_intensity_shift_prob: 0.5
  random_intensity_scale_prob: 0.5
  elastic_deform_prob: 0.0
  gaussian_noise_prob: 0.0
```

#### Optimizer Configuration
```yaml
optimizer:
  type: adam              # adam, adamw, sgd
  lr: 0.0001             # Learning rate
  weight_decay: 0.00001  # L2 regularization
  betas: [0.9, 0.999]    # Adam betas
```

#### Scheduler Configuration
```yaml
scheduler:
  type: cosine           # cosine, step, multistep
  min_lr: 0.000001      # Minimum learning rate
  warmup_epochs: 0      # Warmup epochs
  warmup_start_lr: 0.00001
```

#### Loss Configuration
```yaml
loss:
  type: dice             # dice, focal, dice_focal, bce
  smooth: 0.00001       # Smoothing factor
  focal_alpha: 0.25     # Focal loss alpha
  focal_gamma: 2.0      # Focal loss gamma
  reduction: mean       # mean, sum, none
```

#### Checkpoint Configuration
```yaml
checkpoint:
  checkpoint_dir: checkpoints
  save_every: 10              # Save every N epochs
  save_best: true             # Save best model
  metric_to_track: val_dice   # Metric to track
  mode: max                   # max or min
  save_last: true             # Save last checkpoint
  max_checkpoints: 5          # Keep only N best
```

#### Logging Configuration
```yaml
logging:
  log_dir: logs
  log_every: 10                  # Log every N batches
  tensorboard: true              # Enable TensorBoard
  console_log_level: INFO        # Console verbosity
  file_log_level: DEBUG          # File verbosity
```

#### W&B Configuration
```yaml
wandb:
  enabled: false
  project: pivot
  entity: your-username
  tags: [experiment, v1]
  notes: "Experiment description"
  resume: auto              # auto, allow, never, must
```

#### Hardware Configuration
```yaml
hardware:
  device: cuda               # cuda, rocm, xpu, cpu
  gpu_ids: [0]              # GPU IDs to use
  mixed_precision: true     # Enable AMP
  cudnn_benchmark: true     # cudnn auto-tuner
  deterministic: false      # Deterministic mode
  seed: 42                  # Random seed
```

## Available Templates

### 1. `train.yaml` - Default Training
Standard configuration for general use.

### 2. `fast_dev.yaml` - Fast Development
- 2 epochs for quick testing
- Small batch size (1)
- No multiprocessing (num_workers=0)
- Augmentation disabled
- Deterministic mode for debugging

**Use for:** Quick testing, debugging, development

### 3. `high_performance.yaml` - Production Training
- 300 epochs
- Large batch size (4) with gradient accumulation
- Full augmentation pipeline
- Data caching enabled
- Early stopping
- W&B logging enabled

**Use for:** Final model training, production runs

### 4. `amd_rocm.yaml` - AMD GPU Optimized
- ROCm-compatible settings
- Conservative batch size for stability
- Mixed precision enabled

**Use for:** AMD Radeon/Instinct GPUs

### 5. `intel_xpu.yaml` - Intel GPU Optimized
- Intel XPU device configuration
- Small batch size for integrated GPUs
- Gradient accumulation to compensate
- Intel-specific optimizations

**Use for:** Intel Arc, Iris Xe, UHD Graphics

### 6. `inference.yaml` - Inference Only
- Optimized for deployment
- High overlap for quality
- Test-time augmentation
- Probability map saving

**Use for:** Model deployment, inference pipelines

## Programmatic Usage

### Load Configuration

```python
from src.config import Config

# Load from YAML
config = Config.from_yaml("configs/train.yaml")

# Access settings
print(config.train.epochs)
print(config.train.model.type)
print(config.train.hardware.device)
```

### Create Configuration

```python
from src.config import Config, ModelConfig, TrainConfig

# Create programmatically
config = Config(
    experiment_name="my_experiment",
    train=TrainConfig(
        epochs=100,
        model=ModelConfig(
            type="unet3d",
            depth=5
        )
    )
)

# Save to file
config.save("configs/my_config.yaml")
```

### Merge Configurations

```python
# Load base config
config = Config.from_yaml("configs/train.yaml")

# Override with custom settings
overrides = {
    "train": {
        "epochs": 200,
        "batch_size": 4
    }
}
merged = config.merge_from_dict(overrides)

# Or merge from file
merged = config.merge_from_file("configs/overrides.yaml")
```

### Use Default Templates

```python
from src.config import get_fast_dev_config, get_high_performance_config

# Get default configs as dictionaries
fast_config = get_fast_dev_config()
hp_config = get_high_performance_config()
```

## CLI Tools

### Validate Configuration

```bash
python scripts/manage_config.py validate configs/train.yaml
```

Output:
```
✓ Configuration is valid: configs/train.yaml
  Experiment: default_experiment
  Epochs: 100
  Batch size: 2
  Model: unet3d
  Device: cuda
```

### Create New Configuration

```bash
python scripts/manage_config.py create \
  --template high_performance \
  --output configs/my_hp_config.yaml
```

### Merge Configurations

```bash
python scripts/manage_config.py merge \
  --base configs/train.yaml \
  --overrides configs/overrides.yaml configs/gpu_settings.yaml \
  --output configs/merged.yaml
```

### Show Configuration

```bash
python scripts/manage_config.py show configs/train.yaml
```

Output:
```
============================================================
Configuration: configs/train.yaml
============================================================

Experiment: default_experiment
Output Directory: outputs

------------------------------------------------------------
Training Configuration
------------------------------------------------------------
  Epochs: 100
  Batch Size: 2
  Learning Rate: 0.0001
  Device: cuda
  Mixed Precision: True

------------------------------------------------------------
Model Configuration
------------------------------------------------------------
  Type: unet3d
  Input Channels: 1
  Output Channels: 1
  Features: 32
  Depth: 4
...
```

### List Templates

```bash
python scripts/manage_config.py list
```

## Configuration Validation

All configurations are validated on load:

- **Type checking**: Ensures correct data types
- **Range validation**: Values within acceptable ranges
- **Dependency checking**: Related settings are compatible
- **Hardware compatibility**: Device-specific settings are valid

Example validation errors:

```python
# Invalid learning rate
ModelConfig(lr=-0.001)  # ValueError: lr must be > 0

# Invalid batch size
DataConfig(batch_size=0)  # ValueError: batch_size must be >= 1

# Invalid device
HardwareConfig(device="invalid")  # ValueError: device must be one of [cuda, rocm, xpu, cpu]
```

## Best Practices

### Development Workflow

1. **Start with `fast_dev.yaml`** for initial testing
2. **Validate early** with small dataset
3. **Graduate to `train.yaml`** for full run
4. **Use `high_performance.yaml`** for final training

### Configuration Management

1. **Version control configs** alongside code
2. **Use descriptive experiment names**
3. **Document custom settings** in config comments
4. **Save configs with checkpoints** for reproducibility

### Hardware-Specific Settings

```yaml
# NVIDIA GPUs
hardware:
  device: cuda
  mixed_precision: true
  cudnn_benchmark: true

# AMD GPUs
hardware:
  device: cuda  # ROCm uses CUDA backend
  mixed_precision: true

# Intel GPUs
hardware:
  device: xpu
  mixed_precision: false  # Use Intel-specific optimizations
```

## Troubleshooting

### Config Not Loading

```bash
# Validate the config file
python scripts/manage_config.py validate configs/my_config.yaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"
```

### Missing Fields

The config system provides defaults for all fields. If you get errors about missing fields, ensure you're using the latest config structure.

### Backward Compatibility

The system maintains backward compatibility:
- `learning_rate` → automatically mapped to `optimizer.lr`
- Top-level `preprocessing` → moved to `train.preprocessing`

## Advanced Usage

### Custom Config Classes

```python
from dataclasses import dataclass
from src.config.base import ModelConfig

@dataclass
class CustomModelConfig(ModelConfig):
    """Extended model config."""
    custom_param: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.custom_param < 0:
            raise ValueError("custom_param must be >= 0")
```

### Environment Variable Overrides

```bash
# Override config values via environment
export PIVOT_BATCH_SIZE=4
export PIVOT_LEARNING_RATE=0.001

python -m train.main --config configs/train.yaml
```

### Runtime Config Updates

```python
# Load and modify at runtime
config = Config.from_yaml("configs/train.yaml")

# Adjust based on available GPU memory
import torch
if torch.cuda.get_device_properties(0).total_memory < 8e9:
    config.train.data.batch_size = 1
    config.train.preprocessing.patch_size = (64, 64, 64)
```
