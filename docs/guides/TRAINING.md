# Training Guide

This guide covers how to train the PIVOT lung nodule detection model, from basic usage to advanced configuration and optimization.

## Table of Contents

- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Optimization](#optimization)
- [Monitoring Training](#monitoring-training)
- [Multi-GPU Training](#multi-gpu-training)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Training

Train with default configuration:

```bash
# Train with default settings
uv run python -m src.train.main --config configs/train.yaml

# Train with custom config
uv run python -m src.train.main --config my_config.yaml
```

### Minimal Configuration

Create `configs/my_train.yaml`:

```yaml
data:
  train_dir: data/processed/train
  val_dir: data/processed/val
  annotations: data/annotations.csv

model:
  type: unet
  in_channels: 1
  out_channels: 1
  base_filters: 32

training:
  epochs: 100
  batch_size: 2
  learning_rate: 0.001
```

Run training:

```bash
uv run python -m src.train.main --config configs/my_train.yaml
```

## Training Configuration

### Complete Configuration Example

```yaml
# Data Configuration
data:
  train_dir: data/processed/train
  val_dir: data/processed/val
  test_dir: data/processed/test
  annotations: data/annotations.csv

  # Patch extraction
  patch_size: [128, 128, 128]
  stride: [64, 64, 64]

  # Data loading
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

# Model Architecture
model:
  type: unet
  in_channels: 1
  out_channels: 1
  base_filters: 32
  depth: 4
  use_batch_norm: true
  use_dropout: true
  dropout_rate: 0.1
  activation: relu

# Training Configuration
training:
  # Basic settings
  epochs: 100
  batch_size: 2
  accumulation_steps: 4  # Effective batch size = 2 * 4 = 8

  # Learning rate
  learning_rate: 0.001
  lr_scheduler:
    type: cosine
    warmup_epochs: 5
    min_lr: 0.00001

  # Optimizer
  optimizer:
    type: adamw
    weight_decay: 0.0001
    betas: [0.9, 0.999]

  # Loss function
  loss:
    type: multi_task
    detection_weight: 1.0
    regression_weight: 0.5
    triage_weight: 0.3

    detection_loss:
      type: focal
      alpha: 0.25
      gamma: 2.0

    regression_loss:
      type: smooth_l1
      beta: 1.0

    triage_loss:
      type: bce
      pos_weight: 3.0

  # Hard negative mining
  hard_negative_mining:
    enabled: true
    neg_pos_ratio: 3
    batch_hard_ratio: 0.5

# Augmentation
augmentation:
  enabled: true
  random_flip:
    probability: 0.5
    axes: [0, 1, 2]
  random_rotation:
    probability: 0.5
    max_angle: 15
  random_scale:
    probability: 0.5
    scale_range: [0.9, 1.1]
  random_noise:
    probability: 0.3
    noise_std: 0.01
  elastic_deformation:
    probability: 0.2
    alpha: 10
    sigma: 3

# Validation
validation:
  interval: 1  # Validate every epoch
  metrics:
    - detection_dice
    - froc_score
    - center_accuracy
    - size_accuracy
  save_best_metric: froc_score

# Checkpointing
checkpointing:
  save_dir: checkpoints
  save_interval: 5  # Save every 5 epochs
  keep_last_n: 3
  save_best: true

# Logging
logging:
  log_dir: logs
  log_interval: 10  # Log every 10 batches
  tensorboard: true
  wandb:
    enabled: false
    project: pivot-nodule-detection
    entity: your-team

# Hardware
device:
  type: cuda  # cuda, cpu, mps
  gpu_ids: [0, 1]  # Multi-GPU training
  mixed_precision: true
  cudnn_benchmark: true
```

## Model Architecture

### U-Net Configuration

```yaml
model:
  type: unet
  in_channels: 1
  out_channels: 1
  base_filters: 32  # 32, 64, or 128
  depth: 4  # Number of downsampling levels
  use_batch_norm: true
  use_dropout: true
  dropout_rate: 0.1
  activation: relu  # relu, leaky_relu, elu
```

**Architecture Details:**
- **Encoder**: 4 downsampling blocks with max pooling
- **Decoder**: 4 upsampling blocks with skip connections
- **Filters**: [32, 64, 128, 256, 512] (doubles each level)
- **Output**: Sigmoid activation for heatmap prediction

### Custom Architecture

Define custom models in `src/model/`:

```python
from torch import nn

class CustomDetector(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Define your architecture

    def forward(self, x):
        # Forward pass
        return output
```

Register in config:

```yaml
model:
  type: custom
  class: src.model.custom.CustomDetector
  in_channels: 1
  out_channels: 1
```

## Loss Functions

### Focal Loss (Recommended for Detection)

Handles class imbalance by down-weighting easy examples:

```yaml
training:
  loss:
    type: focal
    alpha: 0.25  # Weight for positive class
    gamma: 2.0   # Focusing parameter (higher = more focus on hard examples)
```

**When to use:** Imbalanced datasets with many background voxels

### Dice Loss

Optimizes overlap between prediction and ground truth:

```yaml
training:
  loss:
    type: dice
    smooth: 1.0
    squared_pred: true
```

**When to use:** Small object detection, direct optimization of Dice metric

### Multi-Task Loss

Combines multiple objectives:

```yaml
training:
  loss:
    type: multi_task
    detection_weight: 1.0   # Nodule detection heatmap
    regression_weight: 0.5  # Size regression
    triage_weight: 0.3      # Malignancy classification
```

**When to use:** Multi-task learning (detection + classification + regression)

### Hard Negative Mining

Focus training on difficult examples:

```yaml
training:
  hard_negative_mining:
    enabled: true
    neg_pos_ratio: 3      # 3 negative samples per positive
    batch_hard_ratio: 0.5 # Use hardest 50% of negatives
```

## Optimization

### Optimizers

#### AdamW (Recommended)

```yaml
training:
  optimizer:
    type: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    eps: 1e-8
```

#### SGD with Momentum

```yaml
training:
  optimizer:
    type: sgd
    learning_rate: 0.01
    momentum: 0.9
    weight_decay: 0.0001
    nesterov: true
```

### Learning Rate Schedules

#### Cosine Annealing (Recommended)

```yaml
training:
  lr_scheduler:
    type: cosine
    warmup_epochs: 5
    min_lr: 0.00001
    T_max: 100  # Total epochs
```

#### Step Decay

```yaml
training:
  lr_scheduler:
    type: step
    step_size: 30
    gamma: 0.1  # Multiply LR by 0.1 every 30 epochs
```

#### Reduce on Plateau

```yaml
training:
  lr_scheduler:
    type: plateau
    mode: max
    factor: 0.5
    patience: 5
    min_lr: 0.00001
    monitor: val_froc
```

### Gradient Clipping

Prevent exploding gradients:

```yaml
training:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
    norm_type: 2
```

## Monitoring Training

### TensorBoard

Enable TensorBoard logging:

```yaml
logging:
  tensorboard: true
  log_dir: logs/tensorboard
```

View training progress:

```bash
tensorboard --logdir logs/tensorboard
```

**Metrics logged:**
- Training/validation loss
- Learning rate
- Gradient norms
- Memory usage
- Sample predictions

### Weights & Biases

```yaml
logging:
  wandb:
    enabled: true
    project: pivot-nodule-detection
    entity: your-team
    tags: [unet, focal-loss, aug]
```

### Training Metrics

Monitor these metrics during training:

- **Detection Dice**: Overlap between predicted and true heatmaps
- **FROC Score**: Free-Response ROC for detection sensitivity
- **Center Accuracy**: Precision of nodule localization
- **Size Accuracy**: Accuracy of diameter estimation
- **Loss Components**: Detection, regression, triage losses

### Early Stopping

Stop training when validation metric plateaus:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 15
    min_delta: 0.001
    monitor: val_froc
    mode: max
```

## Multi-GPU Training

### Data Parallel

```yaml
device:
  type: cuda
  gpu_ids: [0, 1, 2, 3]
  parallel_mode: dp  # Data Parallel
```

### Distributed Data Parallel (Recommended)

```yaml
device:
  type: cuda
  gpu_ids: [0, 1, 2, 3]
  parallel_mode: ddp  # Distributed Data Parallel
```

Launch with:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 -m src.train.main --config configs/train.yaml

# Multi-node (2 nodes, 4 GPUs each)
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.1 --master_port=29500 \
  -m src.train.main --config configs/train.yaml

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.1 --master_port=29500 \
  -m src.train.main --config configs/train.yaml
```

### Mixed Precision Training

Use automatic mixed precision (AMP) for faster training:

```yaml
device:
  mixed_precision: true
  amp_dtype: float16  # or bfloat16
```

**Benefits:**
- 2-3x faster training
- Reduced memory usage
- Larger batch sizes possible

## Advanced Techniques

### Transfer Learning

Start from pretrained weights:

```yaml
training:
  pretrained:
    checkpoint: checkpoints/pretrained_unet.pth
    freeze_encoder: true  # Freeze encoder layers
    freeze_epochs: 10     # Unfreeze after 10 epochs
```

### Progressive Training

Gradually increase patch size:

```yaml
training:
  progressive:
    enabled: true
    stages:
      - epochs: 30
        patch_size: [64, 64, 64]
        batch_size: 8
      - epochs: 40
        patch_size: [96, 96, 96]
        batch_size: 4
      - epochs: 30
        patch_size: [128, 128, 128]
        batch_size: 2
```

### Curriculum Learning

Train on easier examples first:

```yaml
training:
  curriculum:
    enabled: true
    strategy: size_based  # Start with larger nodules
    stages:
      - epochs: 20
        min_diameter: 10  # Only nodules >= 10mm
      - epochs: 30
        min_diameter: 5   # Only nodules >= 5mm
      - epochs: 50
        min_diameter: 0   # All nodules
```

### Test-Time Augmentation

Improve inference by averaging predictions:

```yaml
inference:
  tta:
    enabled: true
    num_augmentations: 8
    augmentation_types:
      - flip_x
      - flip_y
      - flip_z
      - rotate_90
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size: `batch_size: 1`
2. Enable gradient accumulation: `accumulation_steps: 4`
3. Use mixed precision: `mixed_precision: true`
4. Reduce patch size: `patch_size: [96, 96, 96]`
5. Enable gradient checkpointing in model

### Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Solutions:**
1. Check data loading and labels
2. Reduce learning rate: `learning_rate: 0.0001`
3. Try different optimizer: `optimizer: adamw`
4. Add warmup: `warmup_epochs: 5`
5. Check for NaN gradients: `gradient_clipping: true`

### Overfitting

**Symptoms:** Training loss decreases but validation loss increases

**Solutions:**
1. Increase augmentation strength
2. Add dropout: `dropout_rate: 0.2`
3. Increase weight decay: `weight_decay: 0.001`
4. Use early stopping: `patience: 10`
5. Get more training data

### Underfitting

**Symptoms:** Both training and validation loss remain high

**Solutions:**
1. Increase model capacity: `base_filters: 64`, `depth: 5`
2. Train longer: `epochs: 200`
3. Increase learning rate: `learning_rate: 0.01`
4. Reduce regularization: `weight_decay: 0.00001`
5. Check data quality and labels

### Slow Training

**Symptoms:** Training takes too long per epoch

**Solutions:**
1. Enable mixed precision: `mixed_precision: true`
2. Increase num_workers: `num_workers: 16`
3. Use persistent workers: `persistent_workers: true`
4. Enable cudnn benchmark: `cudnn_benchmark: true`
5. Use faster data loading: `prefetch_factor: 4`

### Unstable Training

**Symptoms:** Loss oscillates or spikes

**Solutions:**
1. Reduce learning rate: `learning_rate: 0.0001`
2. Enable gradient clipping: `max_norm: 1.0`
3. Use mixed precision with loss scaling
4. Check for corrupted data samples
5. Use smoother LR schedule

## Training via API

For programmatic control, use the API server:

```bash
# Start API server
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Create training session
curl -X POST http://localhost:8000/api/v1/training/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "configs/train.yaml",
    "session_name": "experiment_001"
  }'

# Monitor progress
curl http://localhost:8000/api/v1/training/sessions/{session_id}/metrics

# Stop training
curl -X POST http://localhost:8000/api/v1/training/sessions/{session_id}/stop
```

See [API Usage Guide](API_USAGE.md) for complete API documentation.

## Example Training Session

Complete example workflow:

```bash
# 1. Prepare data
uv run python scripts/preprocess_data.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --config configs/train.yaml

# 2. Create data splits
uv run python scripts/create_splits.py \
  --data-dir data/processed \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15

# 3. Start training
uv run python -m src.train.main \
  --config configs/train.yaml \
  --experiment-name exp_001

# 4. Monitor training
tensorboard --logdir logs/tensorboard

# 5. Evaluate best checkpoint
uv run python -m src.eval.pipeline \
  --checkpoint checkpoints/best_model.pth \
  --data-dir data/processed/test \
  --output-dir results/exp_001
```

## Next Steps

After training:
1. Review [Inference Guide](INFERENCE.md) for running inference
2. Check [Evaluation Pipeline](../PIPELINE.md) for metrics computation
3. Deploy model using [Deployment Guide](DEPLOYMENT.md)

## References

- [CONFIGURATION.md](../CONFIGURATION.md) - Complete configuration reference
- [PIPELINE.md](../PIPELINE.md) - Training pipeline details
- PyTorch Lightning documentation
- Medical imaging best practices
