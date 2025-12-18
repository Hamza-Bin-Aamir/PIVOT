# Inference Guide

This guide covers how to run inference with trained PIVOT models for lung nodule detection on new CT scans.

## Table of Contents

- [Quick Start](#quick-start)
- [Inference Modes](#inference-modes)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Visualization](#visualization)
- [Performance Optimization](#performance-optimization)
- [API Inference](#api-inference)
- [Batch Processing](#batch-processing)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Single Scan Inference

Run inference on a single CT scan:

```bash
uv run python -m src.inference.main \
  --checkpoint checkpoints/best_model.pth \
  --input scan.nii.gz \
  --output results/
```

This will generate:
- `results/scan_detections.json` - Detected nodule coordinates and properties
- `results/scan_heatmap.nii.gz` - Detection heatmap
- `results/scan_visualization.png` - 2D visualization

### Batch Inference

Process multiple scans:

```bash
uv run python -m src.inference.main \
  --checkpoint checkpoints/best_model.pth \
  --input-dir data/test/scans/ \
  --output-dir results/ \
  --batch-size 4 \
  --num-workers 8
```

## Inference Modes

### 1. Sliding Window Inference

Default mode for full 3D scans:

```yaml
inference:
  mode: sliding_window
  patch_size: [128, 128, 128]
  stride: [64, 64, 64]
  overlap: 0.5
  batch_size: 4
```

**Configuration:**
```python
from src.inference.sliding_window import sliding_window_inference

predictions = sliding_window_inference(
    model=model,
    scan=scan,
    patch_size=[128, 128, 128],
    stride=[64, 64, 64],
    batch_size=4,
    device='cuda'
)
```

### 2. Overlap Blending

Smooth predictions from overlapping patches:

```yaml
inference:
  blending:
    method: gaussian  # gaussian, linear, or average
    sigma: 0.125
```

**Methods:**
- **Gaussian**: Smooth blending with Gaussian weights (recommended)
- **Linear**: Linear blending at patch boundaries
- **Average**: Simple averaging of overlaps

### 3. Whole Volume Inference

For small scans that fit in memory:

```yaml
inference:
  mode: whole_volume
  batch_size: 1
```

### 4. Test-Time Augmentation (TTA)

Improve robustness by averaging augmented predictions:

```yaml
inference:
  tta:
    enabled: true
    num_augmentations: 8
    augmentations:
      - flip_x
      - flip_y
      - flip_z
      - rotate_90_xy
      - rotate_90_xz
      - rotate_90_yz
```

## Configuration

### Complete Inference Config

Create `configs/inference.yaml`:

```yaml
# Model settings
model:
  checkpoint: checkpoints/best_model.pth
  architecture: unet
  device: cuda
  mixed_precision: true

# Input settings
input:
  spacing: [1.0, 1.0, 1.0]  # Target spacing for resampling
  clip_range: [-1000, 400]  # HU window
  normalize: true
  normalization_method: minmax

# Inference settings
inference:
  mode: sliding_window
  patch_size: [128, 128, 128]
  stride: [64, 64, 64]
  batch_size: 4

  blending:
    method: gaussian
    sigma: 0.125

  tta:
    enabled: false
    num_augmentations: 8

# Post-processing
post_processing:
  # Peak detection
  peak_detection:
    threshold: 0.5  # Heatmap threshold
    min_distance: 10  # Minimum distance between peaks (mm)
    method: local_maxima  # local_maxima or nms

  # Non-maximum suppression
  nms:
    enabled: true
    iou_threshold: 0.3

  # Size filtering
  size_filter:
    min_diameter: 3.0  # mm
    max_diameter: 30.0  # mm

  # Confidence filtering
  confidence_filter:
    min_confidence: 0.3

# Output settings
output:
  save_heatmap: true
  save_detections: true
  save_visualization: true
  save_report: true

  formats:
    detections: json  # json, csv, or both
    heatmap: nifti    # nifti or numpy

# Visualization
visualization:
  num_slices: 5  # Number of slices to visualize
  colormap: hot
  overlay_alpha: 0.5
  save_3d: false  # Generate 3D rendering
```

## Output Format

### Detection JSON

`results/scan_detections.json`:

```json
{
  "scan_id": "patient_001",
  "model": "unet_focal_v1",
  "timestamp": "2025-12-19T10:30:00Z",
  "metadata": {
    "spacing": [1.0, 1.0, 1.0],
    "shape": [512, 512, 300]
  },
  "detections": [
    {
      "id": 0,
      "coordinates": [245.5, 312.8, 89.2],
      "coordinates_voxel": [245, 312, 89],
      "diameter_mm": 8.5,
      "confidence": 0.92,
      "properties": {
        "volume_mm3": 321.4,
        "mean_intensity": -450.2,
        "std_intensity": 120.5
      },
      "triage": {
        "malignancy_score": 0.76,
        "risk_category": "high"
      }
    },
    {
      "id": 1,
      "coordinates": [198.3, 267.4, 102.7],
      "coordinates_voxel": [198, 267, 102],
      "diameter_mm": 5.2,
      "confidence": 0.85,
      "properties": {
        "volume_mm3": 73.6,
        "mean_intensity": -620.8,
        "std_intensity": 95.3
      },
      "triage": {
        "malignancy_score": 0.34,
        "risk_category": "low"
      }
    }
  ],
  "summary": {
    "num_detections": 2,
    "high_risk_count": 1,
    "low_risk_count": 1,
    "processing_time_seconds": 12.5
  }
}
```

### Detection CSV

`results/scan_detections.csv`:

```csv
scan_id,detection_id,coord_x,coord_y,coord_z,diameter_mm,confidence,malignancy_score,risk_category
patient_001,0,245.5,312.8,89.2,8.5,0.92,0.76,high
patient_001,1,198.3,267.4,102.7,5.2,0.85,0.34,low
```

### Heatmap Output

The heatmap is saved as a NIfTI file with the same dimensions as the input scan:
- Values: 0.0 (background) to 1.0 (high confidence nodule)
- Spacing: Same as input scan
- Can be loaded with any medical imaging software (ITK-SNAP, 3D Slicer, etc.)

## Visualization

### 2D Slice Visualization

Generate visualization with overlaid detections:

```python
from src.viz.overlay import visualize_detections
from src.data.loader import load_scan
from src.inference.output_formatter import load_detections

# Load scan and detections
scan, spacing = load_scan('scan.nii.gz')
detections = load_detections('results/scan_detections.json')

# Visualize
visualize_detections(
    scan=scan,
    detections=detections,
    spacing=spacing,
    output_path='results/visualization.png',
    num_slices=5,
    colormap='hot',
    alpha=0.5
)
```

### 3D Volume Rendering

Generate 3D visualization:

```python
from src.viz.rendering_3d import render_3d_volume

render_3d_volume(
    scan='scan.nii.gz',
    heatmap='results/scan_heatmap.nii.gz',
    detections='results/scan_detections.json',
    output_path='results/3d_rendering.html',
    interactive=True
)
```

### Confidence Map

Visualize confidence scores:

```python
from src.viz.confidence import plot_confidence_distribution

plot_confidence_distribution(
    detections='results/scan_detections.json',
    output_path='results/confidence_distribution.png'
)
```

## Performance Optimization

### GPU Acceleration

```yaml
inference:
  device: cuda
  mixed_precision: true  # 2-3x speedup
  cudnn_benchmark: true
```

### Batch Processing

Process multiple patches simultaneously:

```yaml
inference:
  batch_size: 8  # Increase for more GPU memory
  num_workers: 8  # Parallel data loading
  prefetch_factor: 2
```

### TorchScript Optimization

Convert model to TorchScript for faster inference:

```python
from src.inference.torchscript_optimizer import optimize_model

# Optimize once
optimized_model = optimize_model(
    checkpoint='checkpoints/best_model.pth',
    output_path='checkpoints/optimized_model.pt',
    example_input_shape=[1, 1, 128, 128, 128]
)

# Use optimized model
import torch
model = torch.jit.load('checkpoints/optimized_model.pt')
```

### Memory Optimization

For large scans or limited GPU memory:

```yaml
inference:
  patch_size: [96, 96, 96]  # Smaller patches
  stride: [80, 80, 80]  # Less overlap
  batch_size: 2  # Smaller batches
  clear_cache: true  # Clear GPU cache between batches
```

## API Inference

### REST API

Start the inference API server:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Submit inference request:

```bash
# Upload scan and run inference
curl -X POST http://localhost:8000/api/v1/inference \
  -F "file=@scan.nii.gz" \
  -F "config=@configs/inference.yaml"

# Get results
curl http://localhost:8000/api/v1/inference/{job_id}/results
```

### Python API

```python
from src.api.client import InferenceClient

# Initialize client
client = InferenceClient(base_url='http://localhost:8000')

# Submit inference
job_id = client.submit_inference(
    scan_path='scan.nii.gz',
    config='configs/inference.yaml'
)

# Wait for completion
results = client.wait_for_results(job_id, timeout=300)

# Download outputs
client.download_results(job_id, output_dir='results/')
```

### WebSocket Streaming

Real-time inference updates:

```python
import asyncio
import websockets

async def stream_inference():
    uri = "ws://localhost:8000/api/v1/inference/stream"
    async with websockets.connect(uri) as websocket:
        # Send scan data
        await websocket.send(scan_bytes)

        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Progress: {data['progress']}%")
            if data['status'] == 'completed':
                print(f"Detections: {data['detections']}")
                break

asyncio.run(stream_inference())
```

## Batch Processing

### Process Directory

```python
from src.inference.main import batch_inference

results = batch_inference(
    checkpoint='checkpoints/best_model.pth',
    input_dir='data/test/scans',
    output_dir='results/',
    config='configs/inference.yaml',
    num_workers=8,
    batch_size=4
)

# Results summary
print(f"Processed {results['num_scans']} scans")
print(f"Total detections: {results['total_detections']}")
print(f"Average processing time: {results['avg_time']:.2f}s")
```

### Parallel Processing

Use multiple GPUs:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 uv run python -m src.inference.main \
  --checkpoint checkpoints/best_model.pth \
  --input-dir data/test/batch_0 \
  --output-dir results/batch_0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python -m src.inference.main \
  --checkpoint checkpoints/best_model.pth \
  --input-dir data/test/batch_1 \
  --output-dir results/batch_1 &

wait
```

### Distributed Inference

Use Ray for large-scale distributed inference:

```python
import ray
from src.inference.distributed import distributed_inference

ray.init(address='auto')

results = distributed_inference(
    checkpoint='checkpoints/best_model.pth',
    input_paths=scan_paths,
    num_gpus=8,
    batch_size=4
)
```

## Post-Processing

### Peak Detection

Extract nodule candidates from heatmap:

```python
from src.inference.peak_detection import detect_peaks

peaks = detect_peaks(
    heatmap=heatmap,
    threshold=0.5,
    min_distance=10,  # mm
    spacing=[1.0, 1.0, 1.0]
)
```

### Non-Maximum Suppression

Remove duplicate detections:

```python
from src.inference.post_processing import non_maximum_suppression

filtered_detections = non_maximum_suppression(
    detections=detections,
    iou_threshold=0.3
)
```

### Nodule Properties

Extract detailed nodule characteristics:

```python
from src.inference.nodule_properties import extract_properties

for detection in detections:
    properties = extract_properties(
        scan=scan,
        coordinates=detection['coordinates'],
        diameter=detection['diameter_mm'],
        spacing=spacing
    )

    print(f"Volume: {properties['volume_mm3']:.1f} mm³")
    print(f"Mean HU: {properties['mean_intensity']:.1f}")
    print(f"Shape: {properties['sphericity']:.2f}")
```

### Triage Calibration

Calibrate malignancy scores:

```python
from src.inference.triage_calibration import calibrate_scores

calibrated_detections = calibrate_scores(
    detections=detections,
    calibration_file='calibration/triage_calibration.json'
)
```

## Troubleshooting

### Out of Memory

**Solution:** Reduce patch size or batch size

```yaml
inference:
  patch_size: [96, 96, 96]
  batch_size: 2
```

### Slow Inference

**Solutions:**
1. Enable mixed precision: `mixed_precision: true`
2. Use TorchScript optimization
3. Increase batch size: `batch_size: 8`
4. Reduce overlap: `stride: [96, 96, 96]`

### Poor Detection Quality

**Solutions:**
1. Adjust threshold: `threshold: 0.3`
2. Enable TTA: `tta.enabled: true`
3. Check preprocessing matches training
4. Verify model checkpoint is correct

### Missing Small Nodules

**Solutions:**
1. Lower threshold: `threshold: 0.3`
2. Reduce min_distance: `min_distance: 5`
3. Use smaller stride: `stride: [48, 48, 48]`
4. Enable TTA

### Too Many False Positives

**Solutions:**
1. Increase threshold: `threshold: 0.6`
2. Enable NMS: `nms.enabled: true`
3. Add size filtering: `min_diameter: 4.0`
4. Increase confidence filter: `min_confidence: 0.5`

## Example Inference Pipeline

Complete example:

```python
import torch
from pathlib import Path
from src.inference.main import InferencePipeline
from src.data.loader import load_scan
from src.viz.overlay import visualize_detections
from src.viz.report import generate_report

# Initialize pipeline
pipeline = InferencePipeline(
    checkpoint='checkpoints/best_model.pth',
    config='configs/inference.yaml',
    device='cuda'
)

# Process scan
scan_path = 'data/test/patient_001.nii.gz'
scan, spacing = load_scan(scan_path)

# Run inference
results = pipeline.predict(scan, spacing)

# Save outputs
output_dir = Path('results/patient_001')
output_dir.mkdir(parents=True, exist_ok=True)

# Save detections
pipeline.save_detections(results, output_dir / 'detections.json')

# Save heatmap
pipeline.save_heatmap(results['heatmap'], output_dir / 'heatmap.nii.gz')

# Generate visualization
visualize_detections(
    scan=scan,
    detections=results['detections'],
    spacing=spacing,
    output_path=output_dir / 'visualization.png'
)

# Generate report
generate_report(
    scan_path=scan_path,
    detections=results['detections'],
    output_path=output_dir / 'report.pdf'
)

print(f"Found {len(results['detections'])} nodules")
```

## Next Steps

- Review [Evaluation Pipeline](../PIPELINE.md) for metrics computation
- Deploy inference service using [Deployment Guide](DEPLOYMENT.md)
- Integrate with PACS using DICOM endpoints

## References

- [PIPELINE.md](../PIPELINE.md) - Complete inference pipeline
- Medical imaging best practices
- ITK-SNAP visualization tutorial
