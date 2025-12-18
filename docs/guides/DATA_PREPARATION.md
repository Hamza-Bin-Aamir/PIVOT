# Data Preparation Guide

This guide covers how to prepare your medical imaging data for training the PIVOT lung nodule detection model.

## Table of Contents

- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Directory Structure](#directory-structure)
- [Annotation Format](#annotation-format)
- [Data Preprocessing](#data-preprocessing)
- [Validation](#validation)
- [Best Practices](#best-practices)

## Overview

PIVOT requires CT scans in specific formats along with corresponding annotations for nodule locations and sizes. The data preparation pipeline includes:

1. Converting DICOM files to NIfTI format
2. Creating annotation files in the required format
3. Organizing data into the expected directory structure
4. Running validation checks
5. Preprocessing (resampling, intensity normalization)

## Data Requirements

### CT Scans

- **Format**: NIfTI (`.nii` or `.nii.gz`) or DICOM
- **Modality**: CT (Computed Tomography)
- **Resolution**: Variable (will be resampled to target spacing)
- **Bit Depth**: 16-bit preferred
- **File Naming**: Unique identifier per scan (e.g., `patient_001.nii.gz`)

### Annotations

- **Format**: CSV or JSON
- **Required Fields**:
  - Scan identifier
  - Nodule coordinates (x, y, z in mm)
  - Nodule diameter (mm)
  - Optional: nodule type, malignancy score

## Directory Structure

Organize your data in the following structure:

```
data/
├── raw/
│   ├── scans/
│   │   ├── patient_001.nii.gz
│   │   ├── patient_002.nii.gz
│   │   └── ...
│   └── annotations/
│       └── annotations.csv
├── processed/
│   ├── scans/
│   └── annotations/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Split Files

Each split file (`train.txt`, `val.txt`, `test.txt`) contains one scan identifier per line:

```
patient_001
patient_002
patient_003
```

**Recommended splits:**
- Training: 70-80%
- Validation: 10-15%
- Testing: 10-15%

## Annotation Format

### CSV Format

The annotation CSV should have the following columns:

```csv
scan_id,coord_x,coord_y,coord_z,diameter_mm,nodule_type,malignancy
patient_001,245.5,312.8,89.2,8.5,solid,1
patient_001,198.3,267.4,102.7,5.2,ground_glass,0
patient_002,321.6,289.1,76.3,12.8,solid,1
```

**Column Descriptions:**
- `scan_id`: Unique identifier matching the scan filename (without extension)
- `coord_x`, `coord_y`, `coord_z`: Nodule center coordinates in mm (physical space)
- `diameter_mm`: Nodule diameter in millimeters
- `nodule_type`: (Optional) Type of nodule (solid, ground_glass, part_solid)
- `malignancy`: (Optional) Binary malignancy label (0=benign, 1=malignant)

### JSON Format

Alternatively, use JSON format:

```json
{
  "patient_001": [
    {
      "coordinates": [245.5, 312.8, 89.2],
      "diameter_mm": 8.5,
      "nodule_type": "solid",
      "malignancy": 1
    },
    {
      "coordinates": [198.3, 267.4, 102.7],
      "diameter_mm": 5.2,
      "nodule_type": "ground_glass",
      "malignancy": 0
    }
  ],
  "patient_002": [
    {
      "coordinates": [321.6, 289.1, 76.3],
      "diameter_mm": 12.8,
      "nodule_type": "solid",
      "malignancy": 1
    }
  ]
}
```

## Data Preprocessing

### 1. Convert DICOM to NIfTI

If you have DICOM files, convert them to NIfTI format:

```python
from src.data.loader import convert_dicom_to_nifti

convert_dicom_to_nifti(
    dicom_dir='path/to/dicom/series',
    output_path='data/raw/scans/patient_001.nii.gz'
)
```

### 2. Resample to Target Spacing

Resample all scans to a consistent voxel spacing:

```python
from src.data.resampling import resample_scan

# Load and resample
scan, spacing = load_scan('data/raw/scans/patient_001.nii.gz')
resampled_scan = resample_scan(
    scan,
    current_spacing=spacing,
    target_spacing=[1.0, 1.0, 1.0],  # 1mm isotropic
    order=3  # cubic interpolation
)

# Save resampled scan
save_scan(resampled_scan, 'data/processed/scans/patient_001.nii.gz')
```

**Configuration in `config.yaml`:**

```yaml
preprocessing:
  target_spacing: [1.0, 1.0, 1.0]
  resampling_order: 3
  clip_range: [-1000, 400]  # HU range for lung CT
```

### 3. Intensity Normalization

Normalize intensity values for consistent training:

```python
from src.data.intensity import normalize_intensity, clip_intensity

# Clip to lung window
scan = clip_intensity(scan, min_value=-1000, max_value=400)

# Normalize to [0, 1]
scan = normalize_intensity(scan, method='minmax')

# Or use z-score normalization
scan = normalize_intensity(scan, method='zscore', clip_range=(-3, 3))
```

### 4. Generate Heatmap Labels

Convert point annotations to 3D heatmap targets:

```python
from src.data.heatmap import generate_heatmap

heatmap = generate_heatmap(
    scan_shape=scan.shape,
    coordinates=[[245.5, 312.8, 89.2], [198.3, 267.4, 102.7]],
    diameters=[8.5, 5.2],
    spacing=[1.0, 1.0, 1.0],
    sigma_scale=1.0
)
```

### 5. Complete Preprocessing Pipeline

Use the automated preprocessing script:

```bash
# Preprocess entire dataset
uv run python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --config configs/train.yaml \
    --num-workers 8
```

This will:
- Load raw scans and annotations
- Resample to target spacing
- Normalize intensities
- Generate heatmap labels
- Save processed data with metadata

## Validation

### 1. Check Annotation Format

Validate annotation files:

```python
from src.data.validation import validate_annotations

errors = validate_annotations('data/raw/annotations/annotations.csv')
if errors:
    for error in errors:
        print(f"Error: {error}")
else:
    print("All annotations valid!")
```

### 2. Verify Scan-Annotation Pairing

Ensure all annotated scans exist:

```python
from src.data.validation import check_scan_annotation_pairs

missing, extra = check_scan_annotation_pairs(
    scans_dir='data/raw/scans',
    annotations_file='data/raw/annotations/annotations.csv'
)

print(f"Missing scans: {missing}")
print(f"Scans without annotations: {extra}")
```

### 3. Check Coordinate Validity

Verify nodule coordinates are within scan bounds:

```python
from src.data.validation import validate_coordinates

invalid = validate_coordinates(
    scans_dir='data/raw/scans',
    annotations_file='data/raw/annotations/annotations.csv'
)

for scan_id, coords in invalid.items():
    print(f"{scan_id}: Invalid coordinates {coords}")
```

### 4. Visualize Annotations

Preview annotations on scans:

```python
from src.data.loader import load_scan
from src.data.annotations import load_annotations
from src.viz.overlay import visualize_annotations

scan, spacing = load_scan('data/raw/scans/patient_001.nii.gz')
annotations = load_annotations('data/raw/annotations/annotations.csv')

visualize_annotations(
    scan,
    annotations['patient_001'],
    spacing=spacing,
    slice_index=50,
    output_path='preview.png'
)
```

## Best Practices

### 1. Data Quality

- **Consistent Acquisition**: Use scans from similar protocols and scanners
- **Quality Control**: Review all scans for artifacts, truncation, or poor quality
- **Annotation Accuracy**: Double-check nodule coordinates and sizes
- **Balanced Dataset**: Include diverse nodule sizes, types, and locations

### 2. Data Augmentation

Configure augmentation in your training config:

```yaml
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
```

### 3. Data Splits

- **Stratification**: Ensure balanced nodule sizes across splits
- **Patient-Level Split**: Never split scans from same patient across train/val/test
- **Random Seed**: Use fixed seed for reproducible splits

```python
from src.data.validation import create_data_splits

create_data_splits(
    scans_dir='data/raw/scans',
    annotations_file='data/raw/annotations/annotations.csv',
    output_dir='data/splits',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by='nodule_size',
    random_seed=42
)
```

### 4. Handling Class Imbalance

For malignancy classification or triage:

```yaml
training:
  loss:
    type: weighted_bce
    pos_weight: 3.0  # Weight positive class higher

  sampling:
    type: hard_negative_mining
    neg_pos_ratio: 3
    batch_hard_ratio: 0.5
```

### 5. Large Dataset Management

For datasets that don't fit in memory:

- Use the `LazyDataset` loader with on-disk caching
- Enable memory-mapped file loading
- Use streaming dataloaders with prefetching

```yaml
dataset:
  type: lazy
  cache_dir: data/cache
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
```

## Example Workflow

Complete example for preparing a new dataset:

```python
import os
from pathlib import Path
from src.data.loader import load_scan, save_scan
from src.data.resampling import resample_scan
from src.data.intensity import normalize_intensity, clip_intensity
from src.data.annotations import load_annotations, convert_coordinates
from src.data.heatmap import generate_heatmap
from src.data.validation import validate_annotations, check_scan_annotation_pairs

# 1. Setup directories
data_dir = Path('data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# 2. Validate annotations
print("Validating annotations...")
errors = validate_annotations(raw_dir / 'annotations' / 'annotations.csv')
assert not errors, f"Annotation errors: {errors}"

# 3. Check scan-annotation pairs
print("Checking scan-annotation pairs...")
missing, extra = check_scan_annotation_pairs(
    raw_dir / 'scans',
    raw_dir / 'annotations' / 'annotations.csv'
)
assert not missing, f"Missing scans: {missing}"

# 4. Load annotations
annotations = load_annotations(raw_dir / 'annotations' / 'annotations.csv')

# 5. Process each scan
target_spacing = [1.0, 1.0, 1.0]
for scan_file in (raw_dir / 'scans').glob('*.nii.gz'):
    scan_id = scan_file.stem.replace('.nii', '')
    print(f"Processing {scan_id}...")

    # Load scan
    scan, spacing = load_scan(scan_file)

    # Resample
    scan_resampled = resample_scan(scan, spacing, target_spacing, order=3)

    # Normalize intensity
    scan_normalized = clip_intensity(scan_resampled, -1000, 400)
    scan_normalized = normalize_intensity(scan_normalized, method='minmax')

    # Convert annotation coordinates
    if scan_id in annotations:
        coords = [ann['coordinates'] for ann in annotations[scan_id]]
        diameters = [ann['diameter_mm'] for ann in annotations[scan_id]]

        # Convert from physical to voxel coordinates
        coords_voxel = convert_coordinates(coords, spacing, target_spacing)

        # Generate heatmap
        heatmap = generate_heatmap(
            scan_normalized.shape,
            coords_voxel,
            diameters,
            target_spacing
        )

        # Save heatmap
        save_scan(
            heatmap,
            processed_dir / 'labels' / f'{scan_id}_heatmap.nii.gz',
            spacing=target_spacing
        )

    # Save processed scan
    save_scan(
        scan_normalized,
        processed_dir / 'scans' / f'{scan_id}.nii.gz',
        spacing=target_spacing
    )

print("Data preparation complete!")
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Process scans one at a time, use memory-mapped loading, or reduce batch size

### Issue: Coordinate Mismatch

**Solution**: Verify coordinate system (physical vs voxel), check spacing/origin metadata

### Issue: Poor Training Performance

**Solution**: Check data quality, increase augmentation, verify annotation accuracy, balance dataset

### Issue: Slow Preprocessing

**Solution**: Use multiprocessing (`--num-workers`), cache preprocessed data, use SSD storage

## Next Steps

After data preparation:
1. Review [Training Guide](TRAINING.md) for model training
2. Configure training parameters in `configs/train.yaml`
3. Start training with `uv run python -m src.train.main`

## References

- [ANNOTATIONS.md](../ANNOTATIONS.md) - Detailed annotation format specification
- [PIPELINE.md](../PIPELINE.md) - Complete data pipeline documentation
- Example datasets: LUNA16, LIDC-IDRI
