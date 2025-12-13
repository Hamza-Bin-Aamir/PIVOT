# Medical Scan Loading Examples

This directory contains example scripts demonstrating how to use the medical imaging file loading functionality.

## Examples

### `load_medical_scans.py`

Comprehensive examples of loading various medical imaging formats:

- **DICOM Series Loading**: Load multi-slice CT scans from DICOM directories
- **Single DICOM File**: Load individual DICOM files
- **NIfTI Files**: Load compressed and uncompressed NIfTI files
- **Automatic Format Detection**: Let the loader detect the file format automatically
- **LUNA16 Workflow**: Complete example for LUNA16 dataset
- **LIDC-IDRI Workflow**: Complete example for LIDC-IDRI dataset
- **Metadata Analysis**: Extract and analyze scan metadata

### `resample_medical_scan.py`

End-to-end isotropic resampling workflow:

- **Automatic Loading**: Use `MedicalScanLoader` to ingest scans or DICOM series
- **Shape Forecasting**: Estimate output shape before running the resampler
- **Intensity Preservation**: Resample images while preserving HU ranges
- **Mask Support**: Resample aligned segmentation masks using nearest neighbour
- **Safety Guard**: Configure `--max-voxels` to avoid out-of-memory scenarios

## Quick Start

```bash
# Run all examples
python examples/load_medical_scans.py

# Or import specific functions in your code
python -c "from examples.load_medical_scans import example_luna16_workflow; example_luna16_workflow()"
```

## Usage

```python
from src.data.loader import MedicalScanLoader

# Load any supported format
volume, metadata = MedicalScanLoader.load("/path/to/scan")

# Access volume data
print(f"Shape: {volume.shape}")
print(f"HU range: [{volume.min()}, {volume.max()}]")

# Access metadata
print(f"Spacing: {metadata.spacing} mm")
print(f"Patient ID: {metadata.patient_id}")
print(f"Modality: {metadata.modality}")
```

## Supported Formats

- **DICOM** (`.dcm`, series directories)
- **NIfTI** (`.nii`, `.nii.gz`)
- **MetaImage** (`.mhd`, `.mha`)

## Notes

- Update file paths in examples to match your dataset location
- LUNA16 uses MetaImage format (`.mhd` files)
- LIDC-IDRI uses DICOM format (series directories)
- All loaders return numpy arrays and comprehensive metadata
