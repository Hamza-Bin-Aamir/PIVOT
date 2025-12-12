# Pulmonary Imaging for Volume Oncology Triage

This repository contains resources and tools for the Pulmonary Imaging for Volume Oncology Triage (PIVOT) project. The project uses Pulmonary Imaging to assist in the triage of oncology patients.

## Dataset

This project employs a dual-source data strategy to ensure both generalization against academic benchmarks and specificity to clinical environments. The model is pre-trained on the public LUNA16 dataset (a subset of LIDC-IDRI) and fine-tuned on private clinical data.

### 1. Public Benchmark: LUNA16 / LIDC-IDRI

The primary training and validation are conducted using the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is a curated subset of the larger **LIDC-IDRI** database.

- **Source:** [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/)
- **Composition:** 888 CT scans with slice thickness ≤ 2.5mm.
- **Annotations:** 1,186 nodules marked as "positive" (accepted by at least 3 out of 4 radiologists).
- **Inclusion Criteria:** Nodules with a diameter ≥ 3mm. Non-nodules, nodules < 3mm, and nodules with ambiguous consensus are excluded from the positive class during training but may be used for hard negative mining.

### 2. Private Clinical Data (Optional/Confidential)

For domain adaptation, the model supports fine-tuning on DICOM datasets provided by partner institutions.

- **Format:** Standard DICOM series.
- **Annotations:** Radiologist-verified bounding boxes or segmentation masks.
- **Note:** Due to patient privacy (HIPAA/GDPR), private datasets are **not** included in this repository.

## Preprocessing Pipeline

Raw CT data (DICOM or MHD) undergoes a rigorous 3D preprocessing pipeline before entering the network. The pipeline is implemented in `src/data/preprocess.py`.

1. **Resampling:** All scans are resampled to an isotropic resolution of `1mm x 1mm x 1mm` to handle varying slice thicknesses across scanners.
2. **HU Windowing:** Pixel intensities are clipped to the standard lung window:
   - **Level:** -600 HU
   - **Width:** 1500 HU (Range: -1350 to +150)
3. **Normalization:** Windowed values are linearly scaled to the range `[0, 1]`.
4. **Patch Generation:** The volume is cropped into 3D patches of size `(96, 96, 96)` for training.

## Directory Structure

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
