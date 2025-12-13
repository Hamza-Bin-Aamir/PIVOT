"""Data processing, loading, and augmentation modules."""

from .dataset import LUNADataset
from .loader import (
    DICOMLoader,
    MedicalScanLoader,
    NIfTILoader,
    ScanMetadata,
)
from .preprocess import (
    apply_hu_windowing,
    normalize_to_range,
    preprocess_ct_scan,
)
from .resampling import (
    calculate_isotropic_shape,
    calculate_resampling_factor,
    is_isotropic,
    resample_mask,
    resample_to_isotropic,
)

__all__ = [
    # Dataset
    "LUNADataset",
    # Loaders
    "DICOMLoader",
    "MedicalScanLoader",
    "NIfTILoader",
    "ScanMetadata",
    # Preprocessing
    "apply_hu_windowing",
    "normalize_to_range",
    "preprocess_ct_scan",
    # Resampling utilities
    "calculate_isotropic_shape",
    "calculate_resampling_factor",
    "is_isotropic",
    "resample_mask",
    "resample_to_isotropic",
]
