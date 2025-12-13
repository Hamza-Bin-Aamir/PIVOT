"""Data processing, loading, and augmentation modules."""

from .augment import (
    AugmentationConfig,
    Compose,
    RandomFlip3D,
    RandomGaussianNoise,
    RandomIntensityScale,
    RandomRotate90,
    build_default_augmentation_pipeline,
)
from .dataset import LUNADataset
from .intensity import (
    HistogramMethod,
    NormalizationStats,
    clip_hounsfield,
    normalize_intensity,
)
from .loader import (
    DICOMLoader,
    MedicalScanLoader,
    NIfTILoader,
    ScanMetadata,
)
from .preprocess import apply_hu_windowing, normalize_to_range, preprocess_ct_scan
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
    # Augmentations
    "AugmentationConfig",
    "Compose",
    "RandomFlip3D",
    "RandomGaussianNoise",
    "RandomIntensityScale",
    "RandomRotate90",
    "build_default_augmentation_pipeline",
    # Loaders
    "DICOMLoader",
    "MedicalScanLoader",
    "NIfTILoader",
    "ScanMetadata",
    # Preprocessing
    "HistogramMethod",
    "NormalizationStats",
    "clip_hounsfield",
    "normalize_intensity",
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
