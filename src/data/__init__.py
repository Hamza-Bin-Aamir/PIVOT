"""Data processing, loading, and augmentation modules."""

from .annotations import (
    LUNA16Annotation,
    group_annotations_by_series,
    parse_luna16_annotations,
)
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
from .heatmap import HeatmapConfig, generate_center_heatmap
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
from .triage import (
    TriageScoreBreakdown,
    aggregate_lidc_characteristics,
    compute_triage_score,
)
from .validation import (
    ValidationResult,
    VolumeQualityMetrics,
    check_zero_slices,
    compute_quality_metrics,
    summarize_validation_results,
    validate_annotation_bounds,
    validate_dataset_structure,
    validate_file_exists,
    validate_intensity_range,
    validate_mask_consistency,
    validate_sitk_image,
    validate_spacing,
    validate_volume_shape,
)

__all__ = [
    # Annotations
    "LUNA16Annotation",
    "group_annotations_by_series",
    "parse_luna16_annotations",
    "TriageScoreBreakdown",
    "aggregate_lidc_characteristics",
    "compute_triage_score",
    # Dataset
    "LUNADataset",
    "HeatmapConfig",
    "generate_center_heatmap",
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
    # Validation
    "ValidationResult",
    "VolumeQualityMetrics",
    "check_zero_slices",
    "compute_quality_metrics",
    "summarize_validation_results",
    "validate_annotation_bounds",
    "validate_dataset_structure",
    "validate_file_exists",
    "validate_intensity_range",
    "validate_mask_consistency",
    "validate_sitk_image",
    "validate_spacing",
    "validate_volume_shape",
]
