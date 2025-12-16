"""Unit tests for data validation module."""

import numpy as np
import SimpleITK as sitk

from src.data.validation import (
    ValidationResult,
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


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(True, "Test message", "info")
        assert result.is_valid is True
        assert result.message == "Test message"
        assert result.severity == "info"

    def test_validation_result_bool_conversion(self):
        """Test validation result can be used in boolean context."""
        valid_result = ValidationResult(True, "Pass")
        invalid_result = ValidationResult(False, "Fail")

        assert bool(valid_result) is True
        assert bool(invalid_result) is False
        assert valid_result
        assert not invalid_result

    def test_validation_result_default_severity(self):
        """Test default severity is error."""
        result = ValidationResult(False, "Test")
        assert result.severity == "error"


class TestVolumeShapeValidation:
    """Tests for volume shape validation."""

    def test_validate_volume_shape_valid(self):
        """Test validation passes for valid volume shape."""
        volume = np.random.randn(64, 128, 128)
        result = validate_volume_shape(volume)
        assert result.is_valid is True
        assert "valid" in result.message.lower()

    def test_validate_volume_shape_too_small(self):
        """Test validation fails for volume smaller than minimum."""
        volume = np.random.randn(16, 16, 16)
        result = validate_volume_shape(volume, min_size=(32, 32, 32))
        assert result.is_valid is False
        assert "smaller than minimum" in result.message

    def test_validate_volume_shape_too_large(self):
        """Test validation fails for volume larger than maximum."""
        volume = np.random.randn(600, 600, 600)
        result = validate_volume_shape(volume, min_size=(32, 32, 32), max_size=(512, 512, 512))
        assert result.is_valid is False
        assert "exceeds maximum" in result.message

    def test_validate_volume_shape_not_3d(self):
        """Test validation fails for non-3D array."""
        volume_2d = np.random.randn(128, 128)
        result = validate_volume_shape(volume_2d)
        assert result.is_valid is False
        assert "3D volume" in result.message

    def test_validate_volume_shape_custom_limits(self):
        """Test validation with custom size limits."""
        volume = np.random.randn(50, 50, 50)
        result = validate_volume_shape(volume, min_size=(40, 40, 40), max_size=(60, 60, 60))
        assert result.is_valid is True


class TestSpacingValidation:
    """Tests for voxel spacing validation."""

    def test_validate_spacing_valid_isotropic(self):
        """Test validation passes for valid isotropic spacing."""
        spacing = (1.0, 1.0, 1.0)
        results = validate_spacing(spacing)
        assert len(results) == 1
        assert results[0].is_valid is True

    def test_validate_spacing_valid_anisotropic(self):
        """Test validation passes for valid anisotropic spacing with warning."""
        spacing = (2.5, 0.7, 0.7)
        results = validate_spacing(spacing)
        # Should pass but with a warning
        assert all(r.is_valid for r in results)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) > 0
        assert "anisotropic" in warnings[0].message.lower()

    def test_validate_spacing_negative_value(self):
        """Test validation fails for negative spacing."""
        spacing = (1.0, -0.5, 1.0)
        results = validate_spacing(spacing)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "positive" in errors[0].message.lower()

    def test_validate_spacing_zero_value(self):
        """Test validation fails for zero spacing."""
        spacing = (1.0, 0.0, 1.0)
        results = validate_spacing(spacing)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "positive" in errors[0].message.lower()

    def test_validate_spacing_too_small(self):
        """Test validation fails for spacing below minimum."""
        spacing = (0.05, 0.05, 0.05)
        results = validate_spacing(spacing, min_spacing=0.1)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "below minimum" in errors[0].message

    def test_validate_spacing_too_large(self):
        """Test validation fails for spacing above maximum."""
        spacing = (15.0, 1.0, 1.0)
        results = validate_spacing(spacing, max_spacing=10.0)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "above maximum" in errors[0].message

    def test_validate_spacing_wrong_dimensions(self):
        """Test validation fails for wrong number of spacing values."""
        spacing = (1.0, 1.0)
        results = validate_spacing(spacing)
        assert results[0].is_valid is False
        assert "3 spacing values" in results[0].message

    def test_validate_spacing_disable_anisotropy_warning(self):
        """Test can disable anisotropy warning."""
        spacing = (2.5, 0.7, 0.7)
        results = validate_spacing(spacing, warn_anisotropic=False)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) == 0


class TestIntensityRangeValidation:
    """Tests for intensity range validation."""

    def test_validate_intensity_range_valid(self):
        """Test validation passes for typical CT values."""
        volume = np.random.randint(-1000, 400, size=(64, 64, 64), dtype=np.int16)
        results = validate_intensity_range(volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0

    def test_validate_intensity_range_nan_values(self):
        """Test validation fails for NaN values."""
        volume = np.random.randn(32, 32, 32)
        volume[10, 10, 10] = np.nan
        results = validate_intensity_range(volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "nan" in errors[0].message.lower()

    def test_validate_intensity_range_inf_values(self):
        """Test validation fails for infinite values."""
        volume = np.random.randn(32, 32, 32)
        volume[15, 15, 15] = np.inf
        results = validate_intensity_range(volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "infinite" in errors[0].message.lower()

    def test_validate_intensity_range_below_minimum(self):
        """Test validation fails for values below minimum HU."""
        volume = np.full((32, 32, 32), -2500.0)
        results = validate_intensity_range(volume, min_hu=-2000)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "below expected range" in errors[0].message

    def test_validate_intensity_range_above_maximum(self):
        """Test validation fails for values above maximum HU."""
        volume = np.full((32, 32, 32), 3500.0)
        results = validate_intensity_range(volume, max_hu=3000)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "above expected range" in errors[0].message

    def test_validate_intensity_range_unusual_mean_warning(self):
        """Test warning for unusual mean intensity."""
        volume = np.full((32, 32, 32), -950.0)  # Very air-heavy
        results = validate_intensity_range(volume, warn_unusual=True)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) > 0
        assert "unusual mean" in warnings[0].message.lower()

    def test_validate_intensity_range_low_variance_warning(self):
        """Test warning for suspiciously low variance."""
        volume = np.full((32, 32, 32), -500.0)  # Use typical mean
        volume += np.random.randn(32, 32, 32) * 0.5  # Very small noise
        results = validate_intensity_range(volume, warn_unusual=True)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) > 0
        assert any("variance" in w.message.lower() for w in warnings)

    def test_validate_intensity_range_disable_warnings(self):
        """Test can disable unusual value warnings."""
        volume = np.full((32, 32, 32), -950.0)
        results = validate_intensity_range(volume, warn_unusual=False)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) == 0


class TestAnnotationBoundsValidation:
    """Tests for annotation bounds validation."""

    def test_validate_annotation_bounds_valid(self):
        """Test validation passes for in-bounds annotations."""
        annotations = np.array([[10, 20, 30], [40, 50, 60]])
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is True

    def test_validate_annotation_bounds_empty_annotations(self):
        """Test validation passes for empty annotations."""
        annotations = np.array([]).reshape(0, 3)
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is True
        assert "no annotations" in result.message.lower()

    def test_validate_annotation_bounds_out_of_bounds_z(self):
        """Test validation fails for z coordinate out of bounds."""
        annotations = np.array([[70, 30, 30]])  # z=70 exceeds z_max=64
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is False
        assert "out of bounds" in result.message

    def test_validate_annotation_bounds_negative_coordinate(self):
        """Test validation fails for negative coordinates."""
        annotations = np.array([[-5, 30, 30]])
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is False

    def test_validate_annotation_bounds_with_margin(self):
        """Test validation allows coordinates outside bounds with margin."""
        annotations = np.array([[65, 30, 30]])  # 1 voxel outside
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape, allow_margin=5)
        assert result.is_valid is True

    def test_validate_annotation_bounds_wrong_shape(self):
        """Test validation fails for wrong annotation shape."""
        annotations = np.array([[10, 20]])  # Only 2 coordinates
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is False
        assert "Nx3" in result.message

    def test_validate_annotation_bounds_multiple_out_of_bounds(self):
        """Test validation reports multiple out of bounds annotations."""
        annotations = np.array(
            [
                [70, 30, 30],
                [30, 70, 30],
                [30, 30, 70],
            ]
        )
        volume_shape = (64, 64, 64)
        result = validate_annotation_bounds(annotations, volume_shape)
        assert result.is_valid is False
        assert "out of bounds" in result.message


class TestMaskConsistencyValidation:
    """Tests for mask consistency validation."""

    def test_validate_mask_consistency_valid_binary(self):
        """Test validation passes for valid binary mask."""
        volume = np.random.randn(64, 64, 64)
        mask = np.random.rand(64, 64, 64) > 0.7
        results = validate_mask_consistency(mask, volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0

    def test_validate_mask_consistency_shape_mismatch(self):
        """Test validation fails for shape mismatch."""
        volume = np.random.randn(64, 64, 64)
        mask = np.random.rand(32, 32, 32) > 0.5
        results = validate_mask_consistency(mask, volume)
        assert results[0].is_valid is False
        assert "shape" in results[0].message.lower()

    def test_validate_mask_consistency_negative_values(self):
        """Test validation fails for negative mask values."""
        volume = np.random.randn(64, 64, 64)
        mask = np.random.randint(-1, 2, size=(64, 64, 64))
        results = validate_mask_consistency(mask, volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert "negative" in errors[0].message.lower()

    def test_validate_mask_consistency_high_coverage_warning(self):
        """Test warning for mask covering too much of volume."""
        volume = np.random.randn(64, 64, 64)
        mask = np.ones((64, 64, 64), dtype=bool)  # 100% coverage
        results = validate_mask_consistency(mask, volume, max_mask_ratio=0.3)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) > 0

    def test_validate_mask_consistency_empty_mask_warning(self):
        """Test warning for completely empty mask."""
        volume = np.random.randn(64, 64, 64)
        mask = np.zeros((64, 64, 64), dtype=bool)
        results = validate_mask_consistency(mask, volume)
        warnings = [r for r in results if r.severity == "warning"]
        assert len(warnings) > 0
        assert "empty" in warnings[0].message.lower()

    def test_validate_mask_consistency_multiclass_mask(self):
        """Test validation passes for multi-class mask."""
        volume = np.random.randn(64, 64, 64)
        mask = np.random.randint(0, 4, size=(64, 64, 64))
        results = validate_mask_consistency(mask, volume)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0


class TestZeroSliceDetection:
    """Tests for zero slice detection."""

    def test_check_zero_slices_no_zeros(self):
        """Test detection when no zero slices exist."""
        volume = np.random.randn(64, 64, 64) + 100
        count, indices = check_zero_slices(volume, axis=0)
        assert count == 0
        assert len(indices) == 0

    def test_check_zero_slices_some_zeros(self):
        """Test detection of zero slices."""
        volume = np.random.randn(64, 64, 64)
        volume[10, :, :] = 0
        volume[20, :, :] = 0
        count, indices = check_zero_slices(volume, axis=0)
        assert count == 2
        assert 10 in indices
        assert 20 in indices

    def test_check_zero_slices_near_zero(self):
        """Test detection with small threshold."""
        volume = np.random.randn(64, 64, 64)
        volume[15, :, :] = 1e-8  # Very small but not zero
        count, indices = check_zero_slices(volume, axis=0, threshold=1e-6)
        assert 15 in indices

    def test_check_zero_slices_different_axes(self):
        """Test detection along different axes."""
        volume = np.random.randn(64, 64, 64)
        volume[:, 25, :] = 0  # Zero slice in y-axis

        count_z, _ = check_zero_slices(volume, axis=0)
        count_y, indices_y = check_zero_slices(volume, axis=1)
        count_x, _ = check_zero_slices(volume, axis=2)

        assert count_z == 0
        assert count_y == 1
        assert 25 in indices_y
        assert count_x == 0


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    def test_compute_quality_metrics_basic(self):
        """Test basic quality metrics computation."""
        volume = np.random.randn(64, 128, 128) * 100 - 500
        metrics = compute_quality_metrics(volume)

        assert metrics.shape == (64, 128, 128)
        assert metrics.spacing == (1.0, 1.0, 1.0)
        assert metrics.origin == (0.0, 0.0, 0.0)
        assert len(metrics.intensity_range) == 2
        assert isinstance(metrics.intensity_mean, float)
        assert isinstance(metrics.intensity_std, float)
        assert isinstance(metrics.has_nan, bool)
        assert isinstance(metrics.has_inf, bool)

    def test_compute_quality_metrics_with_spacing(self):
        """Test quality metrics with custom spacing."""
        volume = np.random.randn(64, 64, 64)
        spacing = (2.5, 0.7, 0.7)
        metrics = compute_quality_metrics(volume, spacing=spacing)

        assert metrics.spacing == spacing
        assert metrics.is_isotropic is False

    def test_compute_quality_metrics_detects_nan(self):
        """Test quality metrics detects NaN values."""
        volume = np.random.randn(32, 32, 32)
        volume[10, 10, 10] = np.nan
        metrics = compute_quality_metrics(volume)

        assert metrics.has_nan is True

    def test_compute_quality_metrics_detects_inf(self):
        """Test quality metrics detects infinite values."""
        volume = np.random.randn(32, 32, 32)
        volume[5, 5, 5] = np.inf
        metrics = compute_quality_metrics(volume)

        assert metrics.has_inf is True

    def test_compute_quality_metrics_counts_zero_slices(self):
        """Test quality metrics counts zero slices."""
        volume = np.random.randn(64, 64, 64)
        volume[0, :, :] = 0
        volume[1, :, :] = 0
        metrics = compute_quality_metrics(volume)

        assert metrics.num_zero_slices == 2

    def test_compute_quality_metrics_isotropic_detection(self):
        """Test quality metrics detects isotropic spacing."""
        volume = np.random.randn(32, 32, 32)

        # Isotropic
        metrics_iso = compute_quality_metrics(volume, spacing=(1.0, 1.0, 1.0))
        assert metrics_iso.is_isotropic is True

        # Anisotropic
        metrics_aniso = compute_quality_metrics(volume, spacing=(2.5, 0.7, 0.7))
        assert metrics_aniso.is_isotropic is False

    def test_compute_quality_metrics_file_size_estimate(self):
        """Test quality metrics estimates file size."""
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        metrics = compute_quality_metrics(volume)

        expected_size_mb = (64 * 64 * 64 * 4) / (1024 * 1024)
        assert abs(metrics.estimated_file_size_mb - expected_size_mb) < 0.1


class TestSITKImageValidation:
    """Tests for SimpleITK image validation."""

    def test_validate_sitk_image_valid(self):
        """Test validation passes for valid SITK image."""
        volume = np.random.randn(64, 128, 128).astype(np.float32) * 100 - 500
        image = sitk.GetImageFromArray(volume)
        image.SetSpacing((1.0, 1.0, 2.5))
        image.SetOrigin((0, 0, 0))

        results = validate_sitk_image(image)
        errors = [r for r in results if not r.is_valid and r.severity == "error"]
        assert len(errors) == 0

    def test_validate_sitk_image_too_small(self):
        """Test validation fails for image smaller than minimum."""
        volume = np.random.randn(16, 16, 16).astype(np.float32)
        image = sitk.GetImageFromArray(volume)

        results = validate_sitk_image(image, min_size=(32, 32, 32))
        errors = [r for r in results if not r.is_valid and r.severity == "error"]
        assert len(errors) > 0

    def test_validate_sitk_image_spacing_conversion(self):
        """Test SITK spacing is correctly converted from x,y,z to z,y,x."""
        volume = np.random.randn(64, 64, 64).astype(np.float32)
        image = sitk.GetImageFromArray(volume)
        image.SetSpacing((0.7, 0.7, 2.5))  # x, y, z in ITK order

        results = validate_sitk_image(image)
        # Should detect anisotropic spacing (converted to z,y,x = 2.5, 0.7, 0.7)
        warnings = [r for r in results if r.severity == "warning"]
        assert any("anisotropic" in w.message.lower() for w in warnings)


class TestFileValidation:
    """Tests for file existence validation."""

    def test_validate_file_exists_valid(self, tmp_path):
        """Test validation passes for existing file."""
        test_file = tmp_path / "test.nii"
        test_file.write_text("dummy content")

        result = validate_file_exists(test_file)
        assert result.is_valid is True

    def test_validate_file_exists_not_found(self, tmp_path):
        """Test validation fails for non-existent file."""
        test_file = tmp_path / "nonexistent.nii"

        result = validate_file_exists(test_file)
        assert result.is_valid is False
        assert "not found" in result.message.lower()

    def test_validate_file_exists_is_directory(self, tmp_path):
        """Test validation fails for directory."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        result = validate_file_exists(test_dir)
        assert result.is_valid is False
        assert "not a file" in result.message.lower()

    def test_validate_file_exists_wrong_extension(self, tmp_path):
        """Test validation fails for wrong file extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("dummy")

        result = validate_file_exists(test_file, extensions=[".nii", ".nii.gz"])
        assert result.is_valid is False
        assert "extension" in result.message.lower()

    def test_validate_file_exists_correct_extension(self, tmp_path):
        """Test validation passes for correct file extension."""
        test_file = tmp_path / "test.nii.gz"
        test_file.write_text("dummy")

        result = validate_file_exists(test_file, extensions=[".nii", ".nii.gz"])
        assert result.is_valid is True


class TestDatasetStructureValidation:
    """Tests for dataset structure validation."""

    def test_validate_dataset_structure_valid(self, tmp_path):
        """Test validation passes for valid dataset structure."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "annotations.csv").write_text("dummy")

        results = validate_dataset_structure(
            tmp_path,
            required_subdirs=["images", "labels"],
            required_files=["annotations.csv"],
        )
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0

    def test_validate_dataset_structure_missing_directory(self, tmp_path):
        """Test validation fails for missing directory."""
        (tmp_path / "images").mkdir()
        # labels directory missing

        results = validate_dataset_structure(
            tmp_path,
            required_subdirs=["images", "labels"],
        )
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("labels" in e.message for e in errors)

    def test_validate_dataset_structure_missing_file(self, tmp_path):
        """Test validation fails for missing file."""
        (tmp_path / "images").mkdir()

        results = validate_dataset_structure(
            tmp_path,
            required_files=["annotations.csv"],
        )
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("annotations.csv" in e.message for e in errors)

    def test_validate_dataset_structure_nonexistent_root(self, tmp_path):
        """Test validation fails for non-existent root directory."""
        nonexistent = tmp_path / "does_not_exist"

        results = validate_dataset_structure(nonexistent)
        assert results[0].is_valid is False
        assert "not found" in results[0].message.lower()

    def test_validate_dataset_structure_root_is_file(self, tmp_path):
        """Test validation fails when root is a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("dummy")

        results = validate_dataset_structure(test_file)
        assert results[0].is_valid is False
        assert "not a directory" in results[0].message.lower()

    def test_validate_dataset_structure_required_subdir_is_file(self, tmp_path):
        """Test validation fails when required subdirectory is actually a file."""
        (tmp_path / "images").write_text("not a directory")

        results = validate_dataset_structure(
            tmp_path,
            required_subdirs=["images"],
        )
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("not a directory" in e.message.lower() for e in errors)

    def test_validate_dataset_structure_required_file_is_directory(self, tmp_path):
        """Test validation fails when required file is actually a directory."""
        (tmp_path / "annotations.csv").mkdir()

        results = validate_dataset_structure(
            tmp_path,
            required_files=["annotations.csv"],
        )
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
        assert any("not a file" in e.message.lower() for e in errors)


class TestValidationSummary:
    """Tests for validation result summarization."""

    def test_summarize_validation_results_all_pass(self):
        """Test summary when all validations pass."""
        results = [
            ValidationResult(True, "Check 1 passed", "info"),
            ValidationResult(True, "Check 2 passed", "info"),
            ValidationResult(True, "Check 3 passed", "info"),
        ]
        summary = summarize_validation_results(results)

        assert summary["total_checks"] == 3
        assert summary["passed"] == 3
        assert summary["failed"] == 0
        assert summary["warnings"] == 0
        assert summary["is_valid"] is True
        assert len(summary["errors"]) == 0

    def test_summarize_validation_results_with_errors(self):
        """Test summary when some validations fail."""
        results = [
            ValidationResult(True, "Check 1 passed", "info"),
            ValidationResult(False, "Check 2 failed", "error"),
            ValidationResult(False, "Check 3 failed", "error"),
        ]
        summary = summarize_validation_results(results)

        assert summary["total_checks"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["is_valid"] is False
        assert len(summary["errors"]) == 2
        assert "Check 2 failed" in summary["errors"]

    def test_summarize_validation_results_with_warnings(self):
        """Test summary includes warnings."""
        results = [
            ValidationResult(True, "Check 1 passed", "info"),
            ValidationResult(True, "Warning about something", "warning"),
            ValidationResult(True, "Another warning", "warning"),
        ]
        summary = summarize_validation_results(results)

        assert summary["warnings"] == 2
        assert summary["is_valid"] is True  # Warnings don't fail validation
        assert len(summary["warning_messages"]) == 2

    def test_summarize_validation_results_mixed(self):
        """Test summary with mixed results."""
        results = [
            ValidationResult(True, "Check 1 passed", "info"),
            ValidationResult(False, "Error occurred", "error"),
            ValidationResult(True, "Warning issued", "warning"),
            ValidationResult(True, "Check 2 passed", "info"),
        ]
        summary = summarize_validation_results(results)

        assert summary["total_checks"] == 4
        assert summary["passed"] == 3
        assert summary["failed"] == 1
        assert summary["warnings"] == 1
        assert summary["is_valid"] is False

    def test_summarize_validation_results_empty(self):
        """Test summary with no results."""
        results = []
        summary = summarize_validation_results(results)

        assert summary["total_checks"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["is_valid"] is True  # No errors = valid
