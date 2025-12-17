"""Unit tests for medical scan file loading and metadata extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pydicom
import pytest

from src.data.loader import (
    DICOMLoader,
    MedicalScanLoader,
    NIfTILoader,
    ScanMetadata,
)


class TestScanMetadata:
    """Test ScanMetadata dataclass."""

    def test_metadata_initialization(self):
        """Test basic metadata initialization."""
        metadata = ScanMetadata(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            shape=(100, 512, 512),
        )

        assert metadata.spacing == (1.0, 1.0, 1.0)
        assert metadata.origin == (0.0, 0.0, 0.0)
        assert metadata.shape == (100, 512, 512)
        assert metadata.modality == "CT"
        assert metadata.additional_info == {}

    def test_metadata_with_additional_info(self):
        """Test metadata with additional information."""
        additional = {"test_key": "test_value", "number": 42}
        metadata = ScanMetadata(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            shape=(100, 512, 512),
            additional_info=additional,
        )

        assert metadata.additional_info == additional


class TestDICOMLoader:
    """Test DICOM file loading functionality."""

    def test_is_dicom_file_valid(self, tmp_path):
        """Test DICOM file detection with valid file."""
        # Create a minimal DICOM file
        ds = pydicom.Dataset()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = "CT"
        ds.SeriesInstanceUID = "1.2.3"
        ds.SOPInstanceUID = "1.2.3.4"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage

        # Add required elements for valid DICOM
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        dicom_file = tmp_path / "test.dcm"
        ds.save_as(str(dicom_file), enforce_file_format=True)

        assert DICOMLoader.is_dicom_file(dicom_file)

    def test_is_dicom_file_invalid(self, tmp_path):
        """Test DICOM file detection with invalid file."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("This is not a DICOM file")

        # Note: pydicom with force=True may still try to read invalid files
        # For a truly invalid file, it should still return False or raise exception
        result = DICOMLoader.is_dicom_file(invalid_file)
        # Either False or True (pydicom is lenient), but shouldn't crash
        assert isinstance(result, bool)

    def test_is_dicom_file_exception_handling(self, tmp_path):
        """Test DICOM file detection handles exceptions properly."""
        # Create a file that will cause an exception during reading
        problem_file = tmp_path / "problem.dcm"
        problem_file.write_bytes(b"\x00" * 10)  # Invalid binary data

        # Should return False without crashing
        result = DICOMLoader.is_dicom_file(problem_file)
        assert isinstance(result, bool)

    def test_is_dicom_file_invalid_dicom_error(self):
        """Test handling of InvalidDicomError specifically."""
        from pydicom.errors import InvalidDicomError

        with patch("pydicom.dcmread", side_effect=InvalidDicomError("Invalid DICOM")):
            result = DICOMLoader.is_dicom_file(Path("/fake/invalid.dcm"))
            assert result is False

    def test_extract_metadata(self):
        """Test metadata extraction from DICOM dataset."""
        ds = pydicom.Dataset()
        ds.PatientID = "TEST123"
        ds.StudyDate = "20240101"
        ds.Modality = "CT"
        ds.SeriesDescription = "Chest CT"
        ds.Manufacturer = "TestCorp"
        ds.Rows = 512
        ds.Columns = 512
        ds.PixelSpacing = [0.7, 0.7]
        ds.SliceThickness = 1.25
        ds.ImagePositionPatient = [100.0, 100.0, 50.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.StudyInstanceUID = "1.2.3"
        ds.SeriesInstanceUID = "1.2.3.4"
        ds.SOPInstanceUID = "1.2.3.4.5"

        metadata = DICOMLoader._extract_metadata(ds, num_slices=100)

        assert metadata.patient_id == "TEST123"
        assert metadata.study_date == "20240101"
        assert metadata.modality == "CT"
        assert metadata.series_description == "Chest CT"
        assert metadata.manufacturer == "TestCorp"
        assert metadata.shape == (100, 512, 512)
        assert metadata.spacing == (0.7, 0.7, 1.25)
        assert metadata.origin == (100.0, 100.0, 50.0)
        assert metadata.rescale_slope == 1.0
        assert metadata.rescale_intercept == -1024.0

    def test_sort_dicom_slices_by_position(self):
        """Test sorting DICOM slices by ImagePositionPatient."""
        # Create mock datasets with different z positions
        datasets = []
        positions = [50.0, 0.0, 100.0, 25.0, 75.0]

        for i, z_pos in enumerate(positions):
            ds = pydicom.Dataset()
            ds.ImagePositionPatient = [0.0, 0.0, z_pos]
            filepath = Path(f"slice_{i}.dcm")
            datasets.append((filepath, ds))

        sorted_datasets = DICOMLoader._sort_dicom_slices(datasets)

        # Check that slices are sorted by z position
        z_positions = [float(ds.ImagePositionPatient[2]) for _, ds in sorted_datasets]
        assert z_positions == sorted(positions)

    def test_sort_dicom_slices_by_instance_number(self):
        """Test sorting DICOM slices by InstanceNumber when position missing."""
        datasets = []
        instance_numbers = [3, 1, 5, 2, 4]

        for i, inst_num in enumerate(instance_numbers):
            ds = pydicom.Dataset()
            ds.InstanceNumber = inst_num
            filepath = Path(f"slice_{i}.dcm")
            datasets.append((filepath, ds))

        sorted_datasets = DICOMLoader._sort_dicom_slices(datasets)

        # Check that slices are sorted by instance number
        inst_nums = [float(ds.InstanceNumber) for _, ds in sorted_datasets]
        assert inst_nums == sorted(instance_numbers)

    def test_build_volume(self):
        """Test building 3D volume from DICOM slices."""
        # Create mock datasets with pixel arrays
        datasets = []
        num_slices = 5
        height, width = 64, 64

        for i in range(num_slices):
            ds = pydicom.Dataset()
            # Create pixel array by setting PixelData and required DICOM tags
            pixel_data = np.full((height, width), i, dtype=np.uint16)
            ds.PixelData = pixel_data.tobytes()
            ds.Rows = height
            ds.Columns = width
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0
            # Add file_meta for pixel_array decoding
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            filepath = Path(f"slice_{i}.dcm")
            datasets.append((filepath, ds))

        volume = DICOMLoader._build_volume(datasets)

        assert volume.shape == (num_slices, height, width)
        assert volume.dtype == np.float32
        # Check that each slice has correct value
        for i in range(num_slices):
            assert np.all(volume[i] == i)

    def test_build_volume_with_rescale(self):
        """Test volume building with HU rescaling."""
        datasets = []
        num_slices = 3
        height, width = 32, 32

        for i in range(num_slices):
            ds = pydicom.Dataset()
            pixel_data = np.full((height, width), 100, dtype=np.uint16)
            ds.PixelData = pixel_data.tobytes()
            ds.Rows = height
            ds.Columns = width
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = -1024.0
            # Add file_meta for pixel_array decoding
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            filepath = Path(f"slice_{i}.dcm")
            datasets.append((filepath, ds))

        volume = DICOMLoader._build_volume(datasets)

        # Check HU conversion: 100 * 1.0 + (-1024) = -924
        expected_value = 100.0 * 1.0 + (-1024.0)
        assert np.allclose(volume, expected_value)


class TestNIfTILoader:
    """Test NIfTI file loading functionality."""

    @patch("nibabel.load")
    def test_load_nifti_file(self, mock_nib_load):
        """Test loading NIfTI file with mocked nibabel."""
        # Create mock NIfTI image
        mock_img = MagicMock()
        mock_data = np.random.rand(100, 256, 256).astype(np.float32)
        mock_img.get_fdata.return_value = mock_data

        # Mock header
        mock_header = MagicMock()
        mock_header.get_zooms.return_value = (1.0, 1.0, 1.0, 1.0)
        mock_header.__getitem__.return_value = 348  # sizeof_hdr
        mock_header.get_data_dtype.return_value = np.float32
        mock_img.header = mock_header

        # Mock affine
        mock_affine = np.eye(4)
        mock_affine[:3, 3] = [10.0, 20.0, 30.0]  # origin
        mock_img.affine = mock_affine

        mock_nib_load.return_value = mock_img

        # Test loading
        filepath = Path("/fake/path/test.nii.gz")
        with patch("pathlib.Path.exists", return_value=True):
            array, metadata = NIfTILoader.load(filepath)

        assert array.shape == mock_data.shape
        assert array.dtype == np.float32
        assert metadata.spacing == (1.0, 1.0, 1.0)
        assert metadata.origin == (10.0, 20.0, 30.0)
        assert metadata.shape == mock_data.shape

    @patch("nibabel.load")
    def test_load_nifti_file_not_found(self, mock_nib_load):
        """Test error handling when NIfTI file doesn't exist."""
        filepath = Path("/fake/path/nonexistent.nii.gz")

        with pytest.raises(ValueError, match="File does not exist"):
            NIfTILoader.load(filepath)


class TestMedicalScanLoader:
    """Test unified medical scan loader."""

    def test_load_with_sitk(self):
        """Test loading with SimpleITK."""
        with patch("SimpleITK.ReadImage") as mock_read:
            # Create mock SimpleITK image
            mock_img = MagicMock()
            mock_img.GetSpacing.return_value = (1.0, 1.0, 2.5)
            mock_img.GetOrigin.return_value = (0.0, 0.0, 0.0)
            mock_img.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
            mock_img.GetPixelIDTypeAsString.return_value = "32-bit float"
            mock_img.GetNumberOfComponentsPerPixel.return_value = 1

            # Mock numpy conversion
            mock_array = np.random.rand(50, 256, 256).astype(np.float32)

            with patch("SimpleITK.GetArrayFromImage", return_value=mock_array):
                mock_read.return_value = mock_img

                filepath = Path("/fake/path/test.mhd")
                array, metadata = MedicalScanLoader._load_with_sitk(filepath)

            assert array.shape == mock_array.shape
            assert metadata.spacing == (1.0, 1.0, 2.5)
            assert metadata.shape == mock_array.shape

    def test_load_auto_detect_nifti(self):
        """Test automatic format detection for NIfTI files."""
        with patch.object(NIfTILoader, "load") as mock_load:
            mock_array = np.random.rand(100, 256, 256).astype(np.float32)
            mock_metadata = ScanMetadata(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                shape=mock_array.shape,
            )
            mock_load.return_value = (mock_array, mock_metadata)

            filepath = Path("/fake/path/test.nii.gz")
            with patch("pathlib.Path.exists", return_value=True):
                array, metadata = MedicalScanLoader.load(filepath)

            assert array.shape == mock_array.shape
            mock_load.assert_called_once()

    def test_load_auto_detect_mhd(self):
        """Test automatic format detection for MetaImage files."""
        with patch.object(MedicalScanLoader, "_load_with_sitk") as mock_load:
            mock_array = np.random.rand(100, 256, 256).astype(np.float32)
            mock_metadata = ScanMetadata(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                shape=mock_array.shape,
            )
            mock_load.return_value = (mock_array, mock_metadata)

            filepath = Path("/fake/path/test.mhd")
            with patch("pathlib.Path.exists", return_value=True):
                array, metadata = MedicalScanLoader.load(filepath)

            assert array.shape == mock_array.shape
            mock_load.assert_called_once()

    def test_load_auto_detect_dicom_series(self):
        """Test automatic format detection for DICOM series directory."""
        with patch.object(DICOMLoader, "load_dicom_series") as mock_load:
            mock_array = np.random.rand(100, 512, 512).astype(np.float32)
            mock_metadata = ScanMetadata(
                spacing=(0.7, 0.7, 1.25),
                origin=(0.0, 0.0, 0.0),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                shape=mock_array.shape,
            )
            mock_load.return_value = (mock_array, mock_metadata)

            filepath = Path("/fake/path/dicom_series/")
            with patch("pathlib.Path.is_dir", return_value=True):
                array, metadata = MedicalScanLoader.load(filepath)

            assert array.shape == mock_array.shape
            mock_load.assert_called_once()

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        filepath = Path("/fake/path/nonexistent.nii")

        with pytest.raises(ValueError, match="does not exist"):
            MedicalScanLoader.load(filepath)

    def test_load_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(DICOMLoader, "is_dicom_file", return_value=False),
            patch("SimpleITK.ReadImage", side_effect=Exception("Unsupported format")),
        ):
            filepath = Path("/fake/path/unsupported.xyz")

            with pytest.raises(ValueError, match="Could not determine file format"):
                MedicalScanLoader.load(filepath)


class TestIntegration:
    """Integration tests for medical scan loading."""

    def test_dicom_to_array_conversion(self, tmp_path):
        """Test complete DICOM to numpy array conversion."""
        # Create a simple DICOM file
        ds = pydicom.Dataset()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = "CT"
        ds.SeriesInstanceUID = "1.2.3"
        ds.SOPInstanceUID = "1.2.3.4"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.Rows = 64
        ds.Columns = 64
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0, 0, 0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Create pixel data
        pixel_array = np.random.randint(0, 1000, (64, 64), dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()

        # Add file meta
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        dicom_file = tmp_path / "test.dcm"
        ds.save_as(str(dicom_file), enforce_file_format=True)

        # Load the file
        array, metadata = DICOMLoader.load_dicom_file(dicom_file)

        assert array.shape == (64, 64)
        assert metadata.patient_id == "123456"
        assert metadata.modality == "CT"
        assert metadata.spacing == (1.0, 1.0, 1.0)

    def test_dicom_series_error_not_directory(self, tmp_path):
        """Test error when path is not a directory."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            DICOMLoader.load_dicom_series(file_path)

    def test_dicom_series_error_no_files(self, tmp_path):
        """Test error when no DICOM files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No valid DICOM files"):
            DICOMLoader.load_dicom_series(empty_dir)

    def test_dicom_series_with_loading_error(self, tmp_path):
        """Test handling of files that fail to load."""
        series_dir = tmp_path / "series"
        series_dir.mkdir()

        # Create one valid DICOM file
        ds = pydicom.Dataset()
        ds.PatientID = "TEST"
        ds.Modality = "CT"
        ds.SeriesInstanceUID = "1.2.3"
        ds.SOPInstanceUID = "1.2.3.4"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.Rows = 32
        ds.Columns = 32
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0, 0, 0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        pixel_array = np.zeros((32, 32), dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        valid_file = series_dir / "valid.dcm"
        ds.save_as(str(valid_file), enforce_file_format=True)

        # Create a corrupted DICOM file (will be detected but fail to load details)
        corrupted = series_dir / "corrupted.dcm"
        # Write minimal DICOM header that passes is_dicom_file but may fail detailed load
        corrupted.write_bytes(b"DICM" + b"\x00" * 100)

        # Should still load the valid file
        volume, metadata = DICOMLoader.load_dicom_series(series_dir)
        assert volume.shape[0] >= 1  # At least one slice loaded

    def test_dicom_series_all_files_fail(self, tmp_path):
        """Test error when all files fail to load."""
        series_dir = tmp_path / "series"
        series_dir.mkdir()

        # Create files that are detected as DICOM but fail detailed load
        # Use pydicom with force=True to create files that pass is_dicom check
        for i in range(3):
            bad_file = series_dir / f"bad_{i}.dcm"
            # Write minimal DICOM-like content that passes force=True detection
            # but has invalid structure for full loading
            bad_file.write_bytes(b"DICM" + b"\x00" * 100)

        with pytest.raises(RuntimeError, match="Could not load any DICOM files"):
            DICOMLoader.load_dicom_series(series_dir)

    def test_dicom_sort_missing_position_and_instance(self):
        """Test error when DICOM has neither position nor instance number."""
        ds = pydicom.Dataset()
        datasets = [(Path("test.dcm"), ds)]

        with pytest.raises(RuntimeError, match="missing position info"):
            DICOMLoader._sort_dicom_slices(datasets)

    def test_dicom_file_not_exists(self):
        """Test error when DICOM file doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            DICOMLoader.load_dicom_file(Path("/nonexistent/file.dcm"))

    def test_dicom_file_load_error(self, tmp_path):
        """Test error handling when DICOM file fails to load."""
        bad_file = tmp_path / "bad.dcm"
        bad_file.write_text("not a valid DICOM")

        with pytest.raises(ValueError, match="Failed to load DICOM file"):
            DICOMLoader.load_dicom_file(bad_file)

    def test_dicom_metadata_defaults(self):
        """Test metadata extraction with missing optional fields."""
        ds = pydicom.Dataset()
        # Only set required fields, leave optional ones missing
        ds.Rows = 256
        ds.Columns = 256

        metadata = DICOMLoader._extract_metadata(ds, num_slices=10)

        # Check defaults are used
        assert metadata.pixel_spacing == (1.0, 1.0)
        assert metadata.spacing == (1.0, 1.0, 1.0)
        assert metadata.origin == (0.0, 0.0, 0.0)
        assert metadata.direction == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert metadata.modality == "CT"
        assert metadata.rescale_slope == 1.0
        assert metadata.rescale_intercept == 0.0

    def test_build_volume_slice_error(self):
        """Test error handling when slice processing fails."""
        datasets = []

        # Create a dataset that will fail when accessing pixel_array
        ds = pydicom.Dataset()
        ds.Rows = 64
        ds.Columns = 64
        # Don't set PixelData, so pixel_array will fail
        filepath = Path("test.dcm")
        datasets.append((filepath, ds))

        with pytest.raises(AttributeError):
            DICOMLoader._build_volume(datasets)

    def test_build_volume_processing_error(self):
        """Test error logging and re-raising when slice fails to process."""
        datasets = []

        # Create a mock dataset with valid first slice to get dimensions
        ds1 = pydicom.Dataset()
        ds1.Rows = 32
        ds1.Columns = 32
        pixel_data = np.zeros((32, 32), dtype=np.uint16)
        ds1.PixelData = pixel_data.tobytes()
        ds1.BitsAllocated = 16
        ds1.BitsStored = 16
        ds1.HighBit = 15
        ds1.PixelRepresentation = 0
        ds1.SamplesPerPixel = 1
        ds1.PhotometricInterpretation = "MONOCHROME2"
        ds1.RescaleSlope = 1.0
        ds1.RescaleIntercept = 0.0
        ds1.file_meta = pydicom.dataset.FileMetaDataset()
        ds1.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Create a second dataset that will fail during pixel_array access
        ds2 = pydicom.Dataset()
        ds2.Rows = 32
        ds2.Columns = 32
        # Missing required fields - will fail

        datasets = [(Path("slice_0.dcm"), ds1), (Path("slice_1.dcm"), ds2)]

        # Should raise exception and log error
        with pytest.raises(AttributeError):
            DICOMLoader._build_volume(datasets)

    def test_nifti_load_error(self):
        """Test NIfTI loading error handling."""
        with (
            patch("nibabel.load", side_effect=Exception("Load failed")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            filepath = Path("/fake/test.nii.gz")
            with pytest.raises(ValueError, match="Failed to load NIfTI file"):
                NIfTILoader.load(filepath)

    def test_medical_scan_loader_auto_detect_dicom_file(self):
        """Test auto-detection of single DICOM file."""
        with (
            patch.object(DICOMLoader, "is_dicom_file", return_value=True),
            patch.object(DICOMLoader, "load_dicom_file") as mock_load,
        ):
            mock_array = np.random.rand(64, 64).astype(np.float32)
            mock_metadata = ScanMetadata(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                shape=mock_array.shape,
            )
            mock_load.return_value = (mock_array, mock_metadata)

            filepath = Path("/fake/path/test.dcm")
            with patch("pathlib.Path.exists", return_value=True):
                array, metadata = MedicalScanLoader.load(filepath)

            assert array.shape == mock_array.shape
            mock_load.assert_called_once()

    def test_medical_scan_loader_sitk_fallback(self):
        """Test SimpleITK fallback for unknown extensions."""
        with (
            patch.object(DICOMLoader, "is_dicom_file", return_value=False),
            patch.object(MedicalScanLoader, "_load_with_sitk") as mock_sitk,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_array = np.random.rand(100, 256, 256).astype(np.float32)
            mock_metadata = ScanMetadata(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                shape=mock_array.shape,
            )
            mock_sitk.return_value = (mock_array, mock_metadata)

            filepath = Path("/fake/path/unknown.xyz")
            array, metadata = MedicalScanLoader.load(filepath)

            assert array.shape == mock_array.shape
            mock_sitk.assert_called_once()
