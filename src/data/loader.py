"""Medical scan file loading and metadata extraction.

This module provides functionality for loading various medical imaging formats
including DICOM, NIfTI, and MetaImage files, with comprehensive metadata extraction
for the LUNA16 and LIDC-IDRI datasets.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


@dataclass
class ScanMetadata:
    """Metadata extracted from medical scan files.

    Attributes:
        spacing: Voxel spacing in mm (x, y, z)
        origin: Physical coordinates of the first voxel (x, y, z)
        direction: Orientation matrix (3x3)
        shape: Array dimensions (depth, height, width)
        modality: Imaging modality (e.g., 'CT', 'MR')
        patient_id: Patient identifier
        study_date: Study acquisition date
        series_description: Series description
        slice_thickness: Slice thickness in mm
        pixel_spacing: In-plane pixel spacing in mm (row, col)
        rescale_slope: Rescale slope for HU conversion
        rescale_intercept: Rescale intercept for HU conversion
        manufacturer: Scanner manufacturer
        additional_info: Additional metadata dictionary
    """

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[tuple[float, ...], ...]
    shape: tuple[int, int, int]
    modality: str = "CT"
    patient_id: str = ""
    study_date: str = ""
    series_description: str = ""
    slice_thickness: float = 0.0
    pixel_spacing: tuple[float, float] = (0.0, 0.0)
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    manufacturer: str = ""
    additional_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize additional_info if not provided."""
        if self.additional_info is None:
            self.additional_info = {}


class DICOMLoader:
    """Load and process DICOM files.

    Handles single DICOM files or directories containing DICOM series.
    Supports multi-slice CT scans commonly found in LUNA16/LIDC-IDRI datasets.
    """

    @staticmethod
    def is_dicom_file(filepath: Path) -> bool:
        """Check if file is a valid DICOM file.

        Args:
            filepath: Path to file to check

        Returns:
            True if file is valid DICOM, False otherwise
        """
        try:
            pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)
            return True
        except (InvalidDicomError, Exception):
            return False

    @staticmethod
    def load_dicom_series(directory: Path) -> tuple[np.ndarray, ScanMetadata]:
        """Load a DICOM series from a directory.

        Args:
            directory: Directory containing DICOM files

        Returns:
            Tuple of (3D numpy array, metadata)

        Raises:
            ValueError: If directory contains no valid DICOM files
            RuntimeError: If DICOM files cannot be loaded or sorted
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Find all DICOM files
        dicom_files = [
            f for f in directory.iterdir() if f.is_file() and DICOMLoader.is_dicom_file(f)
        ]

        if not dicom_files:
            raise ValueError(f"No valid DICOM files found in {directory}")

        logger.info(f"Found {len(dicom_files)} DICOM files in {directory}")

        # Load all DICOM files
        dicom_datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(str(filepath))
                dicom_datasets.append((filepath, ds))
            except Exception as e:
                logger.warning(f"Failed to load DICOM file {filepath}: {e}")

        if not dicom_datasets:
            raise RuntimeError(f"Could not load any DICOM files from {directory}")

        # Sort by ImagePositionPatient (z-coordinate)
        sorted_datasets = DICOMLoader._sort_dicom_slices(dicom_datasets)  # type: ignore[arg-type]

        # Extract metadata from first slice
        metadata = DICOMLoader._extract_metadata(sorted_datasets[0][1], len(sorted_datasets))

        # Build 3D volume
        volume = DICOMLoader._build_volume(sorted_datasets)

        logger.info(f"Loaded DICOM series: shape={volume.shape}, spacing={metadata.spacing}")

        return volume, metadata

    @staticmethod
    def load_dicom_file(filepath: Path) -> tuple[np.ndarray, ScanMetadata]:
        """Load a single DICOM file.

        Args:
            filepath: Path to DICOM file

        Returns:
            Tuple of (2D or 3D numpy array, metadata)

        Raises:
            ValueError: If file is not valid DICOM
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise ValueError(f"File does not exist: {filepath}")

        try:
            ds = pydicom.dcmread(str(filepath))
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file {filepath}: {e}") from e

        # Extract pixel array
        array = ds.pixel_array.astype(np.float32)

        # Apply rescale slope and intercept for HU values
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            array = array * ds.RescaleSlope + ds.RescaleIntercept

        # Extract metadata
        metadata = DICOMLoader._extract_metadata(ds, 1 if array.ndim == 2 else array.shape[0])

        return array, metadata

    @staticmethod
    def _sort_dicom_slices(
        dicom_datasets: list[tuple[Path, Dataset]],
    ) -> list[tuple[Path, Dataset]]:
        """Sort DICOM slices by ImagePositionPatient z-coordinate.

        Args:
            dicom_datasets: List of (filepath, dataset) tuples

        Returns:
            Sorted list of (filepath, dataset) tuples

        Raises:
            RuntimeError: If ImagePositionPatient is missing
        """
        slices_with_position = []

        for filepath, ds in dicom_datasets:
            if not hasattr(ds, "ImagePositionPatient"):
                logger.warning(f"DICOM file {filepath} missing ImagePositionPatient")
                # Try using InstanceNumber as fallback
                if hasattr(ds, "InstanceNumber"):
                    slices_with_position.append((filepath, ds, float(ds.InstanceNumber)))
                else:
                    raise RuntimeError(
                        f"Cannot sort DICOM slices: {filepath} missing position info"
                    )
            else:
                # Use z-coordinate (third element) of ImagePositionPatient
                z_pos = float(ds.ImagePositionPatient[2])
                slices_with_position.append((filepath, ds, z_pos))

        # Sort by position
        slices_with_position.sort(key=lambda x: x[2])

        # Return without position value
        return [(filepath, ds) for filepath, ds, _ in slices_with_position]

    @staticmethod
    def _extract_metadata(ds: Dataset, num_slices: int) -> ScanMetadata:
        """Extract metadata from DICOM dataset.

        Args:
            ds: pydicom Dataset
            num_slices: Number of slices in series

        Returns:
            ScanMetadata object
        """
        # Get pixel spacing (in-plane)
        pixel_spacing = (
            tuple(map(float, ds.PixelSpacing)) if hasattr(ds, "PixelSpacing") else (1.0, 1.0)
        )

        # Get slice thickness
        slice_thickness = float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else 1.0

        # Build 3D spacing (x, y, z)
        spacing = (pixel_spacing[1], pixel_spacing[0], slice_thickness)

        # Get image position (origin)
        origin = (
            tuple(map(float, ds.ImagePositionPatient))
            if hasattr(ds, "ImagePositionPatient")
            else (0.0, 0.0, 0.0)
        )

        # Get orientation (direction cosines)
        if hasattr(ds, "ImageOrientationPatient"):
            orientation = list(map(float, ds.ImageOrientationPatient))
            # Build 3x3 direction matrix
            row_cosine = np.array(orientation[:3])
            col_cosine = np.array(orientation[3:6])
            slice_cosine = np.cross(row_cosine, col_cosine)
            direction_matrix = np.column_stack([row_cosine, col_cosine, slice_cosine])
            direction = tuple(tuple(row) for row in direction_matrix)
        else:
            # Identity matrix
            direction = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        # Get image shape
        rows = ds.Rows if hasattr(ds, "Rows") else 512
        cols = ds.Columns if hasattr(ds, "Columns") else 512
        shape = (num_slices, rows, cols)

        # Extract other metadata
        modality = ds.Modality if hasattr(ds, "Modality") else "CT"
        patient_id = ds.PatientID if hasattr(ds, "PatientID") else ""
        study_date = ds.StudyDate if hasattr(ds, "StudyDate") else ""
        series_desc = ds.SeriesDescription if hasattr(ds, "SeriesDescription") else ""
        manufacturer = ds.Manufacturer if hasattr(ds, "Manufacturer") else ""

        rescale_slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        rescale_intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0

        # Additional information
        additional_info = {
            "study_instance_uid": str(ds.StudyInstanceUID)
            if hasattr(ds, "StudyInstanceUID")
            else "",
            "series_instance_uid": str(ds.SeriesInstanceUID)
            if hasattr(ds, "SeriesInstanceUID")
            else "",
            "sop_instance_uid": str(ds.SOPInstanceUID) if hasattr(ds, "SOPInstanceUID") else "",
            "acquisition_date": str(ds.AcquisitionDate) if hasattr(ds, "AcquisitionDate") else "",
            "kvp": float(ds.KVP) if hasattr(ds, "KVP") else None,
            "exposure": float(ds.Exposure) if hasattr(ds, "Exposure") else None,
        }

        return ScanMetadata(
            spacing=spacing,  # type: ignore[arg-type]
            origin=origin,  # type: ignore[arg-type]
            direction=direction,
            shape=shape,
            modality=modality,
            patient_id=patient_id,
            study_date=study_date,
            series_description=series_desc,
            slice_thickness=slice_thickness,
            pixel_spacing=pixel_spacing,  # type: ignore[arg-type]
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            manufacturer=manufacturer,
            additional_info=additional_info,
        )

    @staticmethod
    def _build_volume(sorted_datasets: list[tuple[Path, Dataset]]) -> np.ndarray:
        """Build 3D volume from sorted DICOM slices.

        Args:
            sorted_datasets: Sorted list of (filepath, dataset) tuples

        Returns:
            3D numpy array with shape (depth, height, width)
        """
        # Get first slice to determine dimensions
        first_array = sorted_datasets[0][1].pixel_array
        depth = len(sorted_datasets)
        height, width = first_array.shape

        # Initialize volume
        volume = np.zeros((depth, height, width), dtype=np.float32)

        # Fill volume slice by slice
        for i, (filepath, ds) in enumerate(sorted_datasets):
            try:
                array = ds.pixel_array.astype(np.float32)

                # Apply rescale for HU values
                if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                    array = array * ds.RescaleSlope + ds.RescaleIntercept

                volume[i] = array
            except Exception as e:
                logger.error(f"Failed to process slice {i} from {filepath}: {e}")
                raise

        return volume


class NIfTILoader:
    """Load NIfTI format medical images.

    Supports .nii and .nii.gz files commonly used in medical imaging research.
    """

    @staticmethod
    def load(filepath: Path) -> tuple[np.ndarray, ScanMetadata]:
        """Load NIfTI file.

        Args:
            filepath: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            Tuple of (3D numpy array, metadata)

        Raises:
            ValueError: If file cannot be loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"File does not exist: {filepath}")

        try:
            nii_img = nib.load(str(filepath))
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI file {filepath}: {e}") from e

        # Get image data
        array = np.array(nii_img.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]

        # Extract metadata from header
        header = nii_img.header
        affine = nii_img.affine  # type: ignore[attr-defined]

        # Get voxel spacing
        spacing = tuple(float(x) for x in header.get_zooms()[:3])  # type: ignore[attr-defined]

        # Get origin from affine matrix
        origin = tuple(float(x) for x in affine[:3, 3])

        # Get direction from affine matrix
        # Normalize the direction vectors
        direction_matrix = affine[:3, :3] / np.array(spacing)
        direction = tuple(tuple(float(x) for x in row) for row in direction_matrix)

        # Get shape
        shape = array.shape[:3]

        metadata = ScanMetadata(
            spacing=spacing,  # type: ignore[arg-type]
            origin=origin,  # type: ignore[arg-type]
            direction=direction,
            shape=shape,
            modality="CT",  # Default, can be overridden
            additional_info={
                "nifti_version": str(header["sizeof_hdr"]),  # type: ignore[index]
                "data_type": str(header.get_data_dtype()),  # type: ignore[attr-defined]
            },
        )

        logger.info(f"Loaded NIfTI file: shape={array.shape}, spacing={spacing}")

        return array, metadata


class MedicalScanLoader:
    """Unified interface for loading medical scan files.

    Automatically detects file format (DICOM, NIfTI, MetaImage) and loads
    the appropriate data with metadata extraction.
    """

    @staticmethod
    def load(filepath: Path | str) -> tuple[np.ndarray, ScanMetadata]:
        """Load medical scan file with automatic format detection.

        Args:
            filepath: Path to file or directory (for DICOM series)

        Returns:
            Tuple of (numpy array, metadata)

        Raises:
            ValueError: If file format is not supported or cannot be loaded
        """
        filepath = Path(filepath)

        # Handle directory (DICOM series)
        if filepath.is_dir():
            logger.info(f"Loading DICOM series from directory: {filepath}")
            return DICOMLoader.load_dicom_series(filepath)

        # Handle single file
        if not filepath.exists():
            raise ValueError(f"File or directory does not exist: {filepath}")

        # Check file extension
        suffix = filepath.suffix.lower()
        stem = filepath.stem.lower()

        # NIfTI files
        if suffix in [".nii", ".gz"] or stem.endswith(".nii"):
            logger.info(f"Loading NIfTI file: {filepath}")
            return NIfTILoader.load(filepath)

        # MetaImage files (.mhd, .mha)
        if suffix in [".mhd", ".mha"]:
            logger.info(f"Loading MetaImage file with SimpleITK: {filepath}")
            return MedicalScanLoader._load_with_sitk(filepath)

        # Try DICOM
        if DICOMLoader.is_dicom_file(filepath):
            logger.info(f"Loading DICOM file: {filepath}")
            return DICOMLoader.load_dicom_file(filepath)

        # Try SimpleITK as fallback
        try:
            logger.info(f"Attempting to load with SimpleITK: {filepath}")
            return MedicalScanLoader._load_with_sitk(filepath)
        except Exception as e:
            raise ValueError(
                f"Could not determine file format or load {filepath}. "
                f"Supported formats: DICOM, NIfTI (.nii, .nii.gz), MetaImage (.mhd, .mha). "
                f"Error: {e}"
            ) from e

    @staticmethod
    def _load_with_sitk(filepath: Path) -> tuple[np.ndarray, ScanMetadata]:
        """Load medical image using SimpleITK.

        Args:
            filepath: Path to image file

        Returns:
            Tuple of (numpy array, metadata)
        """
        image = sitk.ReadImage(str(filepath))

        # Convert to numpy array
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # Extract metadata
        spacing = image.GetSpacing()  # (x, y, z)
        origin = image.GetOrigin()  # (x, y, z)
        direction_flat = image.GetDirection()  # Flattened 3x3 matrix

        # Reshape direction to 3x3
        direction_matrix = np.array(direction_flat).reshape(3, 3)
        direction = tuple(tuple(float(x) for x in row) for row in direction_matrix)

        shape = array.shape

        metadata = ScanMetadata(
            spacing=spacing,
            origin=origin,
            direction=direction,
            shape=shape,
            modality="CT",
            additional_info={
                "pixel_type": str(image.GetPixelIDTypeAsString()),
                "number_of_components": image.GetNumberOfComponentsPerPixel(),
            },
        )

        logger.info(f"Loaded with SimpleITK: shape={array.shape}, spacing={spacing}")

        return array, metadata
