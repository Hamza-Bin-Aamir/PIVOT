"""Example usage of medical scan file loading and metadata extraction.

This script demonstrates how to use the MedicalScanLoader to load various
medical imaging formats (DICOM, NIfTI, MetaImage) for the LUNA16/LIDC-IDRI datasets.
"""

from pathlib import Path

from src.data.loader import DICOMLoader, MedicalScanLoader, NIfTILoader


def example_load_dicom_series():
    """Example: Load a DICOM series from a directory."""
    print("=" * 80)
    print("Example 1: Loading DICOM Series")
    print("=" * 80)

    # Path to directory containing DICOM files
    dicom_dir = Path("/path/to/dicom/series/")

    try:
        # Load DICOM series
        volume, metadata = DICOMLoader.load_dicom_series(dicom_dir)

        print(f"\nLoaded DICOM series from: {dicom_dir}")
        print(f"Volume shape: {volume.shape}")
        print(f"Data type: {volume.dtype}")
        print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}] HU")
        print("\nMetadata:")
        print(f"  Patient ID: {metadata.patient_id}")
        print(f"  Study Date: {metadata.study_date}")
        print(f"  Modality: {metadata.modality}")
        print(f"  Series Description: {metadata.series_description}")
        print(f"  Voxel Spacing (mm): {metadata.spacing}")
        print(f"  Origin (mm): {metadata.origin}")
        print(f"  Image Shape: {metadata.shape}")
        print(f"  Slice Thickness (mm): {metadata.slice_thickness}")
        print(f"  Pixel Spacing (mm): {metadata.pixel_spacing}")
        print(f"  Rescale Slope: {metadata.rescale_slope}")
        print(f"  Rescale Intercept: {metadata.rescale_intercept}")
        print(f"  Manufacturer: {metadata.manufacturer}")

    except Exception as e:
        print(f"Error loading DICOM series: {e}")


def example_load_single_dicom():
    """Example: Load a single DICOM file."""
    print("\n" + "=" * 80)
    print("Example 2: Loading Single DICOM File")
    print("=" * 80)

    dicom_file = Path("/path/to/single/file.dcm")

    try:
        array, metadata = DICOMLoader.load_dicom_file(dicom_file)

        print(f"\nLoaded DICOM file: {dicom_file.name}")
        print(f"Array shape: {array.shape}")
        print(f"Patient ID: {metadata.patient_id}")
        print(f"Spacing: {metadata.spacing}")

    except Exception as e:
        print(f"Error loading DICOM file: {e}")


def example_load_nifti():
    """Example: Load a NIfTI file."""
    print("\n" + "=" * 80)
    print("Example 3: Loading NIfTI File")
    print("=" * 80)

    nifti_file = Path("/path/to/scan.nii.gz")

    try:
        volume, metadata = NIfTILoader.load(nifti_file)

        print(f"\nLoaded NIfTI file: {nifti_file.name}")
        print(f"Volume shape: {volume.shape}")
        print(f"Voxel spacing (mm): {metadata.spacing}")
        print(f"Origin (mm): {metadata.origin}")
        print(f"Data type: {metadata.additional_info.get('data_type')}")

    except Exception as e:
        print(f"Error loading NIfTI file: {e}")


def example_auto_detect_format():
    """Example: Automatic format detection with MedicalScanLoader."""
    print("\n" + "=" * 80)
    print("Example 4: Automatic Format Detection")
    print("=" * 80)

    # List of different file formats
    files_to_load = [
        "/path/to/dicom/series/",  # DICOM directory
        "/path/to/scan.nii.gz",  # NIfTI compressed
        "/path/to/scan.nii",  # NIfTI
        "/path/to/scan.mhd",  # MetaImage
        "/path/to/single.dcm",  # Single DICOM
    ]

    for filepath in files_to_load:
        filepath = Path(filepath)

        try:
            print(f"\nLoading: {filepath}")
            array, metadata = MedicalScanLoader.load(filepath)

            print("  ✓ Successfully loaded")
            print(f"  Shape: {array.shape}")
            print(f"  Spacing: {metadata.spacing}")
            print(f"  Modality: {metadata.modality}")

        except FileNotFoundError:
            print("  ✗ File not found (expected for demo)")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_luna16_workflow():
    """Example: Complete LUNA16 data loading workflow."""
    print("\n" + "=" * 80)
    print("Example 5: LUNA16 Dataset Loading Workflow")
    print("=" * 80)

    # LUNA16 dataset structure
    luna16_root = Path("/path/to/LUNA16/")
    subset_dir = luna16_root / "subset0"

    # Example: Load a specific scan
    scan_file = subset_dir / "1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"

    try:
        print("\nLOADING LUNA16 SCAN")
        print(f"File: {scan_file.name}")

        # Load scan
        volume, metadata = MedicalScanLoader.load(scan_file)

        print("\nScan Information:")
        print(f"  Shape: {volume.shape}")
        print(f"  Spacing: {metadata.spacing} mm")
        print(f"  Origin: {metadata.origin} mm")
        print(f"  Value range: [{volume.min():.1f}, {volume.max():.1f}] HU")

        # Check if isotropic
        is_isotropic = len(set(metadata.spacing)) == 1
        print(f"  Isotropic: {is_isotropic}")

        if not is_isotropic:
            print("\n  Note: Scan needs resampling to isotropic resolution")
            print("  Recommended target spacing: (1.0, 1.0, 1.0) mm")

    except Exception as e:
        print(f"Error: {e}")


def example_lidc_idri_workflow():
    """Example: LIDC-IDRI data loading workflow."""
    print("\n" + "=" * 80)
    print("Example 6: LIDC-IDRI Dataset Loading Workflow")
    print("=" * 80)

    # LIDC-IDRI dataset structure (DICOM series)
    lidc_root = Path("/path/to/LIDC-IDRI/")
    patient_dir = (
        lidc_root
        / "LIDC-IDRI-0001"
        / "1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178"
    )
    series_dir = patient_dir / "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"

    try:
        print("\nLOADING LIDC-IDRI SCAN")
        print(f"Patient directory: {patient_dir.name}")

        # Load DICOM series
        volume, metadata = DICOMLoader.load_dicom_series(series_dir)

        print("\nScan Information:")
        print(f"  Patient ID: {metadata.patient_id}")
        print(f"  Study Date: {metadata.study_date}")
        print(f"  Series: {metadata.series_description}")
        print(f"  Shape: {volume.shape}")
        print(f"  Spacing: {metadata.spacing} mm")
        print(f"  Manufacturer: {metadata.manufacturer}")

        # Additional DICOM metadata
        if metadata.additional_info:
            print("\nAdditional Information:")
            print(f"  Study UID: {metadata.additional_info.get('study_instance_uid', 'N/A')}")
            print(f"  Series UID: {metadata.additional_info.get('series_instance_uid', 'N/A')}")
            if metadata.additional_info.get("kvp"):
                print(f"  KVP: {metadata.additional_info['kvp']} kV")

    except Exception as e:
        print(f"Error: {e}")


def example_metadata_analysis():
    """Example: Analyzing metadata from multiple scans."""
    print("\n" + "=" * 80)
    print("Example 7: Metadata Analysis")
    print("=" * 80)

    scan_files = [
        "/path/to/scan1.mhd",
        "/path/to/scan2.mhd",
        "/path/to/scan3.mhd",
    ]

    spacings = []
    shapes = []

    for scan_file in scan_files:
        try:
            _, metadata = MedicalScanLoader.load(Path(scan_file))
            spacings.append(metadata.spacing)
            shapes.append(metadata.shape)
        except Exception:
            continue

    if spacings:
        print(f"\nAnalyzed {len(spacings)} scans:")
        print(f"  Spacing variations: {set(spacings)}")
        print(f"  Shape variations: {set(shapes)}")

        # Check consistency
        unique_spacings = len(set(spacings))
        if unique_spacings == 1:
            print("  ✓ All scans have consistent spacing")
        else:
            print(f"  ⚠ Warning: {unique_spacings} different spacing values found")
            print("  → Resampling to isotropic resolution recommended")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MEDICAL SCAN LOADING EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate medical imaging file loading for LUNA16/LIDC-IDRI")
    print("datasets. Update the file paths to match your data location.")

    # Run examples
    example_load_dicom_series()
    example_load_single_dicom()
    example_load_nifti()
    example_auto_detect_format()
    example_luna16_workflow()
    example_lidc_idri_workflow()
    example_metadata_analysis()

    print("\n" + "=" * 80)
    print("QUICK START GUIDE")
    print("=" * 80)
    print("""
1. For LUNA16 (.mhd files):
   from src.data.loader import MedicalScanLoader
   volume, metadata = MedicalScanLoader.load("/path/to/scan.mhd")

2. For LIDC-IDRI (DICOM series):
   from src.data.loader import DICOMLoader
   volume, metadata = DICOMLoader.load_dicom_series("/path/to/dicom_dir/")

3. For any format (auto-detect):
   from src.data.loader import MedicalScanLoader
   volume, metadata = MedicalScanLoader.load("/path/to/scan")

4. Access metadata:
   print(f"Spacing: {metadata.spacing} mm")
   print(f"Patient: {metadata.patient_id}")
   print(f"Shape: {metadata.shape}")
    """)


if __name__ == "__main__":
    main()
