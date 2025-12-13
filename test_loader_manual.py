#!/usr/bin/env python3
"""Manual test script for medical scan file loading.

Run this script to test the loader with your own medical imaging files.
"""

import sys
from pathlib import Path

from src.data.loader import MedicalScanLoader


def test_file(filepath: str):
    """Test loading a single file or directory."""
    filepath = Path(filepath)

    print("=" * 80)
    print(f"Testing file loader with: {filepath}")
    print("=" * 80)

    if not filepath.exists():
        print(f"❌ Error: File or directory does not exist: {filepath}")
        return False

    try:
        # Load the file
        print(f"\n📂 Loading: {filepath.name if filepath.is_file() else filepath}")
        volume, metadata = MedicalScanLoader.load(filepath)

        # Display results
        print("\n✅ Successfully loaded!")
        print("\n📊 Volume Information:")
        print(f"  Shape: {volume.shape}")
        print(f"  Data type: {volume.dtype}")
        print(f"  Value range: [{volume.min():.2f}, {volume.max():.2f}]")
        print(f"  Memory size: {volume.nbytes / 1024 / 1024:.2f} MB")

        print("\n📋 Metadata:")
        print(f"  Modality: {metadata.modality}")
        print(f"  Voxel Spacing (mm): {metadata.spacing}")
        print(f"  Origin (mm): {metadata.origin}")
        print(f"  Image Shape: {metadata.shape}")

        if metadata.patient_id:
            print(f"  Patient ID: {metadata.patient_id}")
        if metadata.study_date:
            print(f"  Study Date: {metadata.study_date}")
        if metadata.series_description:
            print(f"  Series: {metadata.series_description}")
        if metadata.manufacturer:
            print(f"  Manufacturer: {metadata.manufacturer}")

        print("\n🔍 Additional Details:")
        print(f"  Slice Thickness: {metadata.slice_thickness} mm")
        print(f"  Pixel Spacing: {metadata.pixel_spacing} mm")
        print(f"  Rescale Slope: {metadata.rescale_slope}")
        print(f"  Rescale Intercept: {metadata.rescale_intercept}")

        # Check if isotropic
        is_isotropic = len(set(metadata.spacing)) == 1
        print("\n📐 Spatial Properties:")
        print(f"  Isotropic: {'✓ Yes' if is_isotropic else '✗ No'}")
        if not is_isotropic:
            print("  ⚠️  Note: Scan needs resampling to isotropic resolution")
            print("  💡 Recommended target: (1.0, 1.0, 1.0) mm")

        # Show additional metadata if available
        if metadata.additional_info:
            print("\n🔧 Additional Metadata:")
            for key, value in metadata.additional_info.items():
                if value is not None:
                    print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_usage():
    """Display usage instructions."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           Medical Scan File Loader - Manual Test Script                   ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python test_loader_manual.py <path_to_file_or_directory>

EXAMPLES:
    # Test a single DICOM file
    python test_loader_manual.py /path/to/scan.dcm

    # Test a DICOM series directory
    python test_loader_manual.py /path/to/dicom_series/

    # Test a NIfTI file
    python test_loader_manual.py /path/to/scan.nii.gz

    # Test a MetaImage file (LUNA16 format)
    python test_loader_manual.py /path/to/scan.mhd

SUPPORTED FORMATS:
    • DICOM (.dcm, directories)
    • NIfTI (.nii, .nii.gz)
    • MetaImage (.mhd, .mha)
    • Any format supported by SimpleITK

LUNA16 DATASET EXAMPLE:
    python test_loader_manual.py subset0/1.3.6.1.4.1.14519.5.2.1.*.mhd

LIDC-IDRI DATASET EXAMPLE:
    python test_loader_manual.py LIDC-IDRI-0001/*/series_directory/

""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)

    filepath = sys.argv[1]

    # Test the file
    success = test_file(filepath)

    print("\n" + "=" * 80)
    if success:
        print("✅ Test completed successfully!")
        print("\n💡 Next steps:")
        print("   - Check that spacing matches expectations")
        print("   - Verify HU value ranges for CT scans")
        print("   - Confirm orientation is correct")
        print("   - Use this data in your preprocessing pipeline")
    else:
        print("❌ Test failed - see error above")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
