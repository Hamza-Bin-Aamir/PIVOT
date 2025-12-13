# Annotation Parsing Utilities

The `src/data/annotations.py` module centralises the tooling for transforming the
LUNA16 CSV metadata and the richer LIDC-IDRI XML reports into strongly-typed,
Python-friendly structures. This page explains the available helpers, their
expected inputs, and how they fit into the broader data pipeline.

## LUNA16 CSV Metadata

The official LUNA16 release provides a single `annotations.csv` file containing
per-nodule measurements. Use `parse_luna16_annotations` to read the file into a
list of immutable `LUNA16Annotation` instances:

```python
from pathlib import Path
from src.data.annotations import parse_luna16_annotations

annotations = parse_luna16_annotations(
    Path("/path/to/annotations.csv"),
    min_diameter_mm=3.0,              # optional size filter
    allowed_series_uids={"1.3.6.1"},  # optional whitelist
)
```

Each `LUNA16Annotation` contains the CT `series_uid`, the nodule centre
coordinates in `(x, y, z)` order, and the measured diameter in millimetres.
Invalid rows (missing IDs or numeric columns) raise `ValueError` with the row
number to make debugging easier. Files that cannot be located raise
`FileNotFoundError` before the CSV reader is initialised.

Group annotations by series when preparing dataset splits with
`group_annotations_by_series`, which returns a dictionary keyed by
`series_uid`:

```python
from src.data.annotations import group_annotations_by_series

by_series = group_annotations_by_series(annotations)
print(by_series["1.3.6.1"])  # -> list[LUNA16Annotation]
```

## LIDC-IDRI XML Reports

The parent LIDC-IDRI dataset stores richer radiologist feedback in XML
"read" reports. The `parse_lidc_annotations` helper scans those files and
returns a list of `LIDCAnnotation` objects, each bundling an unblinded nodule
with its review session metadata and slice-by-slice ROIs:

```python
from src.data.annotations import parse_lidc_annotations

annotations = parse_lidc_annotations(
    "/path/to/LIDC-IDRI-0001/annotations.xml",
    min_roi_count=2,                 # require two or more traced slices
    allowed_nodule_ids={"Nodule-3"},  # optional whitelist
)
```

An `LIDCAnnotation` includes:

- `series_uid`: the CT Series Instance UID extracted from the XML header.
- `nodule_id`: the radiologist-provided identifier for the nodule.
- `reading_session_id`: either the radiologist ID or a generated
  `session-{index}` fallback when the XML omits it.
- `rois`: an ordered tuple of `LIDCRoi` items. Each ROI stores the SOP UID (if
  present), the slice `z_position`, and a tuple of `(x, y)` edge coordinates.
- `characteristics`: optional radiologist scoring attributes captured as
  `(name, integer_value)` pairs.

The parser skips ROIs that are explicitly excluded (`<inclusion>false</inclusion>`)
or that contain no edge coordinates. You can require a minimum number of ROIs
per returned nodule with `min_roi_count`. Malformed XML generates a descriptive
`ValueError`, while a missing file raises `FileNotFoundError`.

## Data Quality Guarantees

- **Validation-first**: numeric conversion and structural checks run before the
  data leaves the parser. Any failure is reported with context so CI or ETL
  jobs fail fast.
- **Immutable outputs**: dataclasses are frozen and use `slots` to minimise
  accidental mutation and keep memory usage predictable when grouping by series
  or building training targets.
- **Test coverage**: `tests/test_annotations.py` exercises every branch—happy
  paths, filtering, and error handling—for both LUNA16 and LIDC-IDRI helpers.
  The repository locks `src/data/annotations.py` at 100% coverage to guard
  against regressions.

## Triage Score Ground Truth

`src/data/triage.py` derives a 1-10 triage score from the radiologist
characteristics captured in LIDC-IDRI XML files. The helper follows the scoring
heuristics tracked in Issue #12:

- **Base mapping**: linearly map the averaged malignancy vote (1-5) onto the
  triage scale (1-10).
- **Size adjustment**: shrink sub-4mm nodules by one point and increment nodules
  above 8/12/20mm by +1/+2/+3 respectively.
- **Morphology bonuses**: add +2 when mean spiculation ≥ 3.5 and +1 when texture
  or internal structure indicate a ground-glass appearance (≤ 2.5).
- **Growth bonus**: add +3 if temporal growth scores ≥ 4 or free-text labels flag
  the nodule as growing/progressive.

All contributions are summed then clipped to stay within 1-10, and the function
returns a `TriageScoreBreakdown` so downstream code can preserve explainability.
See `tests/test_triage.py` for concrete examples.

## Center Heatmap Ground Truth

Ground-truth heatmaps for the nodule-center detection branch live in
`src/data/heatmap.py`. Use `generate_center_heatmap` to stamp a Gaussian peak
around each `(z, y, x)` center coordinate:

```python
from src.data.heatmap import HeatmapConfig, generate_center_heatmap

heatmap = generate_center_heatmap(
    volume_shape=(128, 128, 128),
    centers=[(64.0, 64.0, 64.0), (80.3, 45.1, 90.7)],
    config=HeatmapConfig(sigma_mm=3.0, spacing=(1.0, 1.0, 1.0)),
)
```

Key behaviours:

- **Gaussian footprint**: configurable via `sigma_mm` and the physical voxel
  `spacing`. The helper truncates the kernel at `truncate × sigma` per axis to
  keep patches compact.
- **Overlap handling**: by default, peaks blend with a per-voxel `max`, but you
  can switch to `sum` mode for downstream calibration experiments.
- **Normalisation**: outputs are scaled back to `[0, 1]` after all peaks are
  applied so the training loss matches the expected binary target range.
- **Robustness**: centers falling outside the volume are ignored, and all
  inputs are validated for three-dimensionality.

Unit coverage sits in `tests/test_heatmap.py`, covering spacing-aware kernels
and overlapping peaks.

## Integration Points

1. **Dataset preparation**: use the CSV helper to build look-up tables when
   converting raw DICOM volumes into `.npy` tensors. The grouped view assists
   in building per-series patch lists.
2. **Ground truth generation**: leverage the ROI polygons from the XML parser
   to rasterise slice masks, compute malignancy priors, or fuse multi-reader
   annotations.
3. **Quality assurance**: failed parses surface corrupted annotation files
   early, helping data ops teams patch upstream inconsistencies before training
   jobs launch.

Refer back to the [Pipeline overview](PIPELINE.md#2-data-intake--ground-truth)
for the milestone context and to track follow-up tasks (triage scores, heatmap
rasterisation, etc.).
