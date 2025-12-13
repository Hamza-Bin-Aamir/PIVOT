# End-to-End Pipeline

The pipeline in this document mirrors the active GitHub project issues. Each
stage lists the primary responsibilities and the issue numbers that track the
work. Follow the stages from top to bottom to understand how raw data flows
through preprocessing, model training, inference, and the serving stack.

```
Raw CT Scans + Radiologist Annotations
        |
        v
Data Intake & Validation (#4, #9, #10-#13)
        |
        v
Preprocessing & Patches (#5-#8)
        |
        v
Model & Training Core (#14-#37)
        |
        v
Inference & Post-processing (#38-#46)
        |
        v
Evaluation & Visualisation (#47-#59)
        |
        v
Testing & Quality Gates (#60-#66)
        |
        v
API, Dashboard, Deployment, Ops (#67-#127)
```

## 1. Foundation & Configuration

- **Project bootstrap**: Initialize structure and dependencies (Issue #1).
- **Docker & runtime setup**: Provide parity across GPU backends (Issue #2).
- **Configuration system**: Define and validate typed configs (Issue #3).

## 2. Data Intake & Ground Truth

- **DICOM ingestion**: pydicom loader with full metadata extraction (Issue #4).
- **Isotropic resampling**: Trilinear interpolation to 1mm spacing (Issue #5).
- **Intensity normalization**: HU clipping to [-1000, 400], scaling to [0, 1], optional histogram equalisation (Issue #6).
- **Augmentation**: 3D transforms for rotations, scaling, noise (Issue #7).
- **Dataset wrapper**: Balanced sampling, caching, multiprocessing (Issue #8).
- **Quality assurance**: Scan completeness, artifact detection (Issue #9).
- **Annotation parsing**: LUNA16 CSV, LIDC-IDRI XML, triage scoring, heatmaps (Issues #10-#13).

## 3. Model Architecture

- **U-Net encoder/decoder/bottleneck**: Four-level 3D backbone (Issues #14-#16).
- **Task heads**: Segmentation, center heatmap, size regression, triage (Issues #17-#20).
- **Multi-task integration**: Single Lightning-compatible module (Issue #21).

## 4. Losses & Optimisation

- **Loss components**: Dice, BCE, focal, smooth L1, weighted BCE (Issues #22-#26).
- **Loss aggregation**: Weighted combination with logging (Issue #27).
- **Trainer scaffold**: Lightning module, optimisers, schedulers, AMP (Issues #28-#31).
- **Curriculum**: Hard negative mining, data loaders, validation loops (Issues #32-#34).
- **Training resilience**: Checkpointing, early stopping, experiment tracking (Issues #35-#37).

## 5. Training State & Observability

- **State tracking**: Dataclasses, callbacks, streaming metrics (Issues #78-#82).
- **Process management**: Spawned training jobs, session manager, status endpoints (Issues #83-#90).
- **Control plane**: Pause/resume/stop training with consistent checkpoints (Issues #91-#93).

## 6. Inference Pipeline

- **Sliding window inference**: Patch extraction, overlap blending, loaders (Issues #38-#40).
- **Peak detection & properties**: Center heatmap, nodule properties, filters (Issues #41-#43).
- **Calibration & formatting**: Triage calibration, structured outputs, TorchScript (Issues #44-#46).

## 7. Evaluation & Reporting

- **Metrics**: Dice, accuracy, localization, FROC, size estimation (Issues #47-#53).
- **Visualisations**: Overlay slices, multi-viewer, 3D rendering, Grad-CAM (Issues #54-#58).
- **Clinician reporting**: Text reports with QA checks (Issue #59).

## 8. Testing & Quality Gates

- **Unit tests**: Preprocessing, models, losses (Issues #60-#62).
- **Integration tests**: End-to-end training and inference (Issues #63-#64).
- **Performance & regression**: Benchmarking and golden outputs (Issues #65-#66).

## 9. Deployment & Operations

- **Inference API**: REST endpoints, DICOM I/O, batch processing (Issues #67-#69).
- **Monitoring & documentation**: Deployment guide, throughput dashboards (Issues #70-#75).
- **Public documentation**: API usage, OpenAPI specs, guides, whitepaper (Issues #71-#77).

## 10. Platform & Security Layer

- **FastAPI server**: Core app, middleware, versioning (Issue #82).
- **Authentication & authorization**: API keys, RBAC, HTTPS (Issues #112-#114).
- **Performance**: Caching, compression, indexed queries (Issues #115-#117).
- **Observability**: Structured logging, health checks, alerts, webhooks (Issues #105, #109-#111).
- **Experiment services**: Metrics APIs, comparison, experiment tracking (Issues #87-#125).
- **Job orchestration**: Queueing, scheduling, configuration versioning (Issues #126-#127).

## Using This Pipeline

1. **Planning**: Align feature work with the stage that best fits the issue you
   are tackling. Progress generally moves downward through the stages.
2. **Implementation**: Keep data contracts consistent—later stages depend on
   the outputs defined above.
3. **Validation**: When finishing a stage, close or update the linked issues and
   run the relevant tests or benchmarks before moving forward.

For clarifications, cross-reference the issue tracker or open a discussion
thread describing the data sample and expected transformation.
