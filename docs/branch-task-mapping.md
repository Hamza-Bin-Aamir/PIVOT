# Branch-Task Mapping

This document maps logical Git branches (tasks) to their constituent GitHub issues from the backlog.

---

## Phase 1: Foundation & Setup

### Branch 1: `setup/project-foundation`
**Purpose**: Establish core project infrastructure and environment

**Issues**:
- #1: setup[env]: Initialize project structure and dependencies
- #2: setup[docker]: Create Docker environment for training and inference
- #3: setup[config]: Implement configuration management system

---

## Phase 2: Data Pipeline

### Branch 2: `feat/data-loading-core`
**Purpose**: Implement DICOM loading and basic preprocessing

**Issues**:
- #4: feat[data]: Implement DICOM loading and metadata extraction
- #5: feat[data]: Build isotropic resampling module
- #6: feat[data]: Create intensity normalization pipeline

---

### Branch 3: `feat/data-augmentation`
**Purpose**: Implement data augmentation pipeline

**Issues**:
- #7: feat[data]: Implement 3D data augmentation

---

### Branch 4: `feat/data-annotations`
**Purpose**: Parse and process dataset annotations

**Issues**:
- #10: feat[data]: Parse LUNA16 annotations
- #11: feat[data]: Parse LIDC-IDRI annotations XML

---

### Branch 5: `feat/ground-truth-generation`
**Purpose**: Generate training ground truth labels

**Issues**:
- #12: feat[data]: Generate triage score ground truth
- #13: feat[data]: Create center heatmap ground truth

---

### Branch 6: `feat/dataset-integration`
**Purpose**: Integrate all data components into PyTorch Dataset

**Issues**:
- #8: feat[data]: Build PyTorch Dataset class
- #9: feat[data]: Create data validation and quality checks

---

## Phase 3: Model Architecture

### Branch 7: `feat/unet-backbone`
**Purpose**: Implement core 3D U-Net architecture

**Issues**:
- #14: feat[model]: Implement 3D U-Net encoder
- #15: feat[model]: Implement 3D U-Net decoder
- #16: feat[model]: Implement 3D U-Net bottleneck

---

### Branch 8: `feat/task-heads`
**Purpose**: Implement multi-task prediction heads

**Issues**:
- #17: feat[model]: Build segmentation head
- #18: feat[model]: Build center point detection head
- #19: feat[model]: Build size regression head
- #20: feat[model]: Build malignancy triage head

---

### Branch 9: `feat/model-integration`
**Purpose**: Integrate all model components into unified architecture

**Issues**:
- #21: feat[model]: Integrate multi-task model

---

## Phase 4: Loss Functions

### Branch 10: `feat/loss-functions`
**Purpose**: Implement all task-specific loss functions

**Issues**:
- #22: feat[loss]: Implement Dice loss for segmentation
- #23: feat[loss]: Implement binary cross-entropy loss
- #24: feat[loss]: Implement focal loss for center detection
- #25: feat[loss]: Implement smooth L1 loss for size regression
- #26: feat[loss]: Implement weighted BCE for triage scores
- #27: feat[loss]: Create multi-task loss aggregator

---

## Phase 5: Training Infrastructure

### Branch 11: `feat/training-setup`
**Purpose**: Set up PyTorch Lightning training framework

**Issues**:
- #28: feat[train]: Set up PyTorch Lightning module
- #29: feat[train]: Configure AdamW optimizer
- #30: feat[train]: Implement learning rate scheduling

---

### Branch 12: `feat/training-optimization`
**Purpose**: Implement training optimizations and techniques

**Issues**:
- #31: feat[train]: Add mixed precision training
- #32: feat[train]: Implement hard negative mining
- #33: feat[train]: Create training data loader
- #34: feat[train]: Create validation data loader

---

### Branch 13: `feat/training-monitoring`
**Purpose**: Implement training monitoring and checkpointing

**Issues**:
- #35: feat[train]: Add model checkpointing
- #36: feat[train]: Implement early stopping
- #37: feat[train]: Set up experiment tracking with W&B

---

### Branch 14: `feat/training-state-tracking`
**Purpose**: Implement real-time training state tracking

**Issues**:
- #78: feat[train]: Implement training state tracker
- #79: feat[train]: Add epoch metrics collector
- #80: feat[train]: Create training progress callback
- #81: feat[train]: Implement real-time metrics streaming

---

## Phase 6: Inference Pipeline

### Branch 15: `feat/inference-core`
**Purpose**: Implement sliding window inference and overlap handling

**Issues**:
- #38: feat[inference]: Implement sliding window inference
- #39: feat[inference]: Implement overlap blending
- #40: feat[inference]: Create inference data loader

---

### Branch 16: `feat/inference-postprocessing`
**Purpose**: Implement inference post-processing and nodule extraction

**Issues**:
- #41: feat[inference]: Implement peak detection on center heatmap
- #42: feat[inference]: Extract nodule properties
- #43: feat[inference]: Implement post-processing filters

---

### Branch 17: `feat/inference-optimization`
**Purpose**: Optimize and format inference outputs

**Issues**:
- #44: feat[inference]: Build triage score calibration
- #45: feat[inference]: Create structured output formatter
- #46: feat[inference]: Optimize inference with TorchScript

---

## Phase 7: Evaluation & Metrics

### Branch 18: `feat/evaluation-metrics`
**Purpose**: Implement comprehensive evaluation metrics

**Issues**:
- #47: feat[eval]: Implement FROC curve calculation
- #48: feat[eval]: Calculate Dice score for segmentation
- #49: feat[eval]: Measure center point accuracy
- #50: feat[eval]: Evaluate size estimation accuracy
- #51: feat[eval]: Validate triage score correlation

---

### Branch 19: `feat/evaluation-pipeline`
**Purpose**: Create evaluation pipeline and reporting

**Issues**:
- #52: feat[eval]: Create evaluation pipeline
- #53: feat[eval]: Generate performance report

---

## Phase 8: Visualization

### Branch 20: `feat/visualization-2d`
**Purpose**: Implement 2D visualization tools

**Issues**:
- #54: feat[viz]: Create 2D overlay visualizations
- #55: feat[viz]: Implement multi-slice viewer

---

### Branch 21: `feat/visualization-3d`
**Purpose**: Implement 3D visualization and attention maps

**Issues**:
- #56: feat[viz]: Build 3D lung rendering
- #57: feat[viz]: Implement Grad-CAM attention maps
- #58: feat[viz]: Create confidence visualization
- #59: feat[viz]: Generate clinician report

---

## Phase 9: Testing

### Branch 22: `test/unit-tests`
**Purpose**: Implement comprehensive unit tests

**Issues**:
- #60: test[unit]: Write tests for preprocessing
- #61: test[unit]: Write tests for model components
- #62: test[unit]: Write tests for loss functions

---

### Branch 23: `test/integration-tests`
**Purpose**: Implement integration and performance tests

**Issues**:
- #63: test[integration]: Test end-to-end training
- #64: test[integration]: Test end-to-end inference
- #65: test[performance]: Benchmark inference speed
- #66: test[regression]: Create regression test suite

---

## Phase 10: API & Monitoring

### Branch 24: `feat/api-foundation`
**Purpose**: Set up FastAPI server and core infrastructure

**Issues**:
- #82: feat[api]: Set up FastAPI server structure
- #83: feat[api]: Implement training process manager
- #84: feat[api]: Create training session manager

---

### Branch 25: `feat/api-status-endpoints`
**Purpose**: Implement status and monitoring endpoints

**Issues**:
- #85: feat[api]: Add GET /status endpoint
- #86: feat[api]: Add GET /epochs endpoint
- #87: feat[api]: Add GET /metrics endpoint
- #88: feat[api]: Add GET /metrics/latest endpoint
- #89: feat[api]: Add GET /metrics/history endpoint

---

### Branch 26: `feat/api-training-control`
**Purpose**: Implement training control endpoints

**Issues**:
- #90: feat[api]: Add POST /training/start endpoint
- #91: feat[api]: Add POST /training/pause endpoint
- #92: feat[api]: Add POST /training/resume endpoint
- #93: feat[api]: Add POST /training/stop endpoint

---

### Branch 27: `feat/api-graphs`
**Purpose**: Implement graph data endpoints

**Issues**:
- #94: feat[api]: Add GET /graphs/loss endpoint
- #95: feat[api]: Add GET /graphs/metrics endpoint
- #96: feat[api]: Add GET /graphs/learning-rate endpoint
- #97: feat[api]: Add GET /graphs/gpu-usage endpoint

---

### Branch 28: `feat/api-realtime`
**Purpose**: Implement real-time communication

**Issues**:
- #98: feat[api]: Implement WebSocket endpoint for live updates
- #99: feat[api]: Add Server-Sent Events (SSE) endpoint
- #100: feat[api]: Implement notification system

---

### Branch 29: `feat/dashboard`
**Purpose**: Create web-based monitoring dashboard

**Issues**:
- #101: feat[dashboard]: Create simple HTML dashboard
- #102: feat[dashboard]: Add training configuration panel
- #103: feat[dashboard]: Add metrics visualization panel
- #104: feat[dashboard]: Add training logs viewer

---

### Branch 30: `feat/api-infrastructure`
**Purpose**: Add logging, persistence, and checkpointing

**Issues**:
- #105: feat[api]: Implement structured logging
- #106: feat[api]: Add database for metrics persistence
- #107: feat[api]: Implement checkpoint management API

---

### Branch 31: `feat/system-monitoring`
**Purpose**: Implement system resource monitoring and health checks

**Issues**:
- #108: feat[monitor]: Add system resource monitoring
- #109: feat[monitor]: Implement health checks
- #110: feat[monitor]: Add alerting rules engine
- #111: feat[monitor]: Implement webhook notifications

---

## Phase 11: Security & Performance

### Branch 32: `feat/security`
**Purpose**: Implement authentication and authorization

**Issues**:
- #112: feat[security]: Add API authentication
- #113: feat[security]: Add authorization middleware
- #114: feat[security]: Add HTTPS support

---

### Branch 33: `feat/performance-optimization`
**Purpose**: Implement performance optimizations

**Issues**:
- #115: feat[perf]: Implement response caching
- #116: feat[perf]: Add response compression
- #117: feat[perf]: Optimize database queries

---

## Phase 12: Testing & Documentation (API)

### Branch 34: `test/api-tests`
**Purpose**: Implement comprehensive API tests

**Issues**:
- #118: test[api]: Write unit tests for API endpoints
- #119: test[api]: Write integration tests for training flow
- #120: test[api]: Add load testing

---

### Branch 35: `docs/api-documentation`
**Purpose**: Create comprehensive API documentation

**Issues**:
- #121: docs[api]: Generate OpenAPI documentation
- #122: docs[api]: Write API usage guide
- #123: docs[deploy]: Write deployment guide

---

## Phase 13: Deployment & Production

### Branch 36: `feat/deployment-api`
**Purpose**: Create deployment infrastructure for API/inference

**Issues**:
- #67: feat[deploy]: Create REST API for inference
- #68: feat[deploy]: Add DICOM input/output support
- #69: feat[deploy]: Implement batch processing
- #70: feat[deploy]: Create monitoring dashboard

---

### Branch 37: `docs/operations`
**Purpose**: Create operations and user documentation

**Issues**:
- #71: feat[ops]: Write deployment documentation
- #72: feat[ops]: Create model versioning system
- #73: docs[guide]: Write data preparation guide
- #74: docs[guide]: Write training guide
- #75: docs[guide]: Write inference guide
- #76: docs[api]: Generate API documentation
- #77: docs[paper]: Write technical report

---

## Phase 14: Advanced Features

### Branch 38: `feat/api-advanced`
**Purpose**: Implement advanced API features

**Issues**:
- #124: feat[api]: Add model comparison endpoint
- #125: feat[api]: Add experiment tracking integration
- #126: feat[api]: Add configuration version control
- #127: feat[api]: Implement training queue system

---

## Summary
- **Total Branches**: 38
- **Total Issues**: 127
- **Average Issues per Branch**: 3.3

## Branch Size Distribution
- Small (1-2 issues): 8 branches
- Medium (3-5 issues): 22 branches
- Large (6+ issues): 8 branches
