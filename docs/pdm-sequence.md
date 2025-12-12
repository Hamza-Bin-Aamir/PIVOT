# Precedence Diagramming Method (PDM) - Branch Sequence

This document outlines the dependency sequence for all 38 project branches using the Precedence Diagramming Method (PDM).

---

## PDM Network Diagram (Simplified Overview)

```
START → B1 (Foundation)
         │
         ├──────────┬──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼          ▼
    DATA PIPELINE  MODEL     LOSS     TRAINING   API/OPS
    (B2→B6)        (B7→B9)   (B10)    (B11→B14)  (B24→B31)
         │          │          │          │          │
         └──────────┴──────────┴──────────┴──────────┘
                             ▼
                    INFERENCE (B15→B17)
                             │
         ┌───────────────────┴────────────────┬────────────┐
         ▼                   ▼                ▼            ▼
    EVALUATION         VISUALIZATION      TESTING     DEPLOYMENT
    (B18→B19)          (B20→B21)          (B22→B23)   (B36→B38)
```

---

## Dependency Table (Complete)

| Branch | Branch Name | Predecessors | Successors | Phase |
|--------|-------------|--------------|------------|-------|
| **B1** | setup/project-foundation | START | B2, B3, B7, B10, B11, B24 | Foundation |
| **B2** | feat/data-loading-core | B1 | B4, B6 | Data |
| **B3** | feat/data-augmentation | B1 | B6 | Data |
| **B4** | feat/data-annotations | B2 | B5 | Data |
| **B5** | feat/ground-truth-generation | B4 | B6 | Data |
| **B6** | feat/dataset-integration | B2, B3, B5 | B12, B15 | Data |
| **B7** | feat/unet-backbone | B1 | B8 | Model |
| **B8** | feat/task-heads | B7 | B9 | Model |
| **B9** | feat/model-integration | B8 | B11 | Model |
| **B10** | feat/loss-functions | B1 | B11 | Loss |
| **B11** | feat/training-setup | B1, B9, B10 | B12, B13 | Training |
| **B12** | feat/training-optimization | B6, B11 | B13 | Training |
| **B13** | feat/training-monitoring | B11, B12 | B14 | Training |
| **B14** | feat/training-state-tracking | B13 | B24 | Training |
| **B15** | feat/inference-core | B6, B9 | B16 | Inference |
| **B16** | feat/inference-postprocessing | B15 | B17 | Inference |
| **B17** | feat/inference-optimization | B16 | B18, B20, B36 | Inference |
| **B18** | feat/evaluation-metrics | B17 | B19 | Evaluation |
| **B19** | feat/evaluation-pipeline | B18 | B22 | Evaluation |
| **B20** | feat/visualization-2d | B17 | B21 | Visualization |
| **B21** | feat/visualization-3d | B20 | B22 | Visualization |
| **B22** | test/unit-tests | B6, B9, B10 | B23 | Testing |
| **B23** | test/integration-tests | B19, B21, B22 | - | Testing |
| **B24** | feat/api-foundation | B1, B14 | B25, B26, B30 | API |
| **B25** | feat/api-status-endpoints | B24 | B28, B34 | API |
| **B26** | feat/api-training-control | B24 | B28, B34 | API |
| **B27** | feat/api-graphs | B25 | B29, B34 | API |
| **B28** | feat/api-realtime | B25, B26 | B29, B34 | API |
| **B29** | feat/dashboard | B27, B28 | B34 | API |
| **B30** | feat/api-infrastructure | B24 | B31, B34 | API |
| **B31** | feat/system-monitoring | B30 | B32, B34 | Monitoring |
| **B32** | feat/security | B31 | B33, B34 | Security |
| **B33** | feat/performance-optimization | B32 | B34 | Performance |
| **B34** | test/api-tests | B25, B26, B27, B28, B29, B30, B31, B32, B33 | B35 | Testing |
| **B35** | docs/api-documentation | B34 | B36 | Documentation |
| **B36** | feat/deployment-api | B17, B35 | B37 | Deployment |
| **B37** | docs/operations | B36 | B38 | Documentation |
| **B38** | feat/api-advanced | B37 | END | Advanced |

---

## Detailed Phase Breakdown

### **Phase 1: Foundation** (Critical Path Start)

#### B1: setup/project-foundation
- **Predecessors**: START
- **Successors**: B2, B3, B7, B10, B11, B24
- **Rationale**: Core infrastructure must be established before any development
- **Enables**: All parallel development streams

---

### **Phase 2: Data Pipeline** (Stream A - Critical Path)

#### B2: feat/data-loading-core
- **Predecessors**: B1
- **Successors**: B4, B6
- **Rationale**: DICOM loading and preprocessing are foundations for annotations and dataset

#### B3: feat/data-augmentation
- **Predecessors**: B1
- **Successors**: B6
- **Rationale**: Independent augmentation can be developed in parallel

#### B4: feat/data-annotations
- **Predecessors**: B2
- **Successors**: B5
- **Rationale**: Annotations require loaded data

#### B5: feat/ground-truth-generation
- **Predecessors**: B4
- **Successors**: B6
- **Rationale**: Ground truth needs parsed annotations

#### B6: feat/dataset-integration
- **Predecessors**: B2, B3, B5
- **Successors**: B12, B15
- **Rationale**: Integrates all data components (loading, augmentation, ground truth)
- **Critical Junction**: Enables both training optimization and inference

---

### **Phase 3: Model Architecture** (Stream B - Parallel)

#### B7: feat/unet-backbone
- **Predecessors**: B1
- **Successors**: B8
- **Rationale**: Core U-Net can develop independently

#### B8: feat/task-heads
- **Predecessors**: B7
- **Successors**: B9
- **Rationale**: Heads require base architecture

#### B9: feat/model-integration
- **Predecessors**: B8
- **Successors**: B11
- **Rationale**: Complete model needed for training setup

---

### **Phase 4: Loss Functions** (Stream C - Parallel)

#### B10: feat/loss-functions
- **Predecessors**: B1
- **Successors**: B11
- **Rationale**: Independent development, needed for training

---

### **Phase 5: Training Infrastructure** (Stream D - Convergence Point)

#### B11: feat/training-setup
- **Predecessors**: B1, B9, B10
- **Successors**: B12, B13
- **Rationale**: Requires model and losses; sets up PyTorch Lightning framework

#### B12: feat/training-optimization
- **Predecessors**: B6, B11
- **Successors**: B13
- **Rationale**: Needs dataset and training setup for data loaders

#### B13: feat/training-monitoring
- **Predecessors**: B11, B12
- **Successors**: B14
- **Rationale**: Builds on training setup and optimization

#### B14: feat/training-state-tracking
- **Predecessors**: B13
- **Successors**: B24
- **Rationale**: Real-time tracking enables API monitoring

---

### **Phase 6: Inference Pipeline** (Parallel with Training)

#### B15: feat/inference-core
- **Predecessors**: B6, B9
- **Successors**: B16
- **Rationale**: Needs dataset handling and complete model

#### B16: feat/inference-postprocessing
- **Predecessors**: B15
- **Successors**: B17
- **Rationale**: Builds on core inference

#### B17: feat/inference-optimization
- **Predecessors**: B16
- **Successors**: B18, B20, B36
- **Rationale**: Optimized inference enables evaluation, visualization, and deployment

---

### **Phase 7: Evaluation & Visualization** (Parallel Streams)

#### B18: feat/evaluation-metrics
- **Predecessors**: B17
- **Successors**: B19
- **Rationale**: Metrics need optimized inference

#### B19: feat/evaluation-pipeline
- **Predecessors**: B18
- **Successors**: B22
- **Rationale**: Pipeline uses metrics

#### B20: feat/visualization-2d
- **Predecessors**: B17
- **Successors**: B21
- **Rationale**: Basic visualization first

#### B21: feat/visualization-3d
- **Predecessors**: B20
- **Successors**: B22
- **Rationale**: Advanced visualization builds on 2D

---

### **Phase 8: Core Testing** (Integration Point)

#### B22: test/unit-tests
- **Predecessors**: B6, B9, B10
- **Successors**: B23
- **Rationale**: Can test components as they're ready

#### B23: test/integration-tests
- **Predecessors**: B19, B21, B22
- **Successors**: -
- **Rationale**: Integration tests need complete evaluation and visualization

---

### **Phase 9: API Development** (Stream E - Parallel with Evaluation)

#### B24: feat/api-foundation
- **Predecessors**: B1, B14
- **Successors**: B25, B26, B30
- **Rationale**: Foundation needs training state tracking from B14

#### B25: feat/api-status-endpoints
- **Predecessors**: B24
- **Successors**: B28, B34
- **Rationale**: Status endpoints are core functionality

#### B26: feat/api-training-control
- **Predecessors**: B24
- **Successors**: B28, B34
- **Rationale**: Control endpoints parallel to status

#### B27: feat/api-graphs
- **Predecessors**: B25
- **Successors**: B29, B34
- **Rationale**: Graphs depend on status data

#### B28: feat/api-realtime
- **Predecessors**: B25, B26
- **Successors**: B29, B34
- **Rationale**: Real-time needs both status and control

#### B29: feat/dashboard
- **Predecessors**: B27, B28
- **Successors**: B34
- **Rationale**: Dashboard consumes graphs and real-time updates

---

### **Phase 10: Infrastructure & Monitoring** (Parallel with API)

#### B30: feat/api-infrastructure
- **Predecessors**: B24
- **Successors**: B31, B34
- **Rationale**: Logging and persistence are foundational

#### B31: feat/system-monitoring
- **Predecessors**: B30
- **Successors**: B32, B34
- **Rationale**: Monitoring needs infrastructure

---

### **Phase 11: Security & Performance** (Sequential)

#### B32: feat/security
- **Predecessors**: B31
- **Successors**: B33, B34
- **Rationale**: Security should be implemented before optimization

#### B33: feat/performance-optimization
- **Predecessors**: B32
- **Successors**: B34
- **Rationale**: Optimize after security is in place

---

### **Phase 12: API Testing & Documentation** (Integration)

#### B34: test/api-tests
- **Predecessors**: B25, B26, B27, B28, B29, B30, B31, B32, B33
- **Successors**: B35
- **Rationale**: Test all API components comprehensively

#### B35: docs/api-documentation
- **Predecessors**: B34
- **Successors**: B36
- **Rationale**: Document tested API

---

### **Phase 13: Deployment** (Final Stages)

#### B36: feat/deployment-api
- **Predecessors**: B17, B35
- **Successors**: B37
- **Rationale**: Deploy optimized inference with documented API

#### B37: docs/operations
- **Predecessors**: B36
- **Successors**: B38
- **Rationale**: Create ops docs after deployment infrastructure

#### B38: feat/api-advanced
- **Predecessors**: B37
- **Successors**: END
- **Rationale**: Advanced features built on stable deployment

---

## Critical Paths Analysis

### **Primary Critical Path** (Longest Duration):
```
B1 → B2 → B4 → B5 → B6 → B12 → B13 → B14 → B24 → B30 → B31 → B32 → B33 → B34 → B35 → B36 → B37 → B38
```
**Length**: 18 sequential branches
**Bottleneck**: Data pipeline (B2→B6) and API infrastructure (B24→B38)

### **Alternative Critical Paths**:

**Model-centric path**:
```
B1 → B7 → B8 → B9 → B11 → B13 → B14 → ...
```

**Inference-centric path**:
```
B1 → B2 → B6 → B15 → B16 → B17 → B36 → ...
```

---

## Parallelization Levels

### **Level 0**: Start
- B1

### **Level 1**: After B1
- B2, B3, B7, B10, B11, B24

### **Level 2**:
- B4 (after B2)
- B8 (after B7)

### **Level 3**:
- B5 (after B4)

### **Level 4**:
- B6 (after B2, B3, B5)
- B9 (after B8)
- B25, B26, B30 (after B24)

### **Level 5**:
- B12 (after B6, B11)
- B15 (after B6, B9)
- B27 (after B25)
- B28 (after B25, B26)
- B31 (after B30)

### **Level 6**:
- B13 (after B11, B12)
- B16 (after B15)
- B29 (after B27, B28)
- B32 (after B31)

### **Level 7**:
- B14 (after B13)
- B17 (after B16)
- B33 (after B32)

### **Level 8**:
- B18, B20 (after B17)
- B34 (after B25-B33)

### **Level 9**:
- B19 (after B18)
- B21 (after B20)
- B22 (after B6, B9, B10)
- B35 (after B34)

### **Level 10**:
- B23 (after B19, B21, B22)
- B36 (after B17, B35)

### **Level 11**:
- B37 (after B36)

### **Level 12**:
- B38 (after B37)

---

## Resource Allocation Strategies

### **Single Developer** (Serial Execution):
Follow critical path with occasional detours for independent work:
```
B1 → B2 → B3 → B4 → B5 → B6 → B7 → B8 → B9 → B10 → B11 → B12 → B13 → B14 →
B15 → B16 → B17 → B18 → B19 → B20 → B21 → B22 → B23 → B24 → B25 → B26 →
B27 → B28 → B29 → B30 → B31 → B32 → B33 → B34 → B35 → B36 → B37 → B38
```
**Estimated Duration**: 38 sequential task periods

### **Two Developers** (Optimal Split):
**Developer 1 (Data & Training focus)**:
```
B1 → B2 → B4 → B5 → B6 → B12 → B13 → B14 → B24 → B25 → B26 → B30 → B31 → B34 → B35
```

**Developer 2 (Model & Inference focus)**:
```
Wait for B1 → B7 → B8 → B9 → B11 → B15 → B16 → B17 → B18 → B19 → B20 →
B21 → B22 → B23 → B36 → B37 → B38
```

Plus parallel: B3, B10, B27, B28, B29, B32, B33

### **Three Developers** (Maximum Parallelism):
**Developer 1 (Data Pipeline)**:
```
B1 → B2 → B4 → B5 → B6 → B22 → B23
```

**Developer 2 (Model & Training)**:
```
Wait for B1 → B7 → B8 → B9 → B10 → B11 → B12 → B13 → B14 → B15 → B16 → B17 →
B18 → B19 → B36 → B37 → B38
```

**Developer 3 (API & Infrastructure)**:
```
Wait for B1 → B3 → ... → Wait for B14 → B24 → B25 → B26 → B27 → B28 → B29 →
B30 → B31 → B32 → B33 → B34 → B35 → B20 → B21
```

### **Four+ Developers**:
Additional developers can work on:
- Visualization (B20-B21) independently
- Documentation (B35, B37) in parallel
- Testing (B22-B23, B34) as components become ready
- Advanced features (B38) early prototyping

---

## Key Insights

1. **Critical Bottleneck**: Data pipeline (B2→B6) blocks training and inference
2. **High Parallelism Window**: After B1, up to 6 branches can work simultaneously
3. **Integration Points**: B6, B11, B17, B34 are major convergence points
4. **Testing Strategy**: Unit tests (B22) can start early; integration tests (B23, B34) need more components
5. **API Independence**: API development (B24-B38) can largely parallel ML development
6. **Late-Stage Serialization**: B32→B33→B34→B35→B36→B37→B38 creates bottleneck at end

---

## Recommendations

1. **Start Early on B1**: Foundation is critical path blocker
2. **Prioritize B2-B6**: Data pipeline is longest sequential chain
3. **Parallelize Aggressively**: Utilize B3, B7, B10, B24 parallelism after B1
4. **Front-load Testing**: Start B22 as soon as B6, B9, B10 complete
5. **Overlap API Development**: B24-B31 can proceed during B15-B23
6. **Plan for Integration**: Reserve time at B6, B11, B17, B34 for integration work
7. **Document Continuously**: Don't wait for B35, B37 - document as you build
