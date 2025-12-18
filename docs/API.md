# API Documentation

This document describes the PIVOT FastAPI server implementation for monitoring and controlling training processes.

---

## Branch: `feat/api-status-endpoints`

**Status**: ✅ Complete (6 commits, 1731+ lines changed)
**Base Branch**: `feat/api-foundation`
**Merged**: No (Active)

### Overview

This branch implements comprehensive status and monitoring endpoints for the PIVOT training API. It provides real-time access to training session information, epoch progress, metrics tracking, and system status.

### Implemented Features

#### 1. System Status Endpoint (Issue #85)
**Commit**: `85b8693`
**Endpoint**: `GET /status`
**Purpose**: Retrieve overall system status and training session overview

**Response Model**:
```python
class StatusResponse(BaseModel):
    status: str  # "running" | "idle" | "error"
    active_session_id: str | None
    total_sessions: int
    sessions: list[SessionInfo]
```

**Features**:
- Lists all training sessions with metadata
- Indicates active training session
- Provides system health status
- Returns 500 if session manager not initialized

**Tests**: 7 tests, 100% coverage
- Valid status retrieval
- Multiple sessions handling
- Empty sessions case
- Session manager unavailability
- Response structure validation
- Session ordering
- Metadata completeness

---

#### 2. Epochs Progress Endpoint (Issue #86)
**Commit**: `43e06e1`
**Endpoint**: `GET /epochs/{session_id}` and `GET /epochs`
**Purpose**: Track epoch-by-epoch training progress

**Response Model**:
```python
class EpochInfo(BaseModel):
    epoch: int
    status: str
    train_metrics: dict[str, float] | None
    val_metrics: dict[str, float] | None

class EpochsResponse(BaseModel):
    session_id: str
    experiment_name: str
    total_epochs: int
    completed_epochs: int
    current_epoch: int
    epochs: list[EpochInfo]
```

**Features**:
- Defaults to latest session if no `session_id` provided
- Returns epoch-by-epoch progress
- Includes train and validation metrics per epoch
- Returns 404 for invalid session
- Returns 404 if no sessions exist

**Tests**: 6 tests, 94% coverage
- Valid session retrieval
- Invalid session handling
- Session manager unavailability
- Response structure validation
- Multiple sessions support
- Empty epochs handling

---

#### 3. Metrics Overview Endpoint (Issue #87)
**Commit**: `c4c1412`
**Endpoint**: `GET /metrics/{session_id}` and `GET /metrics`
**Purpose**: Retrieve all training and validation metrics for a session

**Response Model**:
```python
class MetricPoint(BaseModel):
    epoch: int
    value: float
    timestamp: datetime

class MetricSeries(BaseModel):
    name: str
    values: list[MetricPoint]
    latest_value: float | None

class MetricsResponse(BaseModel):
    session_id: str
    experiment_name: str
    train_metrics: dict[str, MetricSeries]
    val_metrics: dict[str, MetricSeries]
    total_metrics: int
```

**Features**:
- Defaults to latest session if no `session_id` provided
- Organizes metrics by type (train/validation)
- Tracks complete metric history
- Includes latest value for quick access
- Returns 404 for invalid session
- Returns 404 if no sessions exist

**Tests**: 7 tests, 93% coverage
- Valid session retrieval
- Invalid session handling
- Session manager unavailability
- Response structure validation
- Multiple sessions support
- Empty metrics handling
- Total metrics count verification

---

#### 4. Latest Metrics Endpoint (Issue #88)
**Commit**: `979bb2e`
**Endpoint**: `GET /metrics/{session_id}/latest` and `GET /metrics/latest`
**Purpose**: Get most recent metric values for quick dashboard updates

**Response Model**:
```python
class LatestMetricsResponse(BaseModel):
    session_id: str
    experiment_name: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    timestamp: datetime | None
```

**Features**:
- Defaults to latest session if no `session_id` provided
- Returns only most recent values (no history)
- Optimized for dashboard refresh
- Includes timestamp of latest update
- Returns 404 for invalid session
- Returns 404 if no sessions exist

**Tests**: 7 tests, coverage integrated with metrics endpoint
- Valid session retrieval
- Invalid session handling
- Session manager unavailability
- Response structure validation
- Timestamp format validation
- Empty metrics for new session
- Multiple sessions support

---

#### 5. Metrics History Endpoint (Issue #89)
**Commit**: `964603b`
**Endpoint**: `GET /metrics/{session_id}/history/{metric_type}/{metric_name}` and `GET /metrics/history/{metric_type}/{metric_name}`
**Purpose**: Retrieve complete history for a specific metric (for graphing)

**Path Parameters**:
- `metric_type`: "train" or "val"
- `metric_name`: Name of the metric (e.g., "loss", "accuracy")

**Response Model**:
```python
class MetricsHistoryResponse(BaseModel):
    session_id: str
    experiment_name: str
    metric_type: str
    metric_name: str
    history: list[MetricPoint]
    total_points: int
```

**Features**:
- Defaults to latest session if no `session_id` provided
- Validates metric_type ("train" or "val")
- Returns chronological history
- Optimized for plotting
- Returns 404 for invalid session
- Returns 400 for invalid metric_type
- Returns 404 if no sessions exist

**Tests**: 9 tests, coverage integrated with metrics endpoint
- Valid train metric retrieval
- Valid validation metric retrieval
- Invalid metric type handling
- Invalid session handling
- Session manager unavailability
- Response structure validation
- Empty history for new session
- Multiple metrics support
- Total points count verification

---

#### 6. Default Session Behavior Enhancement
**Commit**: `1afd288`
**Purpose**: Enable sessionless API calls that default to the most recent training session

**Implementation**:
- Added `get_latest_session_id()` helper function
- Uses `max(sessions, key=lambda s: s.created_at)` for selection
- Created separate wrapper functions for parameterless routes
- Resolved FastAPI route precedence issues

**Route Structure**:
```python
# Specific routes defined BEFORE parameterized routes
@router.get("/metrics/latest")  # Must come first
async def get_latest_metrics_no_session(): ...

@router.get("/metrics/history/{metric_type}/{metric_name}")
async def get_metrics_history_no_session(...): ...

@router.get("/metrics")
async def get_metrics_no_session(): ...

@router.get("/metrics/{session_id}")  # Parameterized comes last
async def get_metrics(session_id: str): ...
```

**Error Handling**:
- Returns 404 "No sessions found" when no sessions exist
- Returns 404 "Session not found: {id}" for invalid session_id
- Returns 500 for session manager initialization errors

**Tests**: 5 comprehensive tests
- Epochs endpoint defaults to latest
- Metrics endpoint defaults to latest
- Latest metrics endpoint defaults to latest
- History endpoint defaults to latest
- Proper error when no sessions exist

**Coverage Impact**:
- `epochs.py`: 94% coverage
- `metrics.py`: 93% coverage

---

### Technical Implementation Details

#### Architecture Decisions

1. **Separate Wrapper Functions vs Optional Parameters**
   - Initial approach: Made `session_id` optional with `session_id: str | None = None`
   - Problem: Main functions need validation logic, complicates error handling
   - Solution: Separate wrapper functions for default routes
   - Benefit: Clean separation, better route ordering control

2. **Route Ordering for FastAPI**
   - FastAPI matches routes in definition order
   - Specific routes (e.g., `/metrics/latest`) must precede parameterized routes (e.g., `/metrics/{session_id}`)
   - Without proper ordering, `/metrics/latest` would match with `session_id="latest"`

3. **Latest Session Selection**
   - Uses `created_at` timestamp for recency determination
   - `max(sessions, key=lambda s: s.created_at)` ensures consistent selection
   - Alternative considered: Using session ID ordering (rejected for unreliability)

4. **Error Message Consistency**
   - "No sessions found": When no sessions exist at all
   - "Session not found: {id}": When specific session_id is invalid
   - Helps distinguish between empty state vs invalid query

#### Code Structure

```
src/api/routers/
├── status.py          # System status endpoint
├── epochs.py          # Epoch progress tracking
└── metrics.py         # Metrics endpoints (all 3 variants)

tests/
├── test_status_endpoint.py           # 7 tests
├── test_epochs_endpoint.py           # 6 tests
├── test_metrics_endpoint.py          # 7 tests
├── test_latest_metrics_endpoint.py   # 7 tests
├── test_metrics_history_endpoint.py  # 9 tests
└── test_default_session.py           # 5 tests for default behavior
```

#### Dependencies

**External**:
- `fastapi`: Web framework
- `pydantic`: Response model validation
- `datetime`: Timestamp handling

**Internal**:
- `src.api.session_manager`: `get_session_manager()`, `TrainingSessionManager`
- `src.api.app`: Router registration

---

### Testing Strategy

#### Test Coverage
- **Total Tests**: 41 tests (36 original + 5 default behavior)
- **Overall Coverage**: 93-94% for API routers
- **Test Execution Time**: ~12 seconds for full suite

#### Test Patterns

1. **Happy Path Testing**
   - Valid session retrieval
   - Proper response structure
   - Data integrity

2. **Error Handling**
   - Invalid session IDs (404)
   - Session manager unavailability (500)
   - No sessions exist (404)
   - Invalid parameters (400)

3. **Edge Cases**
   - Empty metrics/epochs lists
   - Multiple sessions with same timestamps (uses time.sleep(1.1) in tests)
   - Session ordering verification

4. **Default Behavior Testing**
   - Latest session selection accuracy
   - Route precedence verification
   - Error message correctness

#### Test Utilities

```python
# Pattern for creating multiple sessions with distinct timestamps
@pytest.fixture
def create_multiple_sessions(tmp_path):
    manager = TrainingSessionManager(str(tmp_path))
    session1 = manager.create_session("experiment1", {...})
    time.sleep(1.1)  # Ensure different timestamps
    session2 = manager.create_session("experiment2", {...})
    return manager, session1, session2
```

---

### API Usage Examples

#### 1. Check System Status
```bash
curl http://localhost:8000/api/v1/status
```

Response:
```json
{
  "status": "running",
  "active_session_id": "abc123",
  "total_sessions": 3,
  "sessions": [
    {
      "session_id": "abc123",
      "experiment_name": "unet_baseline",
      "created_at": "2025-12-18T10:30:00",
      "status": "running"
    }
  ]
}
```

#### 2. Get Latest Session Epochs (No Session ID)
```bash
curl http://localhost:8000/api/v1/epochs
```

Response:
```json
{
  "session_id": "abc123",
  "experiment_name": "unet_baseline",
  "total_epochs": 100,
  "completed_epochs": 45,
  "current_epoch": 45,
  "epochs": [
    {
      "epoch": 1,
      "status": "completed",
      "train_metrics": {"loss": 0.523, "dice": 0.712},
      "val_metrics": {"loss": 0.498, "dice": 0.735}
    }
  ]
}
```

#### 3. Get Specific Session Metrics
```bash
curl http://localhost:8000/api/v1/metrics/abc123
```

#### 4. Get Latest Metrics for Dashboard
```bash
curl http://localhost:8000/api/v1/metrics/latest
```

Response:
```json
{
  "session_id": "abc123",
  "experiment_name": "unet_baseline",
  "train_metrics": {
    "loss": 0.234,
    "dice": 0.856
  },
  "val_metrics": {
    "loss": 0.212,
    "dice": 0.871
  },
  "timestamp": "2025-12-18T10:45:23"
}
```

#### 5. Get Metric History for Plotting
```bash
curl http://localhost:8000/api/v1/metrics/history/train/loss
```

Response:
```json
{
  "session_id": "abc123",
  "experiment_name": "unet_baseline",
  "metric_type": "train",
  "metric_name": "loss",
  "history": [
    {"epoch": 1, "value": 0.523, "timestamp": "2025-12-18T10:30:15"},
    {"epoch": 2, "value": 0.456, "timestamp": "2025-12-18T10:31:20"},
    {"epoch": 3, "value": 0.389, "timestamp": "2025-12-18T10:32:25"}
  ],
  "total_points": 3
}
```

---

### Integration Points

#### Current Integration
- **Session Manager**: Reads session metadata and status
- **FastAPI App**: Routes registered in `src/api/app.py`
- **Health Endpoint**: Exists from `feat/api-foundation` (Issue #82)

#### Future Integration (Planned)
- **Metrics Collector**: Will populate actual metrics data (currently returns empty)
- **Training Process Manager**: Will provide real-time training status
- **WebSocket Support**: For live metric streaming (Branch 28)
- **Training Control Endpoints**: Start/stop/pause training (Branch 26)
- **Graph Endpoints**: Formatted data for visualization (Branch 27)

---

### Known Limitations

1. **Empty Data Responses**
   - Metrics and epochs currently return empty lists
   - Will be populated when integrated with actual training process
   - Infrastructure is complete and tested

2. **No Pagination**
   - History endpoints return complete dataset
   - May need pagination for very long training runs
   - Consider implementing `limit` and `offset` parameters in future

3. **No Filtering**
   - Cannot filter metrics by epoch range or time period
   - Consider adding query parameters: `?start_epoch=10&end_epoch=50`

4. **No Aggregation**
   - No support for downsampling or aggregating metrics
   - Could be useful for very long training runs (1000+ epochs)

---

### Development Workflow

This branch followed the established workflow from `lessons.md`:

1. **Sequential Implementation**
   - Issue #85 → Implement → Test → Commit
   - Issue #86 → Implement → Test → Commit
   - Issue #87 → Implement → Test → Commit
   - Issue #88 → Implement → Test → Commit
   - Issue #89 → Implement → Test → Commit
   - Default behavior → Implement → Test → Commit

2. **Quality Checks Per Commit**
   - `uv run ruff check --fix`
   - `uv run mypy src/api/`
   - `uv run pytest tests/ -v`
   - Pre-commit hooks automatically run

3. **Test-Driven Development**
   - Tests written alongside implementation
   - Coverage verified before commit
   - Edge cases identified and tested

---

### Performance Considerations

#### Response Times
- Status endpoint: O(n) where n = number of sessions
- Epochs endpoint: O(1) session lookup + O(m) where m = number of epochs
- Metrics endpoint: O(1) session lookup + O(k) where k = number of metrics
- History endpoint: O(1) session lookup + O(p) where p = data points

#### Memory Usage
- Session metadata stored in JSON files (lightweight)
- No in-memory caching currently implemented
- Consider caching for high-frequency requests

#### Scalability
- File-based session storage suitable for single-machine training
- For distributed training, consider database backend
- Current implementation supports up to ~1000 sessions efficiently

---

### Future Enhancements

#### Short-term (Next Branches)
1. Training control endpoints (Branch 26)
2. Graph data endpoints (Branch 27)
3. WebSocket streaming (Branch 28)

#### Medium-term
1. Metric aggregation and downsampling
2. Time-range filtering for history
3. Pagination for large datasets
4. Caching layer for frequently accessed data

#### Long-term
1. Database backend for session storage
2. Multi-node training support
3. Authentication and authorization
4. Rate limiting and request throttling

---

### Commit History

```
1afd288 - feat(api): add default to latest session for all endpoints
964603b - feat(api): add GET /metrics/{session_id}/history/{metric_type}/{metric_name} endpoint
979bb2e - feat(api): add GET /metrics/{session_id}/latest endpoint
c4c1412 - feat(api): add GET /metrics/{session_id} endpoint
43e06e1 - feat(api): add GET /epochs/{session_id} endpoint
85b8693 - feat(api): add GET /status endpoint
```

**Total Changes**:
- 11 files changed
- 1,731 insertions
- 3 deletions
- 5 new source files
- 6 new test files

---

### References

- **Branch Documentation**: `docs/branch-task-mapping.md` (Branch 25)
- **API Foundation**: Branch 24 (`feat/api-foundation`)
- **Issues**: #85, #86, #87, #88, #89
- **Related Branches**:
  - Branch 26: `feat/api-training-control`
  - Branch 27: `feat/api-graphs`
  - Branch 28: `feat/api-websocket`

---

## Related Documentation

- [Configuration Guide](./CONFIGURATION.md)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [Docker Setup](./DOCKER.md)
- [Branch-Task Mapping](./branch-task-mapping.md)
