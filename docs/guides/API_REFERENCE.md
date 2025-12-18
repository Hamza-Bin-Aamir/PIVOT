# API Documentation

Complete reference for the PIVOT Training API server.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDKs](#sdks)

## Overview

The PIVOT API provides RESTful endpoints for managing training sessions, monitoring progress, and running inference programmatically.

**Base URL:** `http://localhost:8000/api/v1`

**API Version:** 0.1.0

## Getting Started

### Start the API Server

```bash
# Development mode
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uv run uvicorn src.api.app:app --workers 4 --host 0.0.0.0 --port 8000
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-19T10:30:00Z",
  "uptime_seconds": 3600.5,
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.8,
    "disk_percent": 38.5
  },
  "components": {
    "database": "healthy",
    "filesystem": "healthy"
  }
}
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## Endpoints

### Training Sessions

#### Create Training Session

```http
POST /api/v1/training/sessions
```

**Request Body:**
```json
{
  "config_path": "configs/train.yaml",
  "session_name": "experiment_001",
  "tags": ["unet", "focal-loss"]
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "created",
  "created_at": "2025-12-19T10:30:00Z",
  "config": {
    "epochs": 100,
    "batch_size": 2,
    "learning_rate": 0.001
  }
}
```

#### Start Training

```http
POST /api/v1/training/sessions/{session_id}/start
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "running",
  "started_at": "2025-12-19T10:30:05Z"
}
```

#### Stop Training

```http
POST /api/v1/training/sessions/{session_id}/stop
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "stopped",
  "stopped_at": "2025-12-19T10:45:00Z"
}
```

#### Get Session Status

```http
GET /api/v1/training/sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "running",
  "created_at": "2025-12-19T10:30:00Z",
  "started_at": "2025-12-19T10:30:05Z",
  "current_epoch": 25,
  "total_epochs": 100,
  "progress": 0.25,
  "latest_metrics": {
    "train_loss": 0.245,
    "val_loss": 0.289,
    "learning_rate": 0.000853
  }
}
```

#### List Training Sessions

```http
GET /api/v1/training/sessions
```

**Query Parameters:**
- `limit` (int): Maximum number of sessions (default: 50)
- `offset` (int): Number of sessions to skip (default: 0)
- `status` (string): Filter by status (running, stopped, completed, failed)

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "sess_abc123",
      "session_name": "experiment_001",
      "status": "running",
      "created_at": "2025-12-19T10:30:00Z",
      "current_epoch": 25
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Metrics

#### Get Session Metrics

```http
GET /api/v1/training/sessions/{session_id}/metrics
```

**Query Parameters:**
- `metric_name` (string): Filter by metric name
- `limit` (int): Maximum number of records (default: 1000)

**Response:**
```json
{
  "session_id": "sess_abc123",
  "metrics": [
    {
      "epoch": 1,
      "timestamp": "2025-12-19T10:31:00Z",
      "train_loss": 0.856,
      "val_loss": 0.912,
      "train_accuracy": 0.634,
      "val_accuracy": 0.598,
      "learning_rate": 0.001
    },
    {
      "epoch": 2,
      "timestamp": "2025-12-19T10:32:00Z",
      "train_loss": 0.734,
      "val_loss": 0.801,
      "train_accuracy": 0.701,
      "val_accuracy": 0.667,
      "learning_rate": 0.00095
    }
  ]
}
```

#### Get Metric History

```http
GET /api/v1/metrics/{session_id}/history
```

**Query Parameters:**
- `metric_name` (string): Specific metric to retrieve
- `start_epoch` (int): Starting epoch
- `end_epoch` (int): Ending epoch

**Response:**
```json
{
  "metric_name": "val_loss",
  "values": [0.912, 0.801, 0.723, 0.656],
  "epochs": [1, 2, 3, 4],
  "timestamps": ["2025-12-19T10:31:00Z", "2025-12-19T10:32:00Z", ...]
}
```

### Checkpoints

#### List Checkpoints

```http
GET /api/v1/checkpoints
```

**Query Parameters:**
- `session_id` (string): Filter by session
- `limit` (int): Maximum number of checkpoints

**Response:**
```json
{
  "checkpoints": [
    {
      "checkpoint_id": "ckpt_001",
      "session_id": "sess_abc123",
      "epoch": 50,
      "metric_value": 0.856,
      "metric_name": "val_froc",
      "file_size_mb": 245.3,
      "created_at": "2025-12-19T11:00:00Z",
      "is_best": true
    }
  ]
}
```

#### Upload Checkpoint

```http
POST /api/v1/checkpoints
```

**Request (multipart/form-data):**
- `file`: Checkpoint file (.pth)
- `session_id`: Associated session
- `epoch`: Epoch number
- `metrics`: JSON string of metrics

**Response:**
```json
{
  "checkpoint_id": "ckpt_002",
  "file_size_mb": 245.3,
  "uploaded_at": "2025-12-19T11:05:00Z"
}
```

#### Download Checkpoint

```http
GET /api/v1/checkpoints/{checkpoint_id}/download
```

**Response:** Binary file download

#### Delete Checkpoint

```http
DELETE /api/v1/checkpoints/{checkpoint_id}
```

**Response:**
```json
{
  "message": "Checkpoint deleted successfully"
}
```

### Monitoring

#### Get System Resources

```http
GET /api/v1/monitor/resources
```

**Response:**
```json
{
  "timestamp": "2025-12-19T10:30:00Z",
  "cpu": {
    "percent": 45.2,
    "count": 16,
    "per_cpu": [42.1, 48.3, 43.5, ...]
  },
  "memory": {
    "percent": 62.8,
    "used_gb": 50.2,
    "total_gb": 80.0
  },
  "disk": {
    "percent": 38.5,
    "used_gb": 385.0,
    "total_gb": 1000.0
  },
  "gpu": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "memory_percent": 75.2,
      "memory_used_gb": 18.0,
      "memory_total_gb": 24.0,
      "utilization": 92.5,
      "temperature": 68.0
    }
  ]
}
```

### Alerts

#### Get Active Alerts

```http
GET /api/v1/alerts
```

**Query Parameters:**
- `status` (string): active, acknowledged, resolved
- `severity` (string): low, medium, high, critical

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "alert_001",
      "rule_name": "high_cpu_usage",
      "severity": "high",
      "message": "CPU usage above 90%",
      "value": 94.5,
      "threshold": 90.0,
      "triggered_at": "2025-12-19T10:28:00Z",
      "status": "active"
    }
  ]
}
```

#### Acknowledge Alert

```http
POST /api/v1/alerts/{alert_id}/acknowledge
```

**Request Body:**
```json
{
  "acknowledged_by": "user@example.com",
  "notes": "Investigating high CPU usage"
}
```

#### Create Alert Rule

```http
POST /api/v1/alerts/rules
```

**Request Body:**
```json
{
  "name": "high_memory_usage",
  "metric": "memory_percent",
  "condition": "greater_than",
  "threshold": 85.0,
  "severity": "high",
  "enabled": true
}
```

### Webhooks

#### Register Webhook

```http
POST /api/v1/webhooks
```

**Request Body:**
```json
{
  "url": "https://example.com/webhook",
  "events": ["training_completed", "alert_triggered"],
  "secret": "your_webhook_secret"
}
```

**Response:**
```json
{
  "webhook_id": "webhook_001",
  "url": "https://example.com/webhook",
  "created_at": "2025-12-19T10:30:00Z",
  "status": "active"
}
```

#### Test Webhook

```http
POST /api/v1/webhooks/{webhook_id}/test
```

**Response:**
```json
{
  "success": true,
  "status_code": 200,
  "response_time_ms": 145
}
```

### Real-Time Updates

#### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/training/{session_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Epoch:', data.epoch);
  console.log('Loss:', data.loss);
};
```

#### Server-Sent Events (SSE)

```javascript
const eventSource = new EventSource('http://localhost:8000/api/v1/sse/training/{session_id}');

eventSource.addEventListener('metrics', (event) => {
  const data = JSON.parse(event.data);
  console.log('New metrics:', data);
});
```

### Notifications

#### Get Notifications

```http
GET /api/v1/notifications
```

**Query Parameters:**
- `type` (string): training, alert, system
- `priority` (string): low, medium, high
- `unread` (boolean): Filter unread notifications

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_001",
      "type": "training",
      "priority": "high",
      "message": "Training session sess_abc123 completed",
      "data": {
        "session_id": "sess_abc123",
        "final_metrics": {
          "val_froc": 0.856
        }
      },
      "read": false,
      "created_at": "2025-12-19T12:00:00Z"
    }
  ]
}
```

#### Mark as Read

```http
POST /api/v1/notifications/{notification_id}/read
```

#### Clear All Notifications

```http
DELETE /api/v1/notifications
```

## Data Models

### TrainingSession

```typescript
interface TrainingSession {
  session_id: string;
  session_name?: string;
  status: 'created' | 'running' | 'stopped' | 'completed' | 'failed';
  config: object;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  current_epoch?: number;
  total_epochs: number;
  tags?: string[];
}
```

### Metrics

```typescript
interface Metrics {
  epoch: number;
  timestamp: string;
  train_loss: number;
  val_loss: number;
  train_accuracy?: number;
  val_accuracy?: number;
  learning_rate: number;
  [key: string]: any;
}
```

### Checkpoint

```typescript
interface Checkpoint {
  checkpoint_id: string;
  session_id: string;
  epoch: number;
  metric_name: string;
  metric_value: number;
  file_size_mb: number;
  file_path: string;
  is_best: boolean;
  created_at: string;
}
```

### Alert

```typescript
interface Alert {
  alert_id: string;
  rule_name: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  value: number;
  threshold: number;
  triggered_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  status: 'active' | 'acknowledged' | 'resolved';
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid session ID format",
    "details": {
      "field": "session_id",
      "expected": "string matching ^sess_[a-z0-9]+$"
    }
  }
}
```

### Common Error Codes

- `400` - Bad Request (invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `409` - Conflict (session already running)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

## Rate Limiting

**Current limits:**
- 100 requests per minute per IP
- 1000 requests per hour per IP

**Rate limit headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1639920000
```

## SDKs

### Python Client

```python
from src.api.client import PIVOTClient

# Initialize client
client = PIVOTClient(base_url='http://localhost:8000')

# Create session
session = client.create_session(
    config_path='configs/train.yaml',
    session_name='my_experiment'
)

# Start training
client.start_training(session['session_id'])

# Monitor progress
for metrics in client.stream_metrics(session['session_id']):
    print(f"Epoch {metrics['epoch']}: Loss {metrics['val_loss']:.3f}")

# Wait for completion
final_metrics = client.wait_for_completion(session['session_id'])
```

### JavaScript/TypeScript Client

```typescript
import { PIVOTClient } from './pivot-client';

const client = new PIVOTClient('http://localhost:8000');

// Create and start session
const session = await client.createSession({
  config_path: 'configs/train.yaml',
  session_name: 'my_experiment'
});

await client.startTraining(session.session_id);

// Stream updates via WebSocket
client.streamMetrics(session.session_id, (metrics) => {
  console.log(`Epoch ${metrics.epoch}: Loss ${metrics.val_loss}`);
});
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

```
http://localhost:8000/api/v1/openapi.json
```

Interactive API documentation (Swagger UI):

```
http://localhost:8000/docs
```

ReDoc documentation:

```
http://localhost:8000/redoc
```

## Next Steps

- Review [API Usage Guide](API_USAGE.md) for practical examples
- Check [Deployment Guide](DEPLOYMENT.md) for production setup
- Explore interactive docs at `/docs`
