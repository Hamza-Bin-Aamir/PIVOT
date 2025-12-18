# OpenAPI Documentation

Guide to accessing and using PIVOT's automatically generated OpenAPI specification.

## Overview

PIVOT uses **FastAPI**, which automatically generates OpenAPI 3.0 specification for all API endpoints. This provides:

- **Interactive documentation** with Swagger UI
- **ReDoc documentation** with beautiful UI
- **OpenAPI JSON schema** for client generation
- **Type-safe client SDKs** in multiple languages

## Accessing Documentation

### Interactive API Documentation

PIVOT provides three ways to explore the API:

#### 1. Swagger UI (Interactive)

**URL:** `http://localhost:8000/docs`

Features:
- Try out API endpoints directly in the browser
- View request/response schemas
- See example requests and responses
- Test authentication
- Download OpenAPI schema

Example:
```bash
# Start API server
uv run uvicorn src.api.app:app --reload

# Open browser
open http://localhost:8000/docs
```

#### 2. ReDoc (Read-Only)

**URL:** `http://localhost:8000/redoc`

Features:
- Clean, read-only documentation
- Better for browsing and learning
- Three-panel layout (navigation, content, examples)
- Search functionality
- Download OpenAPI schema

Example:
```bash
# Open ReDoc
open http://localhost:8000/redoc
```

#### 3. OpenAPI JSON Schema

**URL:** `http://localhost:8000/openapi.json`

Features:
- Raw OpenAPI 3.0 specification in JSON format
- Use for client generation
- Import into Postman/Insomnia
- Generate SDKs with OpenAPI Generator

Example:
```bash
# Download schema
curl http://localhost:8000/openapi.json > openapi.json

# Pretty print
curl http://localhost:8000/openapi.json | jq . > openapi.json
```

## OpenAPI Schema Structure

The OpenAPI specification includes:

### API Information

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "PIVOT Training API",
    "description": "API for managing pulmonary nodule detection training",
    "version": "0.1.0"
  },
  "servers": [
    {
      "url": "http://localhost:8000",
      "description": "Development server"
    }
  ]
}
```

### Endpoints

All API endpoints are documented with:
- **Path**: URL pattern with parameters
- **Method**: HTTP method (GET, POST, PUT, DELETE)
- **Parameters**: Query, path, and header parameters
- **Request Body**: Schema for POST/PUT requests
- **Responses**: All possible response codes and schemas
- **Tags**: Endpoint categories

Example endpoint:

```json
{
  "/api/v1/sessions": {
    "post": {
      "tags": ["Training Sessions"],
      "summary": "Create Training Session",
      "operationId": "create_session_api_v1_sessions_post",
      "requestBody": {
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/CreateSessionRequest"
            }
          }
        }
      },
      "responses": {
        "200": {
          "description": "Successful Response",
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/TrainingSession"
              }
            }
          }
        }
      }
    }
  }
}
```

### Data Models

All request/response models are defined:

```json
{
  "components": {
    "schemas": {
      "TrainingSession": {
        "type": "object",
        "properties": {
          "session_id": {"type": "string", "format": "uuid"},
          "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
          "config": {"type": "object"},
          "created_at": {"type": "string", "format": "date-time"},
          "updated_at": {"type": "string", "format": "date-time"}
        },
        "required": ["session_id", "status", "config"]
      }
    }
  }
}
```

## Generating Client SDKs

### Python Client

Using `openapi-python-client`:

```bash
# Install generator
pip install openapi-python-client

# Generate client
openapi-python-client generate \
  --url http://localhost:8000/openapi.json \
  --output-path ./pivot-client

# Use generated client
from pivot_client import Client
from pivot_client.models import CreateSessionRequest

client = Client(base_url="http://localhost:8000")

# Create session
request = CreateSessionRequest(
    config={"data_dir": "/path/to/data"}
)
session = client.sessions.create_session(json_body=request)
print(f"Created session: {session.session_id}")
```

### TypeScript Client

Using `openapi-typescript-codegen`:

```bash
# Install generator
npm install -g openapi-typescript-codegen

# Generate client
openapi-typescript-codegen \
  --input http://localhost:8000/openapi.json \
  --output ./src/api-client \
  --client axios

# Use generated client
import { SessionsService } from './api-client';

const session = await SessionsService.createSession({
  requestBody: {
    config: { data_dir: '/path/to/data' }
  }
});

console.log(`Created session: ${session.session_id}`);
```

### Other Languages

OpenAPI supports client generation for many languages:

```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate Java client
openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g java \
  -o ./java-client

# Generate Go client
openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g go \
  -o ./go-client

# Generate C# client
openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g csharp \
  -o ./csharp-client
```

## Importing into API Tools

### Postman

1. Open Postman
2. Click **Import**
3. Select **Link** tab
4. Enter: `http://localhost:8000/openapi.json`
5. Click **Continue** → **Import**

All endpoints will be added to a new collection.

### Insomnia

1. Open Insomnia
2. Click **Create** → **Import From** → **URL**
3. Enter: `http://localhost:8000/openapi.json`
4. Click **Fetch and Import**

### Bruno

1. Open Bruno
2. Click **Import Collection**
3. Select **OpenAPI**
4. Enter URL: `http://localhost:8000/openapi.json`

## Customizing OpenAPI Metadata

### Update API Information

Edit `src/api/app.py`:

```python
from fastapi import FastAPI

app = FastAPI(
    title="PIVOT Training API",
    description="""
    API for managing pulmonary nodule detection model training.

    ## Features

    * **Training Sessions**: Create and manage training jobs
    * **Real-time Updates**: WebSocket and SSE support
    * **Monitoring**: System metrics and alerts
    * **Checkpoints**: Upload and download model checkpoints
    """,
    version="1.0.0",
    contact={
        "name": "PIVOT Team",
        "email": "support@pivot.example.com",
        "url": "https://pivot.example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.pivot.example.com",
            "description": "Production server"
        }
    ]
)
```

### Add Tags and Descriptions

Group endpoints with tags:

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["Training Sessions"],
    responses={
        404: {"description": "Session not found"},
        500: {"description": "Internal server error"}
    }
)

@router.post("/", response_model=TrainingSession)
async def create_session(
    request: CreateSessionRequest
) -> TrainingSession:
    """
    Create a new training session.

    - **config**: Training configuration (model, data, hyperparameters)

    Returns the created session with a unique session_id.
    """
    # Implementation
```

### Add Examples

Provide example requests/responses:

```python
from pydantic import BaseModel, Field

class CreateSessionRequest(BaseModel):
    config: dict = Field(
        ...,
        description="Training configuration",
        example={
            "data_dir": "/data/luna16",
            "model": {
                "architecture": "unet",
                "in_channels": 1,
                "out_channels": 1
            },
            "training": {
                "batch_size": 4,
                "num_epochs": 100,
                "learning_rate": 0.001
            }
        }
    )
```

## Validation with OpenAPI Tools

### Validate Schema

```bash
# Install validator
npm install -g @apidevtools/swagger-cli

# Validate schema
swagger-cli validate http://localhost:8000/openapi.json
```

### Lint OpenAPI Spec

```bash
# Install linter
npm install -g @stoplight/spectral-cli

# Lint schema
spectral lint http://localhost:8000/openapi.json
```

## API Versioning

PIVOT uses URL versioning:

```
/api/v1/sessions     # Version 1
/api/v2/sessions     # Version 2 (future)
```

Each version has its own OpenAPI schema:

```
/api/v1/openapi.json
/api/v2/openapi.json
```

## Best Practices

1. **Always use generated clients** instead of manual HTTP requests
2. **Version your API** to maintain backward compatibility
3. **Document all endpoints** with clear descriptions and examples
4. **Validate requests** using Pydantic models
5. **Use tags** to organize endpoints logically
6. **Provide examples** for complex request bodies
7. **Document error responses** with proper status codes
8. **Keep schemas up-to-date** by regenerating clients after API changes

## Troubleshooting

### Schema Not Loading

```bash
# Check API server is running
curl http://localhost:8000/health

# Verify OpenAPI endpoint
curl http://localhost:8000/openapi.json | jq .
```

### Client Generation Fails

```bash
# Download schema locally first
curl http://localhost:8000/openapi.json > openapi.json

# Generate from file
openapi-python-client generate --path openapi.json
```

### Missing Endpoints

Ensure routers are included in main app:

```python
from src.api.routers import sessions, metrics, checkpoints

app.include_router(sessions.router)
app.include_router(metrics.router)
app.include_router(checkpoints.router)
```

## Next Steps

- Explore [API Reference](API_REFERENCE.md) for endpoint details
- Read [API Usage Guide](API_USAGE.md) for practical examples
- Review [Deployment Guide](DEPLOYMENT.md) for production setup
- Generate a client SDK for your preferred language
- Import into Postman/Insomnia for manual testing
