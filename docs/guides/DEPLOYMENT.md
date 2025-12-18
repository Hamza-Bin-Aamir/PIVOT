# Deployment Guide

Comprehensive guide for deploying PIVOT in production environments.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Best Practices](#production-best-practices)
- [Monitoring](#monitoring)
- [Security](#security)
- [Backup and Recovery](#backup-and-recovery)

## Overview

PIVOT can be deployed in several configurations:
- **Single Server**: Training and API on one machine
- **Distributed**: Separate training and API servers
- **Containerized**: Docker/Kubernetes deployment
- **Cloud**: AWS, GCP, or Azure deployment

## System Requirements

### Minimum Requirements

**Training Server:**
- CPU: 16+ cores
- RAM: 32GB minimum, 64GB recommended
- GPU: NVIDIA GPU with 12GB+ VRAM (RTX 3090, A5000, or better)
- Storage: 500GB SSD (1TB+ for large datasets)
- OS: Ubuntu 22.04 LTS, Rocky Linux 8+, or Windows Server 2019+

**API Server:**
- CPU: 8+ cores
- RAM: 16GB minimum
- GPU: Optional (for inference)
- Storage: 100GB SSD
- OS: Ubuntu 22.04 LTS or Rocky Linux 8+

### Software Requirements

- Python 3.11 or 3.12
- CUDA 12.1+ (for GPU support)
- Docker 24.0+ (for containerized deployment)
- PostgreSQL 15+ (for production database)
- Nginx 1.24+ (for reverse proxy)

## Installation

### Production Installation

```bash
# 1. Clone repository
git clone https://github.com/Hamza-Bin-Aamir/PIVOT.git
cd PIVOT

# 2. Create production environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install uv
uv sync --extra prod

# 4. Setup database
createdb pivot_production
uv run alembic upgrade head

# 5. Create configuration
cp configs/production.yaml.example configs/production.yaml
# Edit configs/production.yaml with your settings

# 6. Run migrations
uv run python scripts/setup_production.py
```

### System Service Setup

Create systemd service file `/etc/systemd/system/pivot-api.service`:

```ini
[Unit]
Description=PIVOT Training API
After=network.target postgresql.service

[Service]
Type=notify
User=pivot
Group=pivot
WorkingDirectory=/opt/pivot
Environment="PATH=/opt/pivot/venv/bin"
ExecStart=/opt/pivot/venv/bin/uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config logging.conf

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl enable pivot-api
sudo systemctl start pivot-api
sudo systemctl status pivot-api
```

## Configuration

### Production Configuration

`configs/production.yaml`:

```yaml
# Database (PostgreSQL)
database_url: postgresql://pivot:password@localhost:5432/pivot_production

# API Server
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false
  log_level: info

# Security
cors:
  enabled: true
  allow_origins:
    - https://pivot.example.com
    - https://dashboard.example.com
  allow_credentials: true

# Logging
logging:
  log_dir: /var/log/pivot
  log_level: INFO
  max_bytes: 104857600  # 100MB
  backup_count: 10

# Checkpoints
checkpoints:
  storage_path: /data/pivot/checkpoints
  max_checkpoints: 50
  cleanup_interval_hours: 24

# Monitoring
monitoring:
  enabled: true
  prometheus_port: 9090
  metrics_interval_seconds: 60

# Alerting
alerting:
  enabled: true
  smtp_host: smtp.example.com
  smtp_port: 587
  from_email: alerts@example.com
  admin_emails:
    - admin@example.com
```

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://pivot:password@db:5432/pivot_production
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# API
API_SECRET_KEY=your-secret-key-here
API_DEBUG=false
API_LOG_LEVEL=info

# Storage
CHECKPOINT_STORAGE_PATH=/data/checkpoints
DATA_STORAGE_PATH=/data/datasets

# Security
ALLOWED_HOSTS=pivot.example.com,api.pivot.example.com
CORS_ORIGINS=https://pivot.example.com

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Email
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=alerts@example.com
SMTP_PASSWORD=your-smtp-password
```

## Docker Deployment

### Docker Compose

`docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: pivot_production
      POSTGRES_USER: pivot
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - pivot-network
    restart: always

  # Redis (for caching and task queue)
  redis:
    image: redis:7-alpine
    networks:
      - pivot-network
    restart: always

  # API Server
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://pivot:${DB_PASSWORD}@db:5432/pivot_production
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ./configs:/app/configs:ro
      - checkpoint_data:/data/checkpoints
      - log_data:/var/log/pivot
    depends_on:
      - db
      - redis
    networks:
      - pivot-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # Training Worker
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      DATABASE_URL: postgresql://pivot:${DB_PASSWORD}@db:5432/pivot_production
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ./configs:/app/configs:ro
      - ./data:/app/data:ro
      - checkpoint_data:/data/checkpoints
      - log_data:/var/log/pivot
    depends_on:
      - db
      - redis
    networks:
      - pivot-network
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static_files:/usr/share/nginx/html
    depends_on:
      - api
    networks:
      - pivot-network
    restart: always

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - pivot-network
    restart: always

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    networks:
      - pivot-network
    restart: always

volumes:
  postgres_data:
  checkpoint_data:
  log_data:
  static_files:
  prometheus_data:
  grafana_data:

networks:
  pivot-network:
    driver: bridge
```

### Deploy with Docker

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f api

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=4
```

### Nginx Configuration

`nginx/nginx.conf`:

```nginx
upstream api_backend {
    least_conn;
    server api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name pivot.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name pivot.example.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API Proxy
    location /api/ {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;

        # File upload size
        client_max_body_size 500M;
    }

    # Static files
    location /static/ {
        alias /usr/share/nginx/html/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Dashboard
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

## Kubernetes Deployment

### Kubernetes Manifests

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pivot-api
  namespace: pivot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pivot-api
  template:
    metadata:
      labels:
        app: pivot-api
    spec:
      containers:
      - name: api
        image: pivot/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pivot-secrets
              key: database-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: pivot-api-service
  namespace: pivot
spec:
  selector:
    app: pivot-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: pivot-worker
  namespace: pivot
spec:
  serviceName: pivot-worker
  replicas: 2
  selector:
    matchLabels:
      app: pivot-worker
  template:
    metadata:
      labels:
        app: pivot-worker
    spec:
      containers:
      - name: worker
        image: pivot/worker:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pivot-secrets
              key: database-url
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: checkpoint-storage
          mountPath: /data/checkpoints
  volumeClaimTemplates:
  - metadata:
      name: checkpoint-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace pivot

# Create secrets
kubectl create secret generic pivot-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --namespace=pivot

# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n pivot
kubectl logs -f deployment/pivot-api -n pivot

# Scale deployment
kubectl scale deployment pivot-api --replicas=5 -n pivot
```

## Production Best Practices

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_sessions_status ON training_sessions(status);
CREATE INDEX idx_sessions_created_at ON training_sessions(created_at DESC);
CREATE INDEX idx_metrics_session_timestamp ON metrics(session_id, timestamp DESC);
CREATE INDEX idx_epochs_session_epoch ON epochs(session_id, epoch);

-- Enable query optimization
ALTER TABLE training_sessions SET (autovacuum_enabled = true);
ALTER TABLE metrics SET (autovacuum_enabled = true);

-- Setup connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### Logging Configuration

`logging.conf`:

```ini
[loggers]
keys=root,api,training

[handlers]
keys=console,file,error

[formatters]
keys=detailed,simple

[logger_root]
level=INFO
handlers=console,file

[logger_api]
level=INFO
handlers=file,error
qualname=api
propagate=0

[logger_training]
level=INFO
handlers=file
qualname=training
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=INFO
formatter=detailed
args=('/var/log/pivot/app.log', 'a', 104857600, 10)

[handler_error]
class=handlers.RotatingFileHandler
level=ERROR
formatter=detailed
args=('/var/log/pivot/error.log', 'a', 104857600, 10)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
```

### Performance Tuning

```yaml
# API Server
workers: 4  # 2-4x number of CPU cores
worker_class: uvicorn.workers.UvicornWorker
worker_connections: 1000
keepalive: 5
timeout: 300

# Database
pool_size: 20
max_overflow: 10
pool_pre_ping: true
pool_recycle: 3600

# Caching
redis_url: redis://localhost:6379/0
cache_ttl: 300  # 5 minutes

# Rate Limiting
rate_limit_per_minute: 100
rate_limit_per_hour: 1000
```

## Monitoring

### Prometheus Metrics

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pivot-api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Grafana Dashboards

Import pre-built dashboards for monitoring:
- System metrics (CPU, memory, disk)
- API performance (request rate, latency, errors)
- Training progress (epochs, loss, accuracy)
- GPU utilization

## Security

### SSL/TLS Setup

```bash
# Generate certificates with Let's Encrypt
certbot certonly --standalone \
  -d pivot.example.com \
  -d api.pivot.example.com \
  --email admin@example.com \
  --agree-tos

# Auto-renewal
echo "0 0 * * * certbot renew --quiet" | crontab -
```

### Firewall Configuration

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow API port (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 8000

# Enable firewall
sudo ufw enable
```

### API Authentication

Implement API key authentication:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_SECRET_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

## Backup and Recovery

### Database Backups

```bash
#!/bin/bash
# backup.sh - Daily database backup script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/pivot"
DB_NAME="pivot_production"

# Create backup
pg_dump $DB_NAME | gzip > "$BACKUP_DIR/pivot_$DATE.sql.gz"

# Keep only last 30 days
find $BACKUP_DIR -name "pivot_*.sql.gz" -mtime +30 -delete

# Upload to S3
aws s3 sync $BACKUP_DIR s3://pivot-backups/database/
```

### Checkpoint Backups

```bash
#!/bin/bash
# Sync checkpoints to S3

aws s3 sync /data/checkpoints s3://pivot-backups/checkpoints/ \
  --exclude "*.tmp" \
  --delete
```

### Disaster Recovery

```bash
# Restore database
gunzip < pivot_20251219_120000.sql.gz | psql pivot_production

# Restore checkpoints
aws s3 sync s3://pivot-backups/checkpoints/ /data/checkpoints/

# Restart services
sudo systemctl restart pivot-api
sudo systemctl restart pivot-worker
```

## Troubleshooting

See logs:
```bash
# API logs
tail -f /var/log/pivot/app.log

# System logs
journalctl -u pivot-api -f

# Docker logs
docker-compose logs -f api
```

Check service status:
```bash
# Systemd
sudo systemctl status pivot-api

# Docker
docker-compose ps

# Kubernetes
kubectl get pods -n pivot
```

## Next Steps

- Setup monitoring dashboards in Grafana
- Configure automated backups
- Implement CI/CD pipeline
- Setup load balancing and auto-scaling
- Review [API Documentation](API_REFERENCE.md)
