# API Usage Guide

Practical examples and tutorials for using the PIVOT Training API.

## Table of Contents

- [Quick Start](#quick-start)
- [Training Workflows](#training-workflows)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Checkpoint Management](#checkpoint-management)
- [Real-Time Updates](#real-time-updates)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

## Quick Start

### Install Dependencies

```bash
# Server-side
uv add fastapi uvicorn sqlalchemy psutil httpx

# Client-side (Python)
pip install requests websockets
```

### Start API Server

```bash
# Development
uv run uvicorn src.api.app:app --reload --port 8000

# Production
uv run uvicorn src.api.app:app --workers 4 --port 8000
```

### Basic Request

```python
import requests

# Health check
response = requests.get('http://localhost:8000/api/v1/health')
print(response.json())
```

## Training Workflows

### Complete Training Session

```python
import requests
import time

BASE_URL = 'http://localhost:8000/api/v1'

# 1. Create session
session_data = {
    'config_path': 'configs/train.yaml',
    'session_name': 'experiment_001',
    'tags': ['unet', 'focal-loss']
}

response = requests.post(f'{BASE_URL}/training/sessions', json=session_data)
session = response.json()
session_id = session['session_id']
print(f"Created session: {session_id}")

# 2. Start training
response = requests.post(f'{BASE_URL}/training/sessions/{session_id}/start')
print(f"Training started: {response.json()}")

# 3. Monitor progress
while True:
    response = requests.get(f'{BASE_URL}/training/sessions/{session_id}')
    status = response.json()

    print(f"Epoch {status['current_epoch']}/{status['total_epochs']}")
    print(f"Status: {status['status']}")

    if status['status'] in ['completed', 'failed', 'stopped']:
        break

    time.sleep(10)

# 4. Get final metrics
response = requests.get(f'{BASE_URL}/training/sessions/{session_id}/metrics')
metrics = response.json()
print(f"Final validation loss: {metrics['metrics'][-1]['val_loss']}")
```

### Resume Training from Checkpoint

```python
# Create session with checkpoint
session_data = {
    'config_path': 'configs/train.yaml',
    'session_name': 'resume_experiment',
    'checkpoint_id': 'ckpt_abc123'  # Resume from this checkpoint
}

response = requests.post(f'{BASE_URL}/training/sessions', json=session_data)
session = response.json()

# Start training
requests.post(f'{BASE_URL}/training/sessions/{session['session_id']}/start')
```

### Stop Training Gracefully

```python
# Stop training (saves checkpoint before stopping)
response = requests.post(f'{BASE_URL}/training/sessions/{session_id}/stop')
print(response.json())

# Check final status
response = requests.get(f'{BASE_URL}/training/sessions/{session_id}')
status = response.json()
print(f"Stopped at epoch: {status['current_epoch']}")
```

## Monitoring and Debugging

### Real-Time Metrics Polling

```python
import requests
import time

def monitor_training(session_id, interval=10):
    """Poll for training updates."""
    while True:
        # Get current status
        response = requests.get(f'{BASE_URL}/training/sessions/{session_id}')
        status = response.json()

        # Get latest metrics
        response = requests.get(
            f'{BASE_URL}/training/sessions/{session_id}/metrics',
            params={'limit': 1}
        )
        metrics = response.json()['metrics']

        if metrics:
            latest = metrics[-1]
            print(f"Epoch {latest['epoch']} | "
                  f"Train Loss: {latest['train_loss']:.4f} | "
                  f"Val Loss: {latest['val_loss']:.4f} | "
                  f"LR: {latest['learning_rate']:.6f}")

        if status['status'] in ['completed', 'failed', 'stopped']:
            print(f"Training {status['status']}")
            break

        time.sleep(interval)

# Usage
monitor_training('sess_abc123')
```

### Get Metric History

```python
def plot_training_curve(session_id, metric_name='val_loss'):
    """Fetch and plot metric history."""
    import matplotlib.pyplot as plt

    response = requests.get(
        f'{BASE_URL}/metrics/{session_id}/history',
        params={'metric_name': metric_name}
    )
    data = response.json()

    plt.plot(data['epochs'], data['values'])
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training Curve: {metric_name}')
    plt.grid(True)
    plt.savefig(f'{metric_name}_curve.png')
    print(f"Saved plot to {metric_name}_curve.png")

# Usage
plot_training_curve('sess_abc123', 'val_loss')
plot_training_curve('sess_abc123', 'val_froc')
```

### System Resource Monitoring

```python
def check_system_resources():
    """Check if system has enough resources."""
    response = requests.get(f'{BASE_URL}/monitor/resources')
    resources = response.json()

    # Check CPU
    if resources['cpu']['percent'] > 90:
        print("⚠️  Warning: High CPU usage")

    # Check memory
    if resources['memory']['percent'] > 85:
        print("⚠️  Warning: High memory usage")

    # Check GPU
    if 'gpu' in resources:
        for gpu in resources['gpu']:
            if gpu['memory_percent'] > 90:
                print(f"⚠️  Warning: GPU {gpu['id']} memory full")

    print(f"✓ CPU: {resources['cpu']['percent']:.1f}%")
    print(f"✓ Memory: {resources['memory']['percent']:.1f}%")

    return resources

# Check before starting training
check_system_resources()
```

## Checkpoint Management

### Upload Checkpoint

```python
def upload_checkpoint(checkpoint_path, session_id, epoch, metrics):
    """Upload a checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        files = {'file': f}
        data = {
            'session_id': session_id,
            'epoch': epoch,
            'metrics': json.dumps(metrics)
        }

        response = requests.post(
            f'{BASE_URL}/checkpoints',
            files=files,
            data=data
        )

    return response.json()

# Usage
checkpoint_info = upload_checkpoint(
    checkpoint_path='checkpoints/model_epoch_50.pth',
    session_id='sess_abc123',
    epoch=50,
    metrics={'val_loss': 0.234, 'val_froc': 0.856}
)
print(f"Uploaded checkpoint: {checkpoint_info['checkpoint_id']}")
```

### Download Checkpoint

```python
def download_checkpoint(checkpoint_id, output_path):
    """Download a checkpoint file."""
    response = requests.get(
        f'{BASE_URL}/checkpoints/{checkpoint_id}/download',
        stream=True
    )

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {output_path}")

# Usage
download_checkpoint('ckpt_abc123', 'downloaded_model.pth')
```

### List and Compare Checkpoints

```python
def find_best_checkpoint(session_id, metric='val_froc', mode='max'):
    """Find best checkpoint by metric."""
    response = requests.get(
        f'{BASE_URL}/checkpoints',
        params={'session_id': session_id}
    )
    checkpoints = response.json()['checkpoints']

    if not checkpoints:
        return None

    # Sort by metric
    reverse = (mode == 'max')
    best = sorted(
        checkpoints,
        key=lambda x: x.get('metric_value', 0),
        reverse=reverse
    )[0]

    print(f"Best checkpoint: {best['checkpoint_id']}")
    print(f"Epoch: {best['epoch']}")
    print(f"{metric}: {best['metric_value']:.4f}")

    return best

# Usage
best = find_best_checkpoint('sess_abc123', 'val_froc', 'max')
```

## Real-Time Updates

### WebSocket Streaming

```python
import asyncio
import websockets
import json

async def stream_training_updates(session_id):
    """Stream real-time training updates via WebSocket."""
    uri = f"ws://localhost:8000/api/v1/ws/training/{session_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to training session: {session_id}")

        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'epoch_complete':
                print(f"Epoch {data['epoch']} completed")
                print(f"  Train Loss: {data['metrics']['train_loss']:.4f}")
                print(f"  Val Loss: {data['metrics']['val_loss']:.4f}")

            elif data['type'] == 'training_complete':
                print("Training completed!")
                print(f"Final metrics: {data['final_metrics']}")
                break

            elif data['type'] == 'error':
                print(f"Error: {data['message']}")
                break

# Usage
asyncio.run(stream_training_updates('sess_abc123'))
```

### Server-Sent Events (SSE)

```python
import requests

def stream_sse_updates(session_id):
    """Stream updates using Server-Sent Events."""
    url = f'{BASE_URL}/sse/training/{session_id}'

    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    print(f"Update: {data}")

# Usage
stream_sse_updates('sess_abc123')
```

## Integration Examples

### Jupyter Notebook Integration

```python
# In Jupyter notebook
from IPython.display import clear_output
import time

def monitor_in_notebook(session_id):
    """Monitor training in Jupyter notebook."""
    while True:
        response = requests.get(f'{BASE_URL}/training/sessions/{session_id}')
        status = response.json()

        # Clear previous output
        clear_output(wait=True)

        # Display progress
        print(f"Session: {status['session_name']}")
        print(f"Status: {status['status']}")
        print(f"Progress: {status['progress']*100:.1f}%")
        print(f"Epoch: {status['current_epoch']}/{status['total_epochs']}")

        if 'latest_metrics' in status:
            print("\nLatest Metrics:")
            for key, value in status['latest_metrics'].items():
                print(f"  {key}: {value:.4f}")

        if status['status'] in ['completed', 'failed', 'stopped']:
            break

        time.sleep(5)

monitor_in_notebook('sess_abc123')
```

### TensorBoard Integration

```python
def export_to_tensorboard(session_id, log_dir='logs/tensorboard'):
    """Export API metrics to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    # Get all metrics
    response = requests.get(f'{BASE_URL}/training/sessions/{session_id}/metrics')
    metrics = response.json()['metrics']

    # Write to TensorBoard
    writer = SummaryWriter(log_dir)
    for metric_data in metrics:
        epoch = metric_data['epoch']
        for key, value in metric_data.items():
            if key not in ['epoch', 'timestamp'] and isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch)

    writer.close()
    print(f"Exported to TensorBoard: {log_dir}")

# Usage
export_to_tensorboard('sess_abc123')
```

### Slack Notifications

```python
def send_slack_notification(webhook_url, message):
    """Send notification to Slack."""
    import json

    payload = {'text': message}
    requests.post(webhook_url, json=payload)

def monitor_and_notify(session_id, slack_webhook):
    """Monitor training and send Slack updates."""
    while True:
        response = requests.get(f'{BASE_URL}/training/sessions/{session_id}')
        status = response.json()

        if status['status'] == 'completed':
            message = f"✅ Training completed for {session_id}\n"
            message += f"Final metrics: {status['latest_metrics']}"
            send_slack_notification(slack_webhook, message)
            break

        elif status['status'] == 'failed':
            message = f"❌ Training failed for {session_id}"
            send_slack_notification(slack_webhook, message)
            break

        time.sleep(60)  # Check every minute

# Usage
monitor_and_notify('sess_abc123', 'https://hooks.slack.com/...')
```

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(to_email, subject, body):
    """Send email notification."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'pivot@example.com'
    msg['To'] = to_email

    with smtplib.SMTP('localhost') as server:
        server.send_message(msg)

def setup_alert_webhook(email):
    """Setup webhook for email alerts."""
    webhook_data = {
        'url': f'mailto:{email}',
        'events': ['training_completed', 'training_failed', 'alert_triggered']
    }

    response = requests.post(f'{BASE_URL}/webhooks', json=webhook_data)
    return response.json()

# Usage
webhook = setup_alert_webhook('user@example.com')
print(f"Webhook created: {webhook['webhook_id']}")
```

## Best Practices

### Error Handling

```python
def safe_api_call(method, url, **kwargs):
    """Make API call with error handling."""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code}")
        print(f"Details: {e.response.json()}")

    except requests.exceptions.ConnectionError:
        print("Failed to connect to API server")

    except requests.exceptions.Timeout:
        print("Request timed out")

    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

# Usage
session = safe_api_call('POST', f'{BASE_URL}/training/sessions', json=data)
```

### Retry Logic

```python
from time import sleep

def api_call_with_retry(method, url, max_retries=3, **kwargs):
    """API call with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Request failed, retrying in {wait_time}s...")
            sleep(wait_time)

# Usage
session = api_call_with_retry('POST', f'{BASE_URL}/training/sessions', json=data)
```

### Connection Pooling

```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retry():
    """Create requests session with retry and connection pooling."""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session

# Usage
session = create_session_with_retry()
response = session.get(f'{BASE_URL}/health')
```

### Rate Limiting Handling

```python
def handle_rate_limit(response):
    """Handle rate limit from API."""
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        print(f"Rate limited, waiting {retry_after}s...")
        time.sleep(retry_after)
        return True
    return False

def api_call_with_rate_limit(method, url, **kwargs):
    """API call with rate limit handling."""
    while True:
        response = requests.request(method, url, **kwargs)

        if not handle_rate_limit(response):
            response.raise_for_status()
            return response.json()
```

## Complete Example

Here's a complete example that combines multiple features:

```python
import requests
import time
import json
from pathlib import Path

class PIVOTTrainingClient:
    """High-level client for PIVOT API."""

    def __init__(self, base_url='http://localhost:8000/api/v1'):
        self.base_url = base_url
        self.session = requests.Session()

    def create_and_run_experiment(self, config_path, experiment_name):
        """Create and run a complete training experiment."""
        print(f"Starting experiment: {experiment_name}")

        # 1. Check system resources
        print("Checking system resources...")
        resources = self.get_resources()
        if resources['memory']['percent'] > 90:
            print("⚠️  Warning: Low memory available")

        # 2. Create session
        print("Creating training session...")
        session = self.create_session(config_path, experiment_name)
        session_id = session['session_id']

        # 3. Start training
        print("Starting training...")
        self.start_training(session_id)

        # 4. Monitor progress
        print("Monitoring training...")
        self.monitor_training(session_id)

        # 5. Get final results
        print("Fetching results...")
        metrics = self.get_metrics(session_id)
        checkpoints = self.get_checkpoints(session_id)

        return {
            'session_id': session_id,
            'metrics': metrics,
            'checkpoints': checkpoints
        }

    def create_session(self, config_path, name):
        response = self.session.post(
            f'{self.base_url}/training/sessions',
            json={'config_path': config_path, 'session_name': name}
        )
        return response.json()

    def start_training(self, session_id):
        response = self.session.post(
            f'{self.base_url}/training/sessions/{session_id}/start'
        )
        return response.json()

    def monitor_training(self, session_id):
        while True:
            status = self.get_status(session_id)
            metrics = self.get_latest_metrics(session_id)

            if metrics:
                print(f"Epoch {metrics['epoch']}: "
                      f"Loss {metrics['val_loss']:.4f}")

            if status['status'] in ['completed', 'failed', 'stopped']:
                print(f"Training {status['status']}")
                break

            time.sleep(10)

    def get_status(self, session_id):
        response = self.session.get(
            f'{self.base_url}/training/sessions/{session_id}'
        )
        return response.json()

    def get_latest_metrics(self, session_id):
        response = self.session.get(
            f'{self.base_url}/training/sessions/{session_id}/metrics',
            params={'limit': 1}
        )
        metrics = response.json()['metrics']
        return metrics[0] if metrics else None

    def get_metrics(self, session_id):
        response = self.session.get(
            f'{self.base_url}/training/sessions/{session_id}/metrics'
        )
        return response.json()

    def get_checkpoints(self, session_id):
        response = self.session.get(
            f'{self.base_url}/checkpoints',
            params={'session_id': session_id}
        )
        return response.json()

    def get_resources(self):
        response = self.session.get(f'{self.base_url}/monitor/resources')
        return response.json()

# Usage
if __name__ == '__main__':
    client = PIVOTTrainingClient()

    results = client.create_and_run_experiment(
        config_path='configs/train.yaml',
        experiment_name='unet_focal_exp001'
    )

    print(f"Experiment completed!")
    print(f"Final metrics: {results['metrics']['metrics'][-1]}")
    print(f"Checkpoints saved: {len(results['checkpoints']['checkpoints'])}")
```

## Next Steps

- Review [API Reference](API_REFERENCE.md) for complete endpoint documentation
- Check [Deployment Guide](DEPLOYMENT.md) for production setup
- Explore interactive API docs at `http://localhost:8000/docs`
