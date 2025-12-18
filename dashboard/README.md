# PIVOT Training Dashboard

React-based dashboard for monitoring and controlling PIVOT training sessions in real-time.

## Features

- Real-time training metrics visualization
- Training session configuration and management
- Live logs and notifications
- WebSocket/SSE integration for live updates

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
```bash
cp .env.example .env
```

3. Start development server:
```bash
npm run dev
```

The dashboard will be available at http://localhost:5173

## API Integration

The dashboard connects to the PIVOT API server (default: http://localhost:8000/api/v1).

Make sure the API server is running before starting the dashboard:
```bash
# In the main PIVOT directory
uv run uvicorn src.api.app:app --reload
```

## Project Structure

```
dashboard/
├── src/
│   ├── api/          # API client and services
│   ├── hooks/        # React hooks (WebSocket, etc.)
│   ├── components/   # React components (to be added)
│   ├── App.jsx       # Main application component
│   └── main.jsx      # Application entry point
├── package.json
└── vite.config.js
```

## Development

- Build for production: `npm run build`
- Preview production build: `npm run preview`
- Lint code: `npm run lint`
