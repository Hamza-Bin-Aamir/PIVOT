import { useState, useEffect } from 'react';
import { healthService } from './api/services';
import ConfigPanel from './components/ConfigPanel';
import MetricsPanel from './components/MetricsPanel';
import LogsPanel from './components/LogsPanel';
import './App.css';

function App() {
  const [apiStatus, setApiStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('config');

  useEffect(() => {
    // Check API health on mount
    healthService.getHealth()
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('disconnected'));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>PIVOT Training Dashboard</h1>
        <div className="api-status">
          API: <span className={`status-${apiStatus}`}>{apiStatus}</span>
        </div>
      </header>

      <nav className="app-nav">
        <button
          className={activeTab === 'config' ? 'active' : ''}
          onClick={() => setActiveTab('config')}
        >
          Configuration
        </button>
        <button
          className={activeTab === 'metrics' ? 'active' : ''}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics
        </button>
        <button
          className={activeTab === 'logs' ? 'active' : ''}
          onClick={() => setActiveTab('logs')}
        >
          Logs
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'config' && <ConfigPanel />}
        {activeTab === 'metrics' && <MetricsPanel />}
        {activeTab === 'logs' && <LogsPanel />}
      </main>
    </div>
  );
}

export default App;
