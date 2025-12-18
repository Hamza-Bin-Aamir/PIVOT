import { useState, useEffect } from 'react';
import { healthService } from './api/services';
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
        {activeTab === 'config' && (
          <div className="panel">
            <h2>Training Configuration</h2>
            <p>Configuration panel will be added in Issue #102</p>
          </div>
        )}
        {activeTab === 'metrics' && (
          <div className="panel">
            <h2>Metrics Visualization</h2>
            <p>Metrics panel will be added in Issue #103</p>
          </div>
        )}
        {activeTab === 'logs' && (
          <div className="panel">
            <h2>Training Logs</h2>
            <p>Logs panel will be added in Issue #104</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
