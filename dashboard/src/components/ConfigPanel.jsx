/**
 * Training Configuration Panel Component
 */

import { useState, useEffect } from 'react';
import { sessionService } from '../api/services';
import './ConfigPanel.css';

function ConfigPanel() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [configPath, setConfigPath] = useState('');
  const [experimentName, setExperimentName] = useState('');

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const data = await sessionService.getSessions();
      setSessions(data.sessions || []);
      setError(null);
    } catch (err) {
      setError('Failed to load sessions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async (e) => {
    e.preventDefault();
    if (!configPath || !experimentName) {
      setError('Please provide both config path and experiment name');
      return;
    }

    try {
      setLoading(true);
      await sessionService.createSession(configPath, experimentName);
      setConfigPath('');
      setExperimentName('');
      await loadSessions();
      setError(null);
    } catch (err) {
      setError('Failed to create session: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!confirm(`Delete session ${sessionId}?`)) return;

    try {
      setLoading(true);
      await sessionService.deleteSession(sessionId);
      await loadSessions();
      setError(null);
    } catch (err) {
      setError('Failed to delete session: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    const classMap = {
      running: 'status-running',
      completed: 'status-completed',
      failed: 'status-failed',
      idle: 'status-idle',
    };
    return <span className={`status-badge ${classMap[status] || 'status-unknown'}`}>
      {status}
    </span>;
  };

  return (
    <div className="config-panel">
      <div className="panel-section">
        <h2>Create New Training Session</h2>
        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleCreateSession} className="config-form">
          <div className="form-group">
            <label htmlFor="configPath">Configuration File Path</label>
            <input
              type="text"
              id="configPath"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              placeholder="/path/to/config.yaml"
              disabled={loading}
            />
            <span className="help-text">
              Absolute path to training configuration YAML file
            </span>
          </div>

          <div className="form-group">
            <label htmlFor="experimentName">Experiment Name</label>
            <input
              type="text"
              id="experimentName"
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              placeholder="my-training-experiment"
              disabled={loading}
            />
            <span className="help-text">
              Unique name for this training experiment
            </span>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Creating...' : 'Create Session'}
          </button>
        </form>
      </div>

      <div className="panel-section">
        <div className="section-header">
          <h2>Active Sessions</h2>
          <button
            onClick={loadSessions}
            className="btn btn-secondary"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {sessions.length === 0 ? (
          <p className="empty-state">No active sessions. Create one to get started.</p>
        ) : (
          <div className="sessions-grid">
            {sessions.map((session) => (
              <div key={session.id} className="session-card">
                <div className="session-header">
                  <h3>{session.experiment_name || session.id}</h3>
                  {getStatusBadge(session.status)}
                </div>

                <div className="session-details">
                  <div className="detail-row">
                    <span className="label">Session ID:</span>
                    <span className="value">{session.id}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Config:</span>
                    <span className="value">{session.config_path}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Created:</span>
                    <span className="value">
                      {new Date(session.created_at).toLocaleString()}
                    </span>
                  </div>
                  {session.current_epoch !== undefined && (
                    <div className="detail-row">
                      <span className="label">Epoch:</span>
                      <span className="value">
                        {session.current_epoch} / {session.total_epochs || '?'}
                      </span>
                    </div>
                  )}
                </div>

                <div className="session-actions">
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDeleteSession(session.id)}
                    disabled={loading}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ConfigPanel;
