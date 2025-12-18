/**
 * Metrics Visualization Panel Component
 */

import { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { sessionService } from '../api/services';
import { useWebSocket } from '../hooks/useWebSocket';
import './MetricsPanel.css';

function MetricsPanel() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [epochHistory, setEpochHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [liveUpdates, setLiveUpdates] = useState(true);

  // WebSocket for real-time updates
  const handleWebSocketMessage = useCallback((data) => {
    console.log('Received WebSocket update:', data);
    if (data.type === 'metric_update' && selectedSession) {
      loadMetrics(selectedSession);
      loadEpochHistory(selectedSession);
    }
  }, [selectedSession]);

  const { isConnected } = useWebSocket(
    liveUpdates ? selectedSession : null,
    handleWebSocketMessage
  );

  useEffect(() => {
    loadSessions();
  }, []);

  useEffect(() => {
    if (selectedSession) {
      loadMetrics(selectedSession);
      loadEpochHistory(selectedSession);
    }
  }, [selectedSession]);

  const loadSessions = async () => {
    try {
      const data = await sessionService.getSessions();
      setSessions(data.sessions || []);
      if (data.sessions && data.sessions.length > 0 && !selectedSession) {
        setSelectedSession(data.sessions[0].id);
      }
    } catch (err) {
      setError('Failed to load sessions: ' + err.message);
    }
  };

  const loadMetrics = async (sessionId) => {
    try {
      setLoading(true);
      const data = await sessionService.getMetrics(sessionId);
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError('Failed to load metrics: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadEpochHistory = async (sessionId) => {
    try {
      const data = await sessionService.getEpochHistory(sessionId);
      setEpochHistory(data.epochs || []);
    } catch (err) {
      console.error('Failed to load epoch history:', err);
    }
  };

  const formatChartData = () => {
    if (!epochHistory || epochHistory.length === 0) return [];

    return epochHistory.map((epoch) => ({
      epoch: epoch.epoch,
      trainLoss: epoch.train_loss,
      valLoss: epoch.val_loss,
      trainAcc: epoch.train_accuracy ? epoch.train_accuracy * 100 : null,
      valAcc: epoch.val_accuracy ? epoch.val_accuracy * 100 : null,
    }));
  };

  const chartData = formatChartData();

  return (
    <div className="metrics-panel">
      <div className="panel-controls">
        <div className="session-selector">
          <label htmlFor="session-select">Session:</label>
          <select
            id="session-select"
            value={selectedSession || ''}
            onChange={(e) => setSelectedSession(e.target.value)}
          >
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.experiment_name || session.id}
              </option>
            ))}
          </select>
        </div>

        <div className="live-toggle">
          <label>
            <input
              type="checkbox"
              checked={liveUpdates}
              onChange={(e) => setLiveUpdates(e.target.checked)}
            />
            Live Updates
          </label>
          {liveUpdates && (
            <span className={`ws-status ${isConnected ? 'connected' : 'disconnected'}`}>
              {isConnected ? '● Connected' : '○ Disconnected'}
            </span>
          )}
        </div>

        <button
          onClick={() => selectedSession && loadMetrics(selectedSession)}
          className="btn btn-secondary btn-sm"
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {!selectedSession ? (
        <p className="empty-state">No sessions available. Create a session to view metrics.</p>
      ) : (
        <>
          {/* Current Metrics Summary */}
          {metrics && (
            <div className="metrics-summary">
              <div className="metric-card">
                <h3>Current Epoch</h3>
                <div className="metric-value">{metrics.current_epoch || 0}</div>
                <div className="metric-label">of {metrics.total_epochs || '?'}</div>
              </div>

              <div className="metric-card">
                <h3>Training Loss</h3>
                <div className="metric-value">
                  {metrics.latest_loss?.toFixed(4) || 'N/A'}
                </div>
                <div className="metric-label">Latest value</div>
              </div>

              <div className="metric-card">
                <h3>Validation Accuracy</h3>
                <div className="metric-value">
                  {metrics.latest_accuracy ? (metrics.latest_accuracy * 100).toFixed(2) + '%' : 'N/A'}
                </div>
                <div className="metric-label">Latest value</div>
              </div>

              <div className="metric-card">
                <h3>Training Time</h3>
                <div className="metric-value">
                  {metrics.elapsed_time ? Math.floor(metrics.elapsed_time / 60) : 0}m
                </div>
                <div className="metric-label">Elapsed</div>
              </div>
            </div>
          )}

          {/* Loss Chart */}
          {chartData.length > 0 && (
            <div className="chart-container">
              <h3>Loss Over Epochs</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
                  <XAxis
                    dataKey="epoch"
                    stroke="#b0b0b0"
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    stroke="#b0b0b0"
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#2a2a2a', border: '1px solid #3a3a3a' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    stroke="#61dafb"
                    name="Training Loss"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    stroke="#ff9800"
                    name="Validation Loss"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Accuracy Chart */}
          {chartData.length > 0 && chartData.some(d => d.trainAcc !== null) && (
            <div className="chart-container">
              <h3>Accuracy Over Epochs</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
                  <XAxis
                    dataKey="epoch"
                    stroke="#b0b0b0"
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    stroke="#b0b0b0"
                    label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#2a2a2a', border: '1px solid #3a3a3a' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainAcc"
                    stroke="#4caf50"
                    name="Training Accuracy"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    stroke="#f44336"
                    name="Validation Accuracy"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {chartData.length === 0 && (
            <p className="empty-state">No training data available yet. Start training to see metrics.</p>
          )}
        </>
      )}
    </div>
  );
}

export default MetricsPanel;
