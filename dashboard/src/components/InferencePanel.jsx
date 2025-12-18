import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import './InferencePanel.css';

function InferencePanel() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  // Mock patient data (in production, fetch from API)
  useEffect(() => {
    const mockPatients = [
      { id: 'LIDC-IDRI-0001', name: 'Patient 001', scanDate: '2024-01-15', nodules: 3 },
      { id: 'LIDC-IDRI-0002', name: 'Patient 002', scanDate: '2024-01-16', nodules: 1 },
      { id: 'LIDC-IDRI-0003', name: 'Patient 003', scanDate: '2024-01-17', nodules: 2 },
      { id: 'LIDC-IDRI-0004', name: 'Patient 004', scanDate: '2024-01-18', nodules: 0 },
      { id: 'LIDC-IDRI-0005', name: 'Patient 005', scanDate: '2024-01-19', nodules: 5 },
    ];
    setPatients(mockPatients);
  }, []);

  const runInference = async () => {
    if (!selectedPatient) {
      setError('Please select a patient first');
      return;
    }

    setIsRunning(true);
    setError(null);
    setProgress(0);

    try {
      // Simulate inference progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      // Mock inference results (in production, call actual API)
      await new Promise(resolve => setTimeout(resolve, 3000));
      clearInterval(progressInterval);
      setProgress(100);

      const mockResults = {
        patientId: selectedPatient.id,
        processingTime: 2.84,
        nodulesDetected: Math.floor(Math.random() * 6),
        predictions: [
          {
            id: 1,
            location: { x: 128, y: 256, z: 64 },
            size: { x: 12.3, y: 11.8, z: 10.5 },
            confidence: 0.94,
            malignancy: 0.73,
            severity: 'medium',
          },
          {
            id: 2,
            location: { x: 200, y: 180, z: 80 },
            size: { x: 8.2, y: 7.9, z: 8.4 },
            confidence: 0.87,
            malignancy: 0.45,
            severity: 'low',
          },
          {
            id: 3,
            location: { x: 156, y: 320, z: 96 },
            size: { x: 15.6, y: 14.2, z: 13.8 },
            confidence: 0.91,
            malignancy: 0.82,
            severity: 'high',
          },
        ].slice(0, Math.floor(Math.random() * 3) + 1),
      };

      setInferenceResults(mockResults);
    } catch (err) {
      setError(err.message || 'Inference failed');
    } finally {
      setIsRunning(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return '#e74c3c';
      case 'medium': return '#f39c12';
      case 'low': return '#27ae60';
      default: return '#95a5a6';
    }
  };

  return (
    <div className="inference-panel">
      <h2>Inference & Visualization</h2>

      <div className="inference-container">
        {/* Patient Selection */}
        <div className="patient-selection">
          <h3>Select Patient</h3>
          <div className="patient-list">
            {patients.map(patient => (
              <div
                key={patient.id}
                className={`patient-card ${selectedPatient?.id === patient.id ? 'selected' : ''}`}
                onClick={() => setSelectedPatient(patient)}
              >
                <div className="patient-id">{patient.id}</div>
                <div className="patient-info">
                  <span>{patient.name}</span>
                  <span className="scan-date">{patient.scanDate}</span>
                </div>
                <div className="nodule-count">
                  <span className="badge">{patient.nodules} nodules</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Inference Controls */}
        <div className="inference-controls">
          <h3>Inference Controls</h3>
          {selectedPatient && (
            <div className="selected-patient-info">
              <p><strong>Selected:</strong> {selectedPatient.name} ({selectedPatient.id})</p>
              <p><strong>Scan Date:</strong> {selectedPatient.scanDate}</p>
            </div>
          )}

          <button
            onClick={runInference}
            disabled={!selectedPatient || isRunning}
            className="btn-primary"
          >
            {isRunning ? 'Running Inference...' : 'Run Inference'}
          </button>

          {isRunning && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }}>
                {progress}%
              </div>
            </div>
          )}

          {error && <div className="error-message">{error}</div>}
        </div>

        {/* Results Visualization */}
        {inferenceResults && (
          <div className="inference-results">
            <h3>Detection Results</h3>

            <div className="results-summary">
              <div className="summary-card">
                <div className="summary-label">Processing Time</div>
                <div className="summary-value">{inferenceResults.processingTime}s</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Nodules Detected</div>
                <div className="summary-value">{inferenceResults.nodulesDetected}</div>
              </div>
              <div className="summary-card">
                <div className="summary-label">Avg Confidence</div>
                <div className="summary-value">
                  {inferenceResults.predictions.length > 0
                    ? (inferenceResults.predictions.reduce((sum, p) => sum + p.confidence, 0) /
                        inferenceResults.predictions.length).toFixed(2)
                    : 'N/A'}
                </div>
              </div>
            </div>

            {/* 3D Visualization Placeholder */}
            <div className="visualization-container">
              <h4>3D Visualization</h4>
              <div className="viewer-3d">
                <div className="viewer-placeholder">
                  <svg width="100%" height="300" viewBox="0 0 400 300">
                    <rect width="400" height="300" fill="#1e1e1e" />

                    {/* Coordinate axes */}
                    <line x1="200" y1="150" x2="350" y2="150" stroke="#4a9eff" strokeWidth="2" />
                    <line x1="200" y1="150" x2="200" y2="50" stroke="#4aff9e" strokeWidth="2" />
                    <line x1="200" y1="150" x2="150" y2="200" stroke="#ff4a9e" strokeWidth="2" />

                    {/* Draw nodules */}
                    {inferenceResults.predictions.map((pred, idx) => {
                      const x = 200 + (pred.location.x / 2);
                      const y = 150 - (pred.location.y / 2);
                      const r = Math.sqrt(pred.size.x * pred.size.y) / 2;
                      const color = getSeverityColor(pred.severity);

                      return (
                        <g key={idx}>
                          <circle
                            cx={x}
                            cy={y}
                            r={r}
                            fill={color}
                            opacity="0.7"
                            stroke={color}
                            strokeWidth="2"
                          />
                          <text
                            x={x}
                            y={y - r - 5}
                            fill="white"
                            fontSize="12"
                            textAnchor="middle"
                          >
                            #{pred.id}
                          </text>
                        </g>
                      );
                    })}

                    <text x="10" y="20" fill="#ccc" fontSize="14">
                      3D Scan Visualization (Simplified)
                    </text>
                  </svg>
                </div>
              </div>
            </div>

            {/* Detection Details Table */}
            <div className="detection-table">
              <h4>Detection Details</h4>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Location (x, y, z)</th>
                    <th>Size (mm)</th>
                    <th>Confidence</th>
                    <th>Malignancy</th>
                    <th>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {inferenceResults.predictions.map(pred => (
                    <tr key={pred.id}>
                      <td>#{pred.id}</td>
                      <td>
                        ({pred.location.x}, {pred.location.y}, {pred.location.z})
                      </td>
                      <td>
                        {pred.size.x.toFixed(1)} × {pred.size.y.toFixed(1)} × {pred.size.z.toFixed(1)}
                      </td>
                      <td>
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{ width: `${pred.confidence * 100}%` }}
                          />
                          <span>{(pred.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td>
                        <span className={`malignancy-score ${pred.malignancy > 0.7 ? 'high' : pred.malignancy > 0.4 ? 'medium' : 'low'}`}>
                          {(pred.malignancy * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td>
                        <span
                          className="severity-badge"
                          style={{ backgroundColor: getSeverityColor(pred.severity) }}
                        >
                          {pred.severity}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Heatmap Visualization */}
            <div className="heatmap-container">
              <h4>Attention Heatmap</h4>
              <div className="heatmap-grid">
                {Array.from({ length: 64 }).map((_, idx) => {
                  const intensity = Math.random();
                  const color = `rgba(255, ${Math.floor(100 + intensity * 155)}, ${Math.floor(100 - intensity * 100)}, ${0.5 + intensity * 0.5})`;
                  return <div key={idx} className="heatmap-cell" style={{ backgroundColor: color }} />;
                })}
              </div>
              <div className="heatmap-legend">
                <span>Low</span>
                <div className="legend-gradient" />
                <span>High</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default InferencePanel;
