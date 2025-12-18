/**
 * Training Logs Viewer Component
 */

import { useState, useEffect, useCallback } from 'react';
import { notificationService } from '../api/services';
import './LogsPanel.css';

function LogsPanel() {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    type: '',
    priority: '',
    unread_only: false,
  });
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadNotifications();
  }, [filters]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadNotifications();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, filters]);

  const loadNotifications = async () => {
    try {
      setLoading(true);
      const data = await notificationService.getNotifications({
        ...filters,
        limit: 100,
      });
      setNotifications(data.notifications || []);
      setError(null);
    } catch (err) {
      setError('Failed to load notifications: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkAsRead = async (notificationId) => {
    try {
      await notificationService.markAsRead(notificationId);
      await loadNotifications();
    } catch (err) {
      setError('Failed to mark as read: ' + err.message);
    }
  };

  const handleMarkAllAsRead = async () => {
    try {
      await notificationService.markAllAsRead();
      await loadNotifications();
    } catch (err) {
      setError('Failed to mark all as read: ' + err.message);
    }
  };

  const handleClearAll = async () => {
    if (!confirm('Clear all notifications?')) return;

    try {
      await notificationService.clearNotifications();
      await loadNotifications();
    } catch (err) {
      setError('Failed to clear notifications: ' + err.message);
    }
  };

  const getNotificationIcon = (type) => {
    const icons = {
      info: 'ℹ️',
      warning: '⚠️',
      error: '❌',
      success: '✅',
    };
    return icons[type] || '📝';
  };

  const getPriorityClass = (priority) => {
    const classes = {
      low: 'priority-low',
      medium: 'priority-medium',
      high: 'priority-high',
      critical: 'priority-critical',
    };
    return classes[priority] || '';
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="logs-panel">
      <div className="logs-header">
        <h2>Training Logs & Notifications</h2>
        <div className="header-actions">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button
            onClick={loadNotifications}
            className="btn btn-secondary btn-sm"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
          <button
            onClick={handleMarkAllAsRead}
            className="btn btn-secondary btn-sm"
          >
            Mark All Read
          </button>
          <button
            onClick={handleClearAll}
            className="btn btn-danger btn-sm"
          >
            Clear All
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="logs-filters">
        <div className="filter-group">
          <label>Type:</label>
          <select
            value={filters.type}
            onChange={(e) => setFilters({ ...filters, type: e.target.value })}
          >
            <option value="">All</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
            <option value="success">Success</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Priority:</label>
          <select
            value={filters.priority}
            onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
          >
            <option value="">All</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>

        <div className="filter-group">
          <label>
            <input
              type="checkbox"
              checked={filters.unread_only}
              onChange={(e) => setFilters({ ...filters, unread_only: e.target.checked })}
            />
            Unread only
          </label>
        </div>
      </div>

      <div className="logs-container">
        {notifications.length === 0 ? (
          <p className="empty-state">No notifications to display.</p>
        ) : (
          <div className="notifications-list">
            {notifications.map((notification) => (
              <div
                key={notification.id}
                className={`notification-item ${notification.read ? 'read' : 'unread'} ${getPriorityClass(notification.priority)}`}
                onClick={() => !notification.read && handleMarkAsRead(notification.id)}
              >
                <div className="notification-icon">
                  {getNotificationIcon(notification.type)}
                </div>
                <div className="notification-content">
                  <div className="notification-header">
                    <h4>{notification.title}</h4>
                    <div className="notification-meta">
                      <span className={`type-badge type-${notification.type}`}>
                        {notification.type}
                      </span>
                      <span className={`priority-badge priority-${notification.priority}`}>
                        {notification.priority}
                      </span>
                      <span className="timestamp">
                        {formatTimestamp(notification.timestamp)}
                      </span>
                    </div>
                  </div>
                  <p className="notification-message">{notification.message}</p>
                  {notification.session_id && (
                    <div className="notification-session">
                      Session: <code>{notification.session_id}</code>
                    </div>
                  )}
                  {notification.metadata && Object.keys(notification.metadata).length > 0 && (
                    <details className="notification-metadata">
                      <summary>Metadata</summary>
                      <pre>{JSON.stringify(notification.metadata, null, 2)}</pre>
                    </details>
                  )}
                </div>
                {!notification.read && (
                  <div className="unread-indicator" title="Unread">●</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default LogsPanel;
