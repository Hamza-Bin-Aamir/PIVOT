/**
 * API service for training sessions
 */

import apiClient from './client';

export const sessionService = {
  // Get all sessions
  async getSessions() {
    const response = await apiClient.get('/status/sessions');
    return response.data;
  },

  // Get session status
  async getSessionStatus(sessionId) {
    const response = await apiClient.get(`/status/sessions/${sessionId}`);
    return response.data;
  },

  // Create new session
  async createSession(configPath, experimentName) {
    const response = await apiClient.post('/status/sessions', {
      config_path: configPath,
      experiment_name: experimentName,
    });
    return response.data;
  },

  // Delete session
  async deleteSession(sessionId) {
    await apiClient.delete(`/status/sessions/${sessionId}`);
  },

  // Get session metrics
  async getMetrics(sessionId) {
    const response = await apiClient.get(`/metrics/sessions/${sessionId}`);
    return response.data;
  },

  // Get epoch history
  async getEpochHistory(sessionId) {
    const response = await apiClient.get(`/epochs/sessions/${sessionId}`);
    return response.data;
  },
};

export const notificationService = {
  // Get all notifications
  async getNotifications(filters = {}) {
    const params = new URLSearchParams();
    if (filters.session_id) params.append('session_id', filters.session_id);
    if (filters.type) params.append('type', filters.type);
    if (filters.priority) params.append('priority', filters.priority);
    if (filters.unread_only) params.append('unread_only', 'true');
    if (filters.limit) params.append('limit', filters.limit);

    const response = await apiClient.get(`/notifications/?${params}`);
    return response.data;
  },

  // Mark notification as read
  async markAsRead(notificationId) {
    const response = await apiClient.patch(`/notifications/${notificationId}`, {
      read: true,
    });
    return response.data;
  },

  // Mark all as read
  async markAllAsRead(sessionId = null) {
    const params = sessionId ? `?session_id=${sessionId}` : '';
    const response = await apiClient.post(`/notifications/mark-all-read${params}`);
    return response.data;
  },

  // Clear notifications
  async clearNotifications(sessionId = null) {
    const params = sessionId ? `?session_id=${sessionId}` : '';
    await apiClient.delete(`/notifications/${params}`);
  },
};

export const healthService = {
  // Check API health
  async getHealth() {
    const response = await apiClient.get('/health');
    return response.data;
  },
};
