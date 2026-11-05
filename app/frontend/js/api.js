/**
 * HydroPhysics AI — REST API Client
 * Scientific Machine Learning & Hydrology Web Interface
 */

const API = {
  baseUrl: window.location.origin,

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP Error ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      console.error(`API Error on ${endpoint}:`, err);
      throw err;
    }
  },

  // Health and System Diagnostics
  async getHealth() {
    return this.request('/api/health');
  },

  // Catchments
  async getCatchments() {
    return this.request('/api/catchments');
  },

  async getCatchmentDetail(gaugeId) {
    return this.request(`/api/catchments/${gaugeId}`);
  },

  // Data Explorer
  async getTimeseries(gaugeId, startDate = null, endDate = null, downsample = 1) {
    let q = `?downsample_factor=${downsample}`;
    if (startDate) q += `&start_date=${startDate}`;
    if (endDate) q += `&end_date=${endDate}`;
    return this.request(`/api/data/${gaugeId}${q}`);
  },

  async getStats(gaugeId) {
    return this.request(`/api/data/${gaugeId}/stats`);
  },

  async getCorrelations(gaugeId) {
    return this.request(`/api/data/${gaugeId}/correlations`);
  },

  // Model Predictions & Comparison
  async predict(gaugeId, modelType = 'physics_informed_lstm', lookback = 30) {
    return this.request('/api/models/predict', {
      method: 'POST',
      body: JSON.stringify({
        gauge_id: String(gaugeId),
        model_type: modelType,
        lookback_window: lookback,
      }),
    });
  },

  async getModelComparison(gaugeId) {
    return this.request(`/api/models/compare/${gaugeId}`);
  },

  // Physics Diagnostics
  async getPhysicsEquations() {
    return this.request('/api/physics/equations');
  },

  async getWaterBalance(gaugeId) {
    return this.request(`/api/physics/water-balance/${gaugeId}`);
  },

  // Experiments & Background Training
  async launchExperiment(config) {
    return this.request('/api/experiments/launch', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  async getExperiments() {
    return this.request('/api/experiments');
  },

  async getExperimentStatus(jobId) {
    return this.request(`/api/experiments/${jobId}`);
  },

  async getExperimentResults(jobId) {
    return this.request(`/api/experiments/${jobId}/results`);
  },

  getExportCsvUrl(jobId) {
    return `${this.baseUrl}/api/experiments/${jobId}/export/csv`;
  },
};

