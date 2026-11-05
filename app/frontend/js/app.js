/**
 * HydroPhysics AI — Main Application Orchestrator & State Manager
 * Plain Vanilla JavaScript, Zero Framework Dependencies
 */

const App = {
  state: {
    theme: localStorage.getItem('hydro_theme') || 'dark',
    activeTab: 'overview',
    catchments: [],
    selectedCatchmentId: '54057',
    currentCatchmentMeta: null,
    activeJobId: null,
    jobPollInterval: null,
  },

  async init() {
    console.log('Initializing HydroPhysics AI Application...');
    this.initTheme();
    this.bindGlobalEvents();

    try {
      await this.loadCatchments();
      await this.loadCurrentCatchmentData();
      await this.switchTab('overview');
    } catch (err) {
      console.error('Initialization failed:', err);
      this.showToast('Failed to initialize application from backend.', 'error');
    }
  },

  // ---------------------------------------------------------------------------
  // Theme Management (Dark / Light Mode)
  // ---------------------------------------------------------------------------
  initTheme() {
    document.documentElement.setAttribute('data-theme', this.state.theme);
    this.updateThemeButton();
  },

  toggleTheme() {
    this.state.theme = this.state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', this.state.theme);
    localStorage.setItem('hydro_theme', this.state.theme);
    this.updateThemeButton();
    // Re-render active charts to update color schemes
    this.renderCurrentTabCharts();
  },

  updateThemeButton() {
    const btn = document.getElementById('theme-toggle-btn');
    if (!btn) return;
    const isDark = this.state.theme === 'dark';
    btn.innerHTML = isDark
      ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg> Light Mode`
      : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg> Dark Mode`;
  },

  // ---------------------------------------------------------------------------
  // Global Event Binding & Navigation
  // ---------------------------------------------------------------------------
  bindGlobalEvents() {
    // Theme toggle
    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) themeBtn.addEventListener('click', () => this.toggleTheme());

    // Navigation items
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', e => {
        e.preventDefault();
        const tab = item.getAttribute('data-tab');
        if (tab) this.switchTab(tab);
      });
    });

    // Mobile nav toggle
    const mobileBtn = document.getElementById('mobile-nav-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (mobileBtn && sidebar) {
      mobileBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
    }

    // Catchment selector change
    const catchmentSelect = document.getElementById('global-catchment-select');
    if (catchmentSelect) {
      catchmentSelect.addEventListener('change', async e => {
        this.state.selectedCatchmentId = e.target.value;
        await this.loadCurrentCatchmentData();
        await this.renderCurrentTabCharts();
      });
    }

    // Forecasting Form submit
    const forecastBtn = document.getElementById('btn-run-forecast');
    if (forecastBtn) {
      forecastBtn.addEventListener('click', () => this.runForecast());
    }

    // Experiment Form submit
    const launchExpBtn = document.getElementById('btn-launch-experiment');
    if (launchExpBtn) {
      launchExpBtn.addEventListener('click', () => this.launchExperiment());
    }

    // Data Explorer filter apply
    const applyFilterBtn = document.getElementById('btn-apply-data-filter');
    if (applyFilterBtn) {
      applyFilterBtn.addEventListener('click', () => this.loadDataExplorerTimeseries());
    }
  },

  async switchTab(tabId) {
    this.state.activeTab = tabId;

    // Update nav classes
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-tab') === tabId);
    });

    // Update page container views
    document.querySelectorAll('.page-container').forEach(page => {
      page.classList.toggle('active', page.id === `page-${tabId}`);
    });

    // Update Header Title
    const titles = {
      'overview': 'Research Overview & Catchment Status',
      'data-explorer': 'CAMELS-GB Hydrograph & Data Explorer',
      'forecasting-studio': 'Streamflow Forecasting Studio & Uncertainty',
      'model-comparison': 'Cross-Model Benchmark & Scientific Comparison',
      'physics-analysis': 'Catchment Water-Balance & Mass Conservation',
      'experiments': 'Hydrological Experiments & Ablation Matrix',
      'about-methodology': 'Theoretical Methodology & Scientific Provenance',
    };
    const titleEl = document.getElementById('current-page-title');
    if (titleEl) titleEl.textContent = titles[tabId] || 'HydroPhysics AI';

    // Render tab-specific charts and data
    await this.renderCurrentTabCharts();

    // Close mobile sidebar if open
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) sidebar.classList.remove('open');
  },

  // ---------------------------------------------------------------------------
  // Data Loading & State Refresh
  // ---------------------------------------------------------------------------
  async loadCatchments() {
    const list = await API.getCatchments();
    this.state.catchments = list;

    const select = document.getElementById('global-catchment-select');
    if (select) {
      select.innerHTML = '';
      list.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.gauge_id;
        opt.textContent = `${c.gauge_id} — ${c.gauge_name} (${c.area_km2} km²)`;
        select.appendChild(opt);
      });
      select.value = this.state.selectedCatchmentId;
    }
  },

  async loadCurrentCatchmentData() {
    const cid = this.state.selectedCatchmentId;
    try {
      const detail = await API.getCatchmentDetail(cid);
      this.state.currentCatchmentMeta = detail;
      this.updateCatchmentProfileUI(detail);
    } catch (err) {
      console.warn('Could not load catchment detail:', err);
    }
  },

  updateCatchmentProfileUI(detail) {
    const meta = detail.metadata || {};
    const sum = detail.data_summary || {};

    const elName = document.getElementById('profile-catchment-name');
    const elRegime = document.getElementById('profile-regime');
    const elArea = document.getElementById('profile-area');
    const elBfi = document.getElementById('profile-bfi');
    const elMeanP = document.getElementById('profile-mean-p');
    const elRunoffRatio = document.getElementById('profile-runoff-ratio');

    if (elName) elName.textContent = `${meta.gauge_id} - ${meta.gauge_name || 'Catchment'}`;
    if (elRegime) elRegime.textContent = meta.regime || 'Temperate';
    if (elArea) elArea.textContent = `${meta.area} km²`;
    if (elBfi) elBfi.textContent = meta.bfi_host ? meta.bfi_host.toFixed(2) : '—';
    if (elMeanP) elMeanP.textContent = `${sum.mean_precipitation_mm_day || '—'} mm/day`;
    if (elRunoffRatio) elRunoffRatio.textContent = sum.runoff_ratio || '—';
  },

  async renderCurrentTabCharts() {
    const tab = this.state.activeTab;
    const cid = this.state.selectedCatchmentId;

    if (tab === 'overview') {
      await this.loadOverviewTab(cid);
    } else if (tab === 'data-explorer') {
      await this.loadDataExplorerTimeseries();
      await this.loadDataExplorerStats();
    } else if (tab === 'forecasting-studio') {
      await this.runForecast();
    } else if (tab === 'model-comparison') {
      await this.loadModelComparison(cid);
    } else if (tab === 'physics-analysis') {
      await this.loadPhysicsAnalysis(cid);
    } else if (tab === 'experiments') {
      await this.loadExperimentsTab();
    }
  },

  // ---------------------------------------------------------------------------
  // Page 1: Overview Tab
  // ---------------------------------------------------------------------------
  async loadOverviewTab(cid) {
    try {
      const comp = await API.getModelComparison(cid);
      const models = comp.models || {};
      const pi = models.physics_informed_lstm?.metrics || {};
      const lstm = models.conventional_lstm?.metrics || {};
      const rf = models.random_forest?.metrics || {};

      // Update KPI Stat cards
      const elNse = document.getElementById('kpi-pi-nse');
      const elKge = document.getElementById('kpi-pi-kge');
      const elRmse = document.getElementById('kpi-pi-rmse');
      const elWb = document.getElementById('kpi-pi-wb');

      if (elNse) elNse.textContent = pi.nse !== undefined ? pi.nse.toFixed(3) : '—';
      if (elKge) elKge.textContent = pi.kge !== undefined ? pi.kge.toFixed(3) : '—';
      if (elRmse) elRmse.textContent = pi.rmse !== undefined ? `${pi.rmse.toFixed(3)} mm/d` : '—';
      if (elWb) elWb.textContent = pi.mean_abs_wb_residual_mm !== undefined ? `${pi.mean_abs_wb_residual_mm.toFixed(3)} mm/d` : '—';

      // Render Overview Hydrograph
      const pred = await API.predict(cid, 'physics_informed_lstm', 30);
      Charts.renderHydrograph('overview-hydrograph-chart', {
        ...pred,
        model_name: 'Physics-Informed LSTM',
      });
    } catch (err) {
      console.warn('Overview tab load error:', err);
    }
  },

  // ---------------------------------------------------------------------------
  // Page 2: Data Explorer Tab
  // ---------------------------------------------------------------------------
  async loadDataExplorerTimeseries() {
    const cid = this.state.selectedCatchmentId;
    const start = document.getElementById('data-filter-start')?.value || null;
    const end = document.getElementById('data-filter-end')?.value || null;
    const downsample = parseInt(document.getElementById('data-filter-downsample')?.value || '1', 10);

    try {
      const data = await API.getTimeseries(cid, start, end, downsample);
      Charts.renderHydrograph('data-explorer-hydrograph-chart', {
        dates: data.dates,
        observed_mm: data.discharge_spec,
        precipitation_mm: data.precipitation,
        model_name: 'CAMELS-GB Observed',
      });

      // Cross-correlation lag
      const corrs = await API.getCorrelations(cid);
      Charts.renderCorrelationLagChart(
        'data-explorer-correlation-chart',
        corrs.lags_days,
        corrs.precipitation_discharge_correlations
      );
    } catch (err) {
      console.error('Data explorer load error:', err);
    }
  },

  async loadDataExplorerStats() {
    const cid = this.state.selectedCatchmentId;
    try {
      const res = await API.getStats(cid);
      const stats = res.statistics || {};
      const tbody = document.getElementById('stats-table-body');
      if (!tbody) return;

      tbody.innerHTML = '';
      const varLabels = {
        precipitation: 'Precipitation (mm/day)',
        discharge_spec: 'Specific Runoff (mm/day)',
        discharge_vol: 'Volumetric Streamflow (m³/s)',
        pet: 'Potential ET (mm/day)',
        temperature: 'Temperature (°C)',
      };

      for (const [key, label] of Object.entries(varLabels)) {
        const s = stats[key];
        if (!s) continue;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${label}</strong></td>
          <td>${s.count}</td>
          <td>${s.missing_count} (${s.missing_pct}%)</td>
          <td>${s.mean.toFixed(2)}</td>
          <td>${s.std.toFixed(2)}</td>
          <td>${s.min.toFixed(2)}</td>
          <td>${s.median.toFixed(2)}</td>
          <td>${s.p95.toFixed(2)}</td>
          <td>${s.max.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
      }
    } catch (err) {
      console.warn('Stats table error:', err);
    }
  },

  // ---------------------------------------------------------------------------
  // Page 3: Forecasting Studio Tab
  // ---------------------------------------------------------------------------
  async runForecast() {
    const cid = this.state.selectedCatchmentId;
    const modelType = document.getElementById('forecast-model-select')?.value || 'physics_informed_lstm';
    const lookback = parseInt(document.getElementById('forecast-lookback-input')?.value || '30', 10);
    const btn = document.getElementById('btn-run-forecast');

    if (btn) btn.disabled = true;

    try {
      const pred = await API.predict(cid, modelType, lookback);
      const modelLabels = {
        physics_informed_lstm: 'Physics-Informed LSTM',
        conventional_lstm: 'Conventional LSTM',
        random_forest: 'Random Forest Baseline',
      };

      Charts.renderHydrograph('forecast-hydrograph-chart', {
        ...pred,
        model_name: modelLabels[modelType] || 'Model',
      });

      // Quick point accuracy metrics
      const obs = pred.observed_mm || [];
      const y_pred = pred.predicted_mm || [];
      if (obs.length > 0 && y_pred.length > 0) {
        let sqErr = 0, absErr = 0, sumObs = 0;
        for (let i = 0; i < obs.length; i++) {
          const d = obs[i] - y_pred[i];
          sqErr += d * d;
          absErr += Math.abs(d);
          sumObs += obs[i];
        }
        const rmse = Math.sqrt(sqErr / obs.length);
        const mae = absErr / obs.length;
        const meanObs = sumObs / obs.length;
        let denomNse = 0;
        for (let i = 0; i < obs.length; i++) {
          denomNse += Math.pow(obs[i] - meanObs, 2);
        }
        const nse = denomNse > 0 ? 1.0 - (sqErr / denomNse) : 0;

        const elNse = document.getElementById('forecast-metric-nse');
        const elRmse = document.getElementById('forecast-metric-rmse');
        const elMae = document.getElementById('forecast-metric-mae');
        if (elNse) elNse.textContent = nse.toFixed(3);
        if (elRmse) elRmse.textContent = `${rmse.toFixed(3)} mm/d`;
        if (elMae) elMae.textContent = `${mae.toFixed(3)} mm/d`;
      }
    } catch (err) {
      console.error('Forecast execution failed:', err);
      this.showToast('Failed to generate streamflow forecast.', 'error');
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  // ---------------------------------------------------------------------------
  // Page 4: Model Comparison Tab
  // ---------------------------------------------------------------------------
  async loadModelComparison(cid) {
    try {
      const res = await API.getModelComparison(cid);
      const models = res.models || {};

      Charts.renderModelComparisonBarChart('model-comparison-barchart', models);

      // Populate comparison table
      const tbody = document.getElementById('comparison-table-body');
      if (tbody) {
        tbody.innerHTML = '';
        for (const [key, m] of Object.entries(models)) {
          const met = m.metrics || {};
          const tr = document.createElement('tr');
          const isPi = key === 'physics_informed_lstm';
          tr.innerHTML = `
            <td><strong>${m.name}</strong> ${isPi ? '<span class="badge badge-cyan">Physics-Guided</span>' : ''}</td>
            <td><strong>${met.nse !== undefined ? met.nse.toFixed(3) : '—'}</strong></td>
            <td>${met.kge !== undefined ? met.kge.toFixed(3) : '—'}</td>
            <td>${met.rmse !== undefined ? met.rmse.toFixed(3) : '—'}</td>
            <td>${met.mae !== undefined ? met.mae.toFixed(3) : '—'}</td>
            <td>${met.pbias !== undefined ? `${met.pbias.toFixed(1)}%` : '—'}</td>
            <td>${met.mean_abs_wb_residual_mm !== undefined ? `${met.mean_abs_wb_residual_mm.toFixed(3)}` : '<span class="text-muted">N/A</span>'}</td>
            <td>${met.picp_pct !== undefined ? `${met.picp_pct.toFixed(1)}%` : '—'}</td>
            <td>${met.train_time_sec !== undefined ? `${met.train_time_sec.toFixed(1)}s` : '—'}</td>
          `;
          tbody.appendChild(tr);
        }
      }
    } catch (err) {
      console.error('Comparison load error:', err);
    }
  },

  // ---------------------------------------------------------------------------
  // Page 5: Physics Analysis Tab
  // ---------------------------------------------------------------------------
  async loadPhysicsAnalysis(cid) {
    try {
      const wb = await API.getWaterBalance(cid);
      Charts.renderWaterBalance('physics-water-balance-chart', wb);

      const elClosure = document.getElementById('wb-closure-value');
      const elRes = document.getElementById('wb-mean-res-value');
      if (elClosure) elClosure.textContent = `${wb.total_mass_balance_closure_mm} mm`;
      if (elRes) elRes.textContent = `${wb.mean_abs_residual_mm} mm/day`;
    } catch (err) {
      console.error('Physics analysis error:', err);
    }
  },

  // ---------------------------------------------------------------------------
  // Page 6: Experiments Tab & Live Background Tracking
  // ---------------------------------------------------------------------------
  async loadExperimentsTab() {
    await this.refreshExperimentsList();
    Charts.renderScarcityAblation('scarcity-ablation-chart');
  },

  async launchExperiment() {
    const config = {
      gauge_id: document.getElementById('exp-catchment-select')?.value || this.state.selectedCatchmentId,
      data_proportion: parseFloat(document.getElementById('exp-scarcity-select')?.value || '1.0'),
      noise_level: parseFloat(document.getElementById('exp-noise-select')?.value || '0.0'),
      lambda_physics: parseFloat(document.getElementById('exp-lambda-input')?.value || '0.1'),
      lookback_window: parseInt(document.getElementById('exp-lookback-input')?.value || '30', 10),
      epochs: parseInt(document.getElementById('exp-epochs-input')?.value || '15', 10),
      batch_size: parseInt(document.getElementById('exp-batch-size-input')?.value || '32', 10),
      hidden_dim: parseInt(document.getElementById('exp-hidden-dim-input')?.value || '64', 10),
    };

    try {
      const res = await API.launchExperiment(config);
      this.state.activeJobId = res.job_id;
      this.showToast(`Launched experiment ${res.job_id} in background.`, 'success');
      this.startJobPolling(res.job_id);
    } catch (err) {
      this.showToast(`Failed to launch experiment: ${err.message}`, 'error');
    }
  },

  startJobPolling(jobId) {
    const trackerCard = document.getElementById('active-job-tracker');
    if (trackerCard) trackerCard.style.display = 'block';

    if (this.state.jobPollInterval) clearInterval(this.state.jobPollInterval);

    this.state.jobPollInterval = setInterval(async () => {
      try {
        const job = await API.getExperimentStatus(jobId);
        this.updateJobTrackerUI(job);

        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(this.state.jobPollInterval);
          this.state.jobPollInterval = null;
          await this.refreshExperimentsList();

          if (job.status === 'completed') {
            const results = await API.getExperimentResults(jobId);
            const hist = results.results?.models?.physics_informed_lstm?.training_history;
            if (hist) {
              Charts.renderTrainingLosses('experiment-loss-chart', hist);
            }
          }
        }
      } catch (err) {
        console.warn('Job polling error:', err);
      }
    }, 1500);
  },

  updateJobTrackerUI(job) {
    const elId = document.getElementById('job-tracker-id');
    const elStatus = document.getElementById('job-tracker-status');
    const elProg = document.getElementById('job-tracker-progress-bar');
    const elMsg = document.getElementById('job-tracker-message');
    const elLoss = document.getElementById('job-tracker-loss');

    if (elId) elId.textContent = job.job_id;
    if (elStatus) elStatus.textContent = job.status.toUpperCase();
    if (elProg) elProg.style.width = `${job.progress_pct}%`;
    if (elMsg) elMsg.textContent = job.status_message;
    if (elLoss) elLoss.textContent = job.current_train_loss ? `Loss: ${job.current_train_loss}` : '';
  },

  async refreshExperimentsList() {
    try {
      const jobs = await API.getExperiments();
      const tbody = document.getElementById('experiments-table-body');
      if (!tbody) return;

      tbody.innerHTML = '';
      jobs.forEach(j => {
        const tr = document.createElement('tr');
        const cfg = j.config || {};
        const isDone = j.status === 'completed';
        const exportUrl = API.getExportCsvUrl(j.job_id);

        tr.innerHTML = `
          <td><code>${j.job_id}</code></td>
          <td>${cfg.gauge_id || '—'}</td>
          <td>${cfg.data_proportion ? `${cfg.data_proportion * 100}%` : '100%'}</td>
          <td>${cfg.noise_level ? `${cfg.noise_level * 100}%` : '0%'}</td>
          <td>λ = ${cfg.lambda_physics ?? 0.1}</td>
          <td>
            <span class="badge ${isDone ? 'badge-emerald' : (j.status === 'running' ? 'badge-cyan' : 'badge-amber')}">
              ${j.status}
            </span>
          </td>
          <td>
            ${isDone ? `<a href="${exportUrl}" class="btn btn-secondary btn-sm" download>CSV Export</a>` : '—'}
          </td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.warn('Failed to refresh experiment list:', err);
    }
  },

  showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `status-badge ${type === 'error' ? 'badge-rose' : 'badge-cyan'}`;
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.zIndex = '9999';
    toast.style.boxShadow = 'var(--shadow-md)';
    toast.style.padding = '12px 18px';
    toast.style.fontSize = '0.85rem';
    toast.textContent = message;

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  },
};

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => App.init());

