/* ================================================================
   IHSG Predictor — Main Script
   Handles API fetching, Chart.js, scroll animations, etc.
   ================================================================ */

// ─── Utility ────────────────────────────────────────────────────

/** Format number to US locale (comma for thousands, dot for decimals) */
function formatNumber(num, decimals = 2) {
  if (num == null || isNaN(num)) return '—';
  return Number(num).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Helper: show element */
function show(el) { el.classList.remove('hidden'); }
/** Helper: hide element */
function hide(el) { el.classList.add('hidden'); }

// ─── DOM References ─────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── Scroll Animations (Intersection Observer) ──────────────────

function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  $$('.animate-on-scroll').forEach((el) => observer.observe(el));
}

// ─── Navbar: scroll effect & mobile toggle ──────────────────────

function initNavbar() {
  const navbar = $('#navbar');
  const toggle = $('#navToggle');
  const links  = $('#navLinks');

  // Scroll effect: add 'scrolled' class
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 30);
  });

  // Mobile hamburger toggle
  toggle.addEventListener('click', () => {
    links.classList.toggle('open');
  });

  // Close menu on link click (mobile)
  links.querySelectorAll('a').forEach((a) => {
    a.addEventListener('click', () => links.classList.remove('open'));
  });

  // Active link tracking
  const sections = $$('section[id]');
  const navAnchors = links.querySelectorAll('a');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach((sec) => {
      const top = sec.offsetTop - 120;
      if (window.scrollY >= top) current = sec.getAttribute('id');
    });
    navAnchors.forEach((a) => {
      a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
    });
  });
}

// ─── API Fetch Helper ───────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Section 1: Prediction ─────────────────────────────────────

async function loadPrediction(isReload = false) {
  const loader = $('#predictionLoader');
  const errorBox = $('#predictionError');
  const cards = $('#predictionCards');
  const meta = $('#metaLine');
  const modelSelect = $('#modelSelect').value;

  if (isReload) {
    show(loader);
    hide(cards);
    hide(meta);
    hide(errorBox);
  }

  try {
    const data = await fetchJSON(`/api/predict?model_type=${modelSelect}`);

    hide(loader);
    show(cards);
    show(meta);

    // Populate Last Close card
    $('#lastClose').textContent = formatNumber(data.last_close);

    // Populate Tomorrow's Prediction card
    $('#prediction').textContent = formatNumber(data.prediction);

    // Populate Difference card
    const deltaEl = $('#delta');
    const deltaPercentEl = $('#deltaPercent');
    const isUp = data.delta >= 0;
    const arrow = isUp ? '▲' : '▼';
    deltaEl.textContent = `${arrow} ${formatNumber(Math.abs(data.delta))}`;
    deltaEl.classList.add(isUp ? 'up' : 'down');
    deltaEl.classList.remove(isUp ? 'down' : 'up');
    deltaPercentEl.textContent = `${isUp ? '+' : '-'}${formatNumber(Math.abs(data.delta_percent))}%`;
    deltaPercentEl.classList.add(isUp ? 'up' : 'down');
    deltaPercentEl.classList.remove(isUp ? 'down' : 'up');

    const modelNameDisplay = data.model_used === 'svr' ? 'SVR + Optuna' : 'XGBoost + Optuna';
    // Metadata
    meta.textContent = `Last data: ${data.last_date}  •  Model: ${modelNameDisplay}`;

    if (!isReload) {
      setTimeout(() => initScrollAnimations(), 50);
    }
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#predictionErrorMsg').textContent = err.message || 'Failed to load prediction data.';
  }
}

$('#modelSelect').addEventListener('change', () => loadPrediction(true));

// ─── Section 2: Historical Chart ───────────────────────────────

let mainChartInstance = null;

async function loadChart() {
  const loader = $('#chartLoader');
  const errorBox = $('#chartError');
  const wrapper = $('#chartWrapper');
  const hint = $('#chartHint');

  try {
    const data = await fetchJSON('/api/chart-data');

    hide(loader);
    show(wrapper);
    show(hint);

    const ctx = $('#mainChart').getContext('2d');

    const actualColor = 'rgba(59, 130, 246, 1)';     // Blue
    const predictedXgbColor = 'rgba(245, 158, 11, 1)';  // Amber
    const predictedSvrColor = 'rgba(16, 185, 129, 1)';  // Green

    mainChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [
          {
            label: 'Actual',
            data: data.actual,
            borderColor: actualColor,
            backgroundColor: 'rgba(59, 130, 246, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: actualColor,
            tension: 0.3,
            fill: false,
          },
          {
            label: 'XGBoost Prediction',
            data: data.predicted_xgb,
            borderColor: predictedXgbColor,
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: predictedXgbColor,
            tension: 0.3,
            fill: false,
            borderDash: [6, 3],
          },
          {
            label: 'SVR Prediction',
            data: data.predicted_svr,
            borderColor: predictedSvrColor,
            backgroundColor: 'rgba(16, 185, 129, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: predictedSvrColor,
            tension: 0.3,
            fill: false,
            borderDash: [3, 3],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#9a9ab8',
              font: { family: 'Inter', size: 12, weight: '500' },
              padding: 20,
              usePointStyle: true,
              pointStyleWidth: 12,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15, 15, 42, 0.92)',
            titleColor: '#e8e8f0',
            bodyColor: '#9a9ab8',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { family: 'Inter', weight: '600' },
            bodyFont: { family: 'Inter' },
            callbacks: {
              label: function (ctx) {
                return `${ctx.dataset.label}: ${formatNumber(ctx.parsed.y)}`;
              },
            },
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'x',
            },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: 'x',
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#6a6a88',
              font: { family: 'Inter', size: 11 },
              maxRotation: 45,
              autoSkip: true,
              maxTicksLimit: 20,
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.04)',
            },
          },
          y: {
            ticks: {
              color: '#6a6a88',
              font: { family: 'Inter', size: 11 },
              callback: (val) => formatNumber(val, 0),
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.04)',
            },
          },
        },
      },
    });

    setTimeout(() => initScrollAnimations(), 50);
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#chartErrorMsg').textContent = err.message || 'Failed to load chart data.';
  }
}

// ─── Section 3: Model Comparison ───────────────────────────────

async function loadComparison() {
  const loader = $('#comparisonLoader');
  const errorBox = $('#comparisonError');
  const content = $('#comparisonContent');

  try {
    const data = await fetchJSON('/api/comparison');

    hide(loader);
    show(content);

    // Helper: determine winner (smaller is better)
    function getWinner(xG, xO, sG, sO, lowerBetter = true) {
      // If any is missing or undefined, give a huge penalty
      const vals = [
        {name: 'XGBoost (Optuna)', val: xO ?? (lowerBetter ? Infinity : -Infinity)},
        {name: 'SVR (Optuna)', val: sO ?? (lowerBetter ? Infinity : -Infinity)},
        {name: 'XGBoost (GridSearch)', val: xG ?? (lowerBetter ? Infinity : -Infinity)},
        {name: 'SVR (GridSearch)', val: sG ?? (lowerBetter ? Infinity : -Infinity)},
      ];
      if (lowerBetter) {
        vals.sort((a,b) => a.val - b.val);
      } else {
        vals.sort((a,b) => b.val - a.val);
      }
      return vals[0].name;
    }

    const rows = [
      {
        label: 'Validation MSE',
        xG: formatNumber(data.xgboost_gridsearch?.best_validation_mse),
        xO: formatNumber(data.xgboost_optuna?.best_validation_mse),
        sG: formatNumber(data.svr_gridsearch?.best_validation_mse),
        sO: formatNumber(data.svr_optuna?.best_validation_mse),
        win: getWinner(
            data.xgboost_gridsearch?.best_validation_mse,
            data.xgboost_optuna?.best_validation_mse,
            data.svr_gridsearch?.best_validation_mse,
            data.svr_optuna?.best_validation_mse
        ),
      },
      {
        label: 'Test MSE',
        xG: formatNumber(data.xgboost_gridsearch?.test_mse),
        xO: formatNumber(data.xgboost_optuna?.test_mse),
        sG: formatNumber(data.svr_gridsearch?.test_mse),
        sO: formatNumber(data.svr_optuna?.test_mse),
        win: getWinner(
            data.xgboost_gridsearch?.test_mse,
            data.xgboost_optuna?.test_mse,
            data.svr_gridsearch?.test_mse,
            data.svr_optuna?.test_mse
        ),
      },
      {
        label: 'Test RMSE',
        xG: formatNumber(data.xgboost_gridsearch?.test_rmse),
        xO: formatNumber(data.xgboost_optuna?.test_rmse),
        sG: formatNumber(data.svr_gridsearch?.test_rmse),
        sO: formatNumber(data.svr_optuna?.test_rmse),
        win: getWinner(
            data.xgboost_gridsearch?.test_rmse,
            data.xgboost_optuna?.test_rmse,
            data.svr_gridsearch?.test_rmse,
            data.svr_optuna?.test_rmse
        ),
      },
      {
        label: 'Execution Time (seconds)',
        xG: formatNumber(data.xgboost_gridsearch?.execution_time_seconds),
        xO: formatNumber(data.xgboost_optuna?.execution_time_seconds),
        sG: formatNumber(data.svr_gridsearch?.execution_time_seconds),
        sO: formatNumber(data.svr_optuna?.execution_time_seconds),
        win: getWinner(
            data.xgboost_gridsearch?.execution_time_seconds,
            data.xgboost_optuna?.execution_time_seconds,
            data.svr_gridsearch?.execution_time_seconds,
            data.svr_optuna?.execution_time_seconds
        ),
      },
    ];

    // Build table
    const tbody = $('#comparisonBody');
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td>${r.label}</td>
        <td>${r.xG}</td>
        <td>${r.xO}</td>
        <td>${r.sG}</td>
        <td>${r.sO}</td>
        <td class="winner">🏆 ${r.win}</td>
      </tr>`
      )
      .join('');

    function renderParams(containerId, params, paramOrder) {
      if (!params) return;
      const ul = $(`#${containerId}`);
      ul.innerHTML = paramOrder
        .map((key) => {
          const val = params[key];
          const display = val != null ? val : '—';
          const dimClass = val == null ? ' dim' : '';
          return `<li><span class="param-key">${key}</span><span class="param-val${dimClass}">${display}</span></li>`;
        })
        .join('');
    }

    renderParams('xgbGridParams', data.xgboost_gridsearch?.best_params, ['n_estimators', 'learning_rate', 'max_depth', 'gamma', 'random_state']);
    renderParams('xgbOptunaParams', data.xgboost_optuna?.best_params, ['n_estimators', 'learning_rate', 'max_depth', 'gamma', 'subsample', 'colsample_bytree', 'random_state']);
    renderParams('svrGridParams', data.svr_gridsearch?.best_params, ['C', 'gamma', 'epsilon', 'kernel']);
    renderParams('svrOptunaParams', data.svr_optuna?.best_params, ['C', 'gamma', 'epsilon', 'kernel']);

    setTimeout(() => initScrollAnimations(), 50);
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#comparisonErrorMsg').textContent = err.message || 'Failed to load comparison data.';
  }
}

// ─── Section 4: Feature Importance ─────────────────────────────

let featureChartInstance = null;

async function loadFeatureImportance() {
  const loader = $('#featureLoader');
  const errorBox = $('#featureError');
  const content = $('#featureContent');

  try {
    const data = await fetchJSON('/api/feature-importance');
    if (!data.features || data.features.length === 0) {
        throw new Error("Feature importance data is empty.");
    }

    hide(loader);
    show(content);

    const ctx = $('#featureChart').getContext('2d');

    const barColors = [
      'rgba(59, 130, 246, 0.85)',
      'rgba(99, 102, 241, 0.85)',
      'rgba(139, 92, 246, 0.85)',
      'rgba(168, 85, 247, 0.85)',
      'rgba(6, 182, 212, 0.85)',
      'rgba(16, 185, 129, 0.85)',
      'rgba(245, 158, 11, 0.85)',
    ];

    const borderColors = [
      'rgba(59, 130, 246, 1)',
      'rgba(99, 102, 241, 1)',
      'rgba(139, 92, 246, 1)',
      'rgba(168, 85, 247, 1)',
      'rgba(6, 182, 212, 1)',
      'rgba(16, 185, 129, 1)',
      'rgba(245, 158, 11, 1)',
    ];

    const features = [...data.features].reverse();
    const scores = [...data.scores].reverse();
    const colors = [...barColors].slice(0, data.features.length).reverse();
    const borders = [...borderColors].slice(0, data.features.length).reverse();

    featureChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: features,
        datasets: [
          {
            label: 'Importance Score',
            data: scores,
            backgroundColor: colors,
            borderColor: borders,
            borderWidth: 1,
            borderRadius: 6,
            barThickness: 28,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 15, 42, 0.92)',
            titleColor: '#e8e8f0',
            bodyColor: '#9a9ab8',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { family: 'Inter', weight: '600' },
            bodyFont: { family: 'Inter' },
            callbacks: {
              label: (ctx) => `Score: ${ctx.parsed.x.toFixed(4)}`,
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#6a6a88',
              font: { family: 'Inter', size: 11 },
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.04)',
            },
            title: {
              display: true,
              text: 'Importance Score',
              color: '#6a6a88',
              font: { family: 'Inter', size: 12 },
            },
          },
          y: {
            ticks: {
              color: '#e8e8f0',
              font: { family: 'Inter', size: 12, weight: '600' },
            },
            grid: {
              display: false,
            },
          },
        },
      },
    });

    setTimeout(() => initScrollAnimations(), 50);
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#featureErrorMsg').textContent = err.message || 'Failed to load feature data. (Make sure XGBoost Optuna has been run)';
  }
}

// ─── INIT: Run all when DOM is ready ─────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollAnimations();

  loadPrediction();
  loadChart();
  loadComparison();
  loadFeatureImportance();
});
