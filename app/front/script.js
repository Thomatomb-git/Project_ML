/* ================================================================
   IHSG Predictor — Main Script
   Mengelola fetch API, Chart.js, animasi scroll, dsb.
   ================================================================ */

// ─── Utilitas ────────────────────────────────────────────────────

/** Format angka ke locale Indonesia (titik ribuan, koma desimal) */
function formatIDR(num, decimals = 2) {
  if (num == null || isNaN(num)) return '—';
  return Number(num).toLocaleString('id-ID', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Helper: tampilkan elemen */
function show(el) { el.classList.remove('hidden'); }
/** Helper: sembunyikan elemen */
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

  // Efek scroll: tambah class 'scrolled'
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 30);
  });

  // Mobile hamburger toggle
  toggle.addEventListener('click', () => {
    links.classList.toggle('open');
  });

  // Tutup menu saat link diklik (mobile)
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

async function loadPrediction() {
  const loader = $('#predictionLoader');
  const errorBox = $('#predictionError');
  const cards = $('#predictionCards');
  const meta = $('#metaLine');

  try {
    const data = await fetchJSON('/api/predict');

    hide(loader);
    show(cards);
    show(meta);

    // Isi card Harga Terakhir
    $('#lastClose').textContent = formatIDR(data.last_close);

    // Isi card Prediksi Besok
    $('#prediction').textContent = formatIDR(data.prediction);

    // Isi card Selisih
    const deltaEl = $('#delta');
    const deltaPercentEl = $('#deltaPercent');
    const isUp = data.delta >= 0;
    const arrow = isUp ? '▲' : '▼';
    deltaEl.textContent = `${arrow} ${formatIDR(Math.abs(data.delta))}`;
    deltaEl.classList.add(isUp ? 'up' : 'down');
    deltaPercentEl.textContent = `${isUp ? '+' : '-'}${formatIDR(Math.abs(data.delta_percent))}%`;
    deltaPercentEl.classList.add(isUp ? 'up' : 'down');

    // Metadata
    meta.textContent = `Data terakhir: ${data.last_date}  •  Model: XGBoost + Optuna`;

    // Trigger animasi scroll untuk card yang baru muncul
    setTimeout(() => initScrollAnimations(), 50);
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#predictionErrorMsg').textContent = err.message || 'Gagal memuat data prediksi.';
  }
}

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

    // Warna dan style
    const actualColor = 'rgba(59, 130, 246, 1)';     // Biru
    const predictedColor = 'rgba(245, 158, 11, 1)';  // Amber/oranye

    mainChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [
          {
            label: 'Aktual',
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
            label: 'Prediksi',
            data: data.predicted,
            borderColor: predictedColor,
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: predictedColor,
            tension: 0.3,
            fill: false,
            borderDash: [6, 3],
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
                return `${ctx.dataset.label}: ${formatIDR(ctx.parsed.y)}`;
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
              callback: (val) => formatIDR(val, 0),
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
    $('#chartErrorMsg').textContent = err.message || 'Gagal memuat data grafik.';
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

    const gs = data.gridsearch;
    const op = data.optuna;

    // Helper: tentukan pemenang (lebih kecil = lebih baik, kecuali waktu eksekusi)
    function winner(gsVal, opVal, lowerBetter = true) {
      if (lowerBetter) return gsVal <= opVal ? 'GridSearchCV' : 'Optuna';
      return gsVal >= opVal ? 'GridSearchCV' : 'Optuna';
    }

    // Hitung MAPE jika tersedia, atau tampilkan "N/A"
    // (MAPE tidak ada di response, kita tampilkan "-" sebagai placeholder)
    const rows = [
      {
        label: 'Validation MSE',
        gs: formatIDR(gs.best_validation_mse),
        op: formatIDR(op.best_validation_mse),
        win: winner(gs.best_validation_mse, op.best_validation_mse),
      },
      {
        label: 'Test MSE',
        gs: formatIDR(gs.test_mse),
        op: formatIDR(op.test_mse),
        win: winner(gs.test_mse, op.test_mse),
      },
      {
        label: 'Test RMSE',
        gs: formatIDR(gs.test_rmse),
        op: formatIDR(op.test_rmse),
        win: winner(gs.test_rmse, op.test_rmse),
      },
      {
        label: 'Waktu Eksekusi (detik)',
        gs: formatIDR(gs.execution_time_seconds),
        op: formatIDR(op.execution_time_seconds),
        win: winner(gs.execution_time_seconds, op.execution_time_seconds),
      },
    ];

    // Bangun tabel
    const tbody = $('#comparisonBody');
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td>${r.label}</td>
        <td>${r.gs}</td>
        <td>${r.op}</td>
        <td class="winner">${r.win === 'Optuna' ? '⚡ ' : '🔧 '}${r.win}</td>
      </tr>`
      )
      .join('');

    // Parameter lists — urutan seragam agar sejajar di kedua kartu
    const paramOrder = [
      'n_estimators', 'learning_rate', 'max_depth', 'gamma',
      'subsample', 'colsample_bytree', 'random_state',
    ];

    function renderParams(containerId, params) {
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

    renderParams('gridParams', gs.best_params);
    renderParams('optunaParams', op.best_params);

    setTimeout(() => initScrollAnimations(), 50);
  } catch (err) {
    hide(loader);
    show(errorBox);
    $('#comparisonErrorMsg').textContent = err.message || 'Gagal memuat data komparasi.';
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

    hide(loader);
    show(content);

    const ctx = $('#featureChart').getContext('2d');

    // Warna gradient untuk setiap bar
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

    // Reverse agar yang terbesar di atas (horizontal bar)
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
        indexAxis: 'y', // Horizontal bar
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
    $('#featureErrorMsg').textContent = err.message || 'Gagal memuat data fitur.';
  }
}

// ─── INIT: Jalankan semua saat DOM siap ─────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Navbar & animasi scroll
  initNavbar();
  initScrollAnimations();

  // Fetch semua data secara paralel
  loadPrediction();
  loadChart();
  loadComparison();
  loadFeatureImportance();
});
