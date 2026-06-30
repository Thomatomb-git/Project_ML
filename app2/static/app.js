const API_URL = "/api";

let metricsData = {};
let predictionsData = null;
let currentModel = 'XG_custom';

const modelNamesMap = {
    'XG_custom': 'XGBoost Custom',
    'XG_grid': 'XGBoost GridSearchCV',
    'XG_optuna': 'XGBoost Optuna',
    'SVR_grid': 'SVR GridSearchCV',
    'SVR_optuna': 'SVR Optuna'
};

document.addEventListener("DOMContentLoaded", async () => {
    await fetchMetrics();
    initChart();
    updateModelDetails();
    
    // Model selection logic
    const buttons = document.querySelectorAll('.model-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            buttons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentModel = e.target.getAttribute('data-model');
            updateModelDetails();
        });
    });

    // Predict button
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
});

async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        metricsData = await response.json();
    } catch (error) {
        console.error("Error fetching metrics:", error);
    }
}

function updateModelDetails() {
    if (!metricsData[currentModel]) return;

    const data = metricsData[currentModel];
    
    document.getElementById('detailModelName').innerText = modelNamesMap[currentModel];
    document.getElementById('detailModelMethod').innerText = data.method || currentModel;
    
    // Format params nicely
    document.getElementById('detailParams').innerText = JSON.stringify(data.best_params, null, 2);
    
    // Format metrics
    const metricsHtml = `
        <li>Test RMSE: <span>${data.test_rmse ? data.test_rmse.toFixed(6) : 'N/A'}</span></li>
        <li>Test MAE: <span>${data.test_mae ? data.test_mae.toFixed(6) : 'N/A'}</span></li>
        <li>MAPE: <span>${data.mape ? data.mape.toFixed(2) + '%' : 'N/A'}</span></li>
        <li>Hit Rate: <span>${data.hit_rate_percentage ? data.hit_rate_percentage.toFixed(2) + '%' : 'N/A'}</span></li>
    `;
    document.getElementById('detailMetrics').innerHTML = metricsHtml;
}

async function runPrediction() {
    const btn = document.getElementById('predictBtn');
    btn.innerText = "Predicting...";
    btn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict/all`);
        predictionsData = await response.json();
        
        // Update Hero 1
        document.getElementById('currentPrice').innerText = predictionsData.current_price.toFixed(2);
        
        if (predictionsData.historical_prices) {
            renderSparkline(predictionsData.historical_prices);
        }
        
        let customVal = predictionsData.XG_custom;
        document.getElementById('predictedPriceXGBCustom').innerText = typeof customVal === 'number' ? customVal.toFixed(2) : String(customVal);

    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Failed to run prediction. Ensure backend is running.");
    } finally {
        btn.innerText = "Run Live Prediction";
        btn.disabled = false;
    }
}

let comparisonChartInstance = null;

function initChart() {
    if (Object.keys(metricsData).length === 0) return;

    const select = document.getElementById('metricSelect');
    
    // Initial Render
    renderChart(select.value);

    // On Change
    select.addEventListener('change', (e) => {
        renderChart(e.target.value);
    });

    // Backtest Model Selector (Hero 4)
    const backtestSelect = document.getElementById('backtestModelSelect');
    const backtestImage = document.getElementById('backtestImage');
    
    backtestSelect.addEventListener('change', (e) => {
        const selectedModel = e.target.value;
        backtestImage.src = `/plots/price_movement_${selectedModel}.png`;
    });
}

function renderChart(metricKey) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    const labels = Object.keys(modelNamesMap).map(k => modelNamesMap[k]);
    const keys = Object.keys(modelNamesMap);
    
    const dataValues = keys.map(k => metricsData[k] ? metricsData[k][metricKey] : 0);

    const isErrorMetric = ['test_rmse', 'test_mae', 'test_mse', 'mape'].includes(metricKey);
    const warningEl = document.getElementById('metricWarning');
    
    if (isErrorMetric) {
        warningEl.style.display = 'block';
    } else {
        warningEl.style.display = 'none';
    }

    // Colors
    let bgColor = 'rgba(0, 240, 255, 0.7)';
    let borderColor = 'rgba(0, 240, 255, 1)';
    if (metricKey === 'hit_rate_percentage') {
        bgColor = 'rgba(0, 255, 102, 0.7)';
        borderColor = 'rgba(0, 255, 102, 1)';
    } else if (isErrorMetric) {
        bgColor = 'rgba(255, 0, 60, 0.7)';
        borderColor = 'rgba(255, 0, 60, 1)';
    }

    const metricLabelMap = {
        'hit_rate_percentage': 'Hit Rate (%)',
        'test_rmse': 'RMSE',
        'test_mae': 'MAE',
        'test_mse': 'MSE',
        'mape': 'MAPE (%)'
    };

    if (comparisonChartInstance) {
        comparisonChartInstance.destroy();
    }

    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: metricLabelMap[metricKey] || metricKey,
                    data: dataValues,
                    backgroundColor: bgColor,
                    borderColor: borderColor,
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#8b9bb4' }
                },
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#8b9bb4' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff' }
                }
            }
        }
    });
}

let sparklineChartInstance = null;

function renderSparkline(historicalPrices) {
    if (!historicalPrices || historicalPrices.length === 0) return;
    
    const ctx = document.getElementById('sparklineChart').getContext('2d');
    
    const firstPrice = historicalPrices[0];
    const lastPrice = historicalPrices[historicalPrices.length - 1];
    const isUp = lastPrice >= firstPrice;
    
    const lineColor = isUp ? 'rgba(0, 255, 102, 1)' : 'rgba(255, 0, 60, 1)';
    const bgColor = isUp ? 'rgba(0, 255, 102, 0.1)' : 'rgba(255, 0, 60, 0.1)';

    if (sparklineChartInstance) {
        sparklineChartInstance.destroy();
    }
    
    const labels = historicalPrices.map((_, i) => i);

    sparklineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: historicalPrices,
                borderColor: lineColor,
                backgroundColor: bgColor,
                borderWidth: 2,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            layout: { padding: 0 }
        }
    });
}
