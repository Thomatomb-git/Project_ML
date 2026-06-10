# 📈 IHSG Stock Price Prediction with XGBoost and SVR: GridSearchCV vs Bayesian Optimization (Optuna) Comparison

---

## 1. Project Summary

Website link: https://projectmlback-production.up.railway.app/

This project aims to predict the **daily closing price** of the Jakarta Composite Index (**IHSG / ^JKSE**) using **XGBoost** and **Support Vector Regression (SVR)** algorithms with a regression approach based on technical indicators. The project also features an interactive **Web Application** (FastAPI) to visualize and test predictions in real-time.

The core of the research is to **compare two hyperparameter tuning methods**:

| | **GridSearchCV** (Baseline) | **Optuna / Bayesian Optimization** (Proposed) |
|---|---|---|
| **Strategy** | Exhaustive discrete search | Probabilistic smart search |
| **Pros** | Guarantees all combinations are tested | Faster & can find parameters in continuous space |
| **Cons** | Slow, limited to a discrete grid | Does not guarantee a global optimum |

### Reference Paper (Baseline)
> **"Stock Price Prediction Method Based on XGboost Algorithm"**  
> Yifan Zhang — Lanzhou University of Technology, 2023  
> DOI: [10.2991/978-94-6463-030-5_60](https://doi.org/10.2991/978-94-6463-030-5_60)

This project's contributions relative to the paper above:
1. Replaced the tuning method from GridSearchCV to **Bayesian Optimization (Optuna)** as the proposed method.
2. Expanded the parameter search space to **continuous space** (instead of discrete).
3. Added anti-overfitting parameters (`subsample`, `colsample_bytree`).
4. Tested on an emerging market index (**IHSG**) with high volatility.

---

## 2. Dataset

| Aspect | Details |
|-------|--------|
| **Subject** | IHSG (Jakarta Composite Index) — Ticker: `^JKSE` |
| **Source** | Yahoo Finance API (`yfinance`) |
| **Period** | January 1, 2010 – December 31, 2025 (~15 years) |
| **Target** | Next day's closing price (T+1) — `Target_Close = Close.shift(-1)` |

### Data Split (Sequential / Chronological)

```
|◀──────── 70% Train ─────────▶|◀── 15% Val ──▶|◀── 15% Test ──▶|
       2010 — ~2021                ~2021-2023         ~2023-2025
```

> ⚠️ **Note**: The split is done **chronologically (sequential split)**, rather than randomly, to conform to the nature of time-series data and prevent data leakage.

---

## 3. Feature Engineering (Technical Indicators)

The model utilizes **7 technical indicators** calculated from the closing price (`Close`) as input features:

| # | Feature | Description | Window |
|---|-------|-----------|--------|
| 1 | **EMA_9** | Exponential Moving Average | 9 days |
| 2 | **SMA_5** | Simple Moving Average (short-term) | 5 days |
| 3 | **SMA_15** | Simple Moving Average (mid-term) | 15 days |
| 4 | **SMA_30** | Simple Moving Average (long-term) | 30 days |
| 5 | **RSI** | Relative Strength Index — momentum oscillator | 14 days |
| 6 | **MACD** | Moving Average Convergence Divergence (EMA12 − EMA26) | 12/26 |
| 7 | **MACD_signal** | MACD Signal line | span 9 |

Library used: [`ta`](https://github.com/bukosabino/ta) (Technical Analysis Library in Python)

---

## 4. Hyperparameter Tuning

Experiments were conducted by tuning parameters for two algorithms (XGBoost and SVR) using two search approaches:

### A. GridSearchCV (Baseline — Following Reference Paper)

Exhaustive search on the following discrete grid (based on Table 1 of the paper for XGBoost, with additions for SVR):

**1. XGBoost Grid**
| Parameter | Candidate Values |
|-----------|---------------|
| `n_estimators` | [100, 200, 300, 400] |
| `learning_rate` | [0.001, 0.005, 0.01, 0.05] |
| `max_depth` | [8, 10, 12, 15] |
| `gamma` | [0.001, 0.005, 0.01, 0.02] |
| `random_state` | 42 |

**2. SVR Grid**
| Parameter | Candidate Values |
|-----------|---------------|
| `C` | [1.0, 10.0, 100.0, 1000.0, 10000.0] |
| `gamma` | [0.001, 0.01, 0.1, 1.0, 10.0] |
| `epsilon` | [0.0001, 0.001, 0.01, 0.1] |
| `kernel` | ['rbf'] |

**Total combinations: 256** — each combination was trained & evaluated on the validation set.

### B. Optuna / Bayesian Optimization (Proposed Method)

Smart search based on Bayes' Theorem over an expanded **continuous** space:

**1. XGBoost Optuna Space**
| Parameter | Search Range | Distribution |
|-----------|----------------|------------|
| `n_estimators` | 100 – 1000 | Uniform integer |
| `learning_rate` | 0.001 – 0.1 | **Log-uniform** |
| `max_depth` | 3 – 16 | Uniform integer |
| `gamma` | 0.0001 – 0.1 | **Log-uniform** |
| `subsample` | 0.6 – 1.0 | Uniform float |
| `colsample_bytree` | 0.6 – 1.0 | Uniform float |
| `random_state` | 42 (fixed) | — |

**2. SVR Optuna Space**
| Parameter | Search Range | Distribution |
|-----------|----------------|------------|
| `C` | 1.0 – 2000.0 | **Log-uniform** |
| `gamma` | 0.001 – 10.0 | **Log-uniform** |
| `epsilon` | 0.0001 – 0.1 | **Log-uniform** |
| `kernel` | 'rbf' (fixed) | — |

**Total iterations: 50 trials** — each iteration learns from the results of the previous iterations.

---

## 5. Experiment Results

### Best Parameters

| Parameter | GridSearchCV | Optuna |
|-----------|-------------|--------|
| `n_estimators` | 400 | 930 |
| `learning_rate` | 0.05 | 0.0367 |
| `max_depth` | 12 | 3 |
| `gamma` | 0.01 | 0.0015 |
| `subsample` | — | 0.6013 |
| `colsample_bytree` | — | 0.7722 |

> 💡 **Insight**: Optuna found that a **shallower** model (`max_depth=3`) with **more trees** (`n_estimators=930`) and additional regularization (`subsample`, `colsample_bytree`) is more optimal — contrary to Grid Search which selected deeper trees (`max_depth=12`).

### Performance Comparison

| Metric | GridSearchCV (Baseline) | Optuna (Proposed) | Winner |
|--------|------------------------|-------------------|----------|
| **Validation MSE** | 127,778.94 | 102,715.10 | ✅ Optuna |
| **Test MSE** | 853,577.93 | 817,097.75 | ✅ Optuna |
| **Test RMSE** | 923.89 | 903.93 | ✅ Optuna |
| **Execution Time** | 122.40 seconds | 21.85 seconds | ✅ Optuna |
| **MAPE** | ~10.65% | ~10.14% | ✅ Optuna |

### Results Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Optuna produced a model that is:                           │
│                                                             │
│  📉 MSE 4.3% lower than GridSearch                          │
│  📉 RMSE 2.2% lower than GridSearch                         │
│  ⚡ 5.6x faster than GridSearch                             │
│  📊 MAPE ~10.14% (average miss of 10% from actual price)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Results Visualization

### Actual vs Predicted Price (Test Set)
Comparison plot of actual IHSG prices vs model predictions during the test period (~2023–2025).  
📁 `outputs/plots/actual_vs_predicted.png`

### Feature Importance
Bar chart showing the relative importance score of each technical indicator after Bayesian optimization.  
📁 `outputs/plots/feature_importance.png`

### Percentage Error Analysis
Daily percentage error chart and error distribution histogram for each model.  
📁 `outputs/plots/percentage_error_xgboost__optuna.png`  
📁 `outputs/plots/percentage_error_xgboost__gridsearch.png`

---

## 7. Project Structure

```
Project_ML/
│
├── README.md                  # Project documentation (this file)
├── Documentation.md           # Detailed research specifications
├── requirements.txt           # Python dependencies
├── Procfile                   # Web deployment configuration file
│
├── app/                       # Web Application
│   ├── back/                  # FastAPI Backend (main.py)
│   └── front/                 # UI Frontend (index.html, style.css, script.js)
│
├── notebooks/                 # Data Preparation & Model Experiments
│   ├── Data_Preparation.ipynb     # Download data & feature engineering
│   ├── XGBoost_GridSearchCV.ipynb # XGBoost using GridSearch
│   ├── XGBoost_Optuna.ipynb       # XGBoost using Optuna
│   ├── SVR_GridSearchCV.ipynb     # SVR using GridSearch
│   └── SVR_Optuna.ipynb           # SVR using Optuna
│
├── data/
│   ├── raw/                   # Raw data from Yahoo Finance
│   └── processed/             # Data after feature engineering
│
└── outputs/
    ├── models/                # Saved XGBoost & SVR models (.pkl)
    ├── metrics/               # Performance metric results (.json)
    └── plots/                 # Visualization plots (.png)
```

---

## 8. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Experiment Execution Steps (Notebooks)
```bash
# 1. Run Jupyter Notebook
jupyter notebook

# 2. Open and run `notebooks/Data_Preparation.ipynb` to download raw data and process features (technical indicators).

# 3. Open and sequentially run the cells in the desired model notebook (e.g., XGBoost_Optuna.ipynb). Evaluation results and plots will be automatically saved in the `outputs/` folder.
```

### Running the Web Application
Once the model is trained and saved in the `outputs/models/` folder, you can run the interactive Web App.

**1. Running the Backend (FastAPI)**
```bash
cd app/back
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
The Backend API will run at `http://localhost:8000`. Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.

**2. Running the Frontend**
Open the `app/front/index.html` file using your web browser, or use an extension like Live Server in VSCode.
The web interface will automatically connect to your local backend.

---

## 9. Evaluation Metrics

| Metric | Formula | Use Case |
|--------|-------|----------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Measures mean squared error — sensitive to outliers |
| **RMSE** | $\sqrt{MSE}$ | Same as MSE but in original units (index points) |
| **MAPE** | $\frac{1}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\| \times 100\%$ | Mean absolute percentage error — intuitive and easy to interpret |

---

## 10. Technologies & Libraries

| Library | Version | Function |
|---------|-------|--------|
| `xgboost` | — | Main model algorithm |
| `scikit-learn` | — | Evaluation metrics & ParameterGrid |
| `optuna` | — | Bayesian hyperparameter optimization |
| `yfinance` | — | Stock data download |
| `ta` | — | Technical indicator calculation |
| `pandas` | — | Data manipulation |
| `numpy` | — | Numerical operations |
| `matplotlib` | — | Plot visualization |
| `seaborn` | — | Visualization styling |
| `fastapi` | — | Web App Backend Framework |
| `uvicorn` | — | ASGI Server for FastAPI |

---

## 11. Conclusion

1. **Optuna (Bayesian Optimization) proves superior** to GridSearchCV across all evaluation metrics: lower MSE, lower RMSE, and **5.6x faster** execution time.

2. Optuna found a parameter configuration that is **significantly different** from GridSearch — resulting in a shallower model (`max_depth=3`) but with more trees (`n_estimators=930`), indicating a more robust ensemble strategy against overfitting.

3. Both models have a **MAPE of around 10%**, suggesting an inherent limitation in predicting stock prices using technical indicators alone, especially for time horizons far from the training data.

4. This project resides entirely in the domain of **classical Machine Learning** — utilizing an ensemble method (gradient boosting) based on decision trees, rather than deep learning / neural networks.

---

## 12. References

1. Zhang, Y. (2023). *Stock Price Prediction Method Based on XGboost Algorithm*. ICBBEM 2022, AHIS 5, pp. 595–603. DOI: [10.2991/978-94-6463-030-5_60](https://doi.org/10.2991/978-94-6463-030-5_60)
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD, pp. 785–794.
3. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD, pp. 2623–2631.
