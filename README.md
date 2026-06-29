# 📈 IHSG Stock Price Prediction with XGBoost and SVR: GridSearchCV vs Bayesian Optimization (Optuna) Comparison

---

## 1. Project Summary

Website link: https://project-ml-erv3.onrender.com/

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
3. Tested on an emerging market index (**IHSG**) with high volatility.
4. Introduced a **Pipeline 2 (Custom XGBoost)** which predicts daily returns instead of direct prices, improving the directional accuracy (Hit Rate).

---

## 2. Dataset

| Aspect | Details |
|-------|--------|
| **Subject** | IHSG (Jakarta Composite Index) — Ticker: `^JKSE` |
| **Source** | Yahoo Finance API (`yfinance`) |
| **Period** | January 1, 2010 – December 31, 2025 (~15 years) |
| **Target (Pipeline 1)** | Next day's closing price (T+1) — `Target_Close = Close.shift(-1)` |
| **Target (Pipeline 2)** | Next day's percentage return — `Target_Return = Returns.shift(-1)` |

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

Experiments were conducted by tuning parameters for two algorithms (XGBoost and SVR) using two search approaches in **Pipeline 1**, and an additional custom approach in **Pipeline 2**.

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

### B. Optuna / Bayesian Optimization (Proposed Method)

Smart search based on Bayes' Theorem over an expanded **continuous** space:

**1. XGBoost Optuna Space**
| Parameter | Search Range | Distribution |
|-----------|----------------|------------|
| `n_estimators` | 100 – 1000 | Uniform integer |
| `learning_rate` | 0.001 – 0.1 | **Log-uniform** |
| `max_depth` | 3 – 16 | Uniform integer |
| `gamma` | 0.0001 – 0.1 | **Log-uniform** |
| `random_state` | 42 (fixed) | — |

**2. SVR Optuna Space**
| Parameter | Search Range | Distribution |
|-----------|----------------|------------|
| `C` | 1.0 – 2000.0 | **Log-uniform** |
| `gamma` | 0.001 – 10.0 | **Log-uniform** |
| `epsilon` | 0.0001 – 0.1 | **Log-uniform** |
| `kernel` | 'rbf' (fixed) | — |

---

## 5. Experiment Results

### Best Parameters

#### 1. XGBoost (Pipeline 1) & Custom (Pipeline 2)
| Parameter | GridSearchCV | Optuna | Custom (Optuna) |
|-----------|-------------|--------|-----------------|
| `n_estimators` | 400 | 184 | 147 |
| `learning_rate` | 0.05 | 0.0677 | 0.0012 |
| `max_depth` | 12 | 4 | 13 |
| `gamma` | 0.01 | 0.0365 | 0.0005 |

> 💡 **XGBoost Insight**: Optuna found that a **shallower** model (`max_depth=4`) with **fewer trees** (`n_estimators=184`) is optimal for Pipeline 1. However, the custom approach for return prediction leaned towards deeper trees (`max_depth=13`).

#### 2. Support Vector Regression (SVR)
| Parameter | GridSearchCV | Optuna |
|-----------|-------------|--------|
| `C` | 1000.0 | 187.20 |
| `gamma` | 0.001 | 0.0060 |
| `epsilon` | 0.001 | 0.0121 |
| `kernel` | rbf | rbf |

### Performance Comparison

#### 1. XGBoost Performance
| Metric | GridSearchCV | Optuna | Custom (Return-based) | Winner |
|--------|--------------|--------|-----------------------|----------|
| **Test MSE** | 853,577.93 | 840,369.95 | 9.7e-05* | ✅ Optuna / Custom* |
| **Test RMSE** | 923.89 | 916.72 | 0.0099* | ✅ Optuna / Custom* |
| **Test MAE** | 804.07 | 781.95 | 0.0070* | ✅ Optuna / Custom* |
| **MAPE** | 10.65% | 10.32% | 119.39%* | ✅ Optuna |
| **Hit Rate** | 46.19% | 46.54% | **49.22%** | ✅ Custom |
| **Execution Time**| 508.18 s | 39.57 s | 34.84 s | ✅ Custom / Optuna |

*\* Custom predicts returns, thus absolute scale metrics (MSE/RMSE/MAE) are fundamentally incomparable.*

#### 2. SVR Performance
| Metric | GridSearchCV | Optuna | Winner |
|--------|--------------|--------|----------|
| **Test MSE** | 6,066.97 | 12,268.37 | ✅ GridSearchCV |
| **Test RMSE** | 77.89 | 110.76 | ✅ GridSearchCV |
| **Test MAE** | 59.15 | 82.54 | ✅ GridSearchCV |
| **MAPE** | 0.82% | 1.10% | ✅ GridSearchCV |
| **Hit Rate** | **47.75%** | 46.71% | ✅ GridSearchCV |
| **Execution Time**| 118.56 s | 65.01 s | ✅ Optuna |

### Results Summary

```text
┌─────────────────────────────────────────────────────────────┐
│  Overall Findings:                                          │
│                                                             │
│  🏆 SVR significantly outperforms XGBoost (Pipeline 1)!     │
│     SVR achieved ~0.82% MAPE compared to XGBoost's ~10%.    │
│     SVR Grid Search found the lowest errors overall.        │
│                                                             │
│  ⚡ Optuna is vastly faster than GridSearch.                │
│     Across both XGBoost and SVR, Optuna found parameters    │
│     in a fraction of the time, although GridSearch edged    │
│     out slightly better accuracy in SVR.                    │
│                                                             │
│  🎯 Pipeline 2 (XGBoost Custom) excels at Directionality!   │
│     While Pipeline 1 minimizes price error, Pipeline 2      │
│     predicts daily returns, achieving the highest Hit Rate  │
│     (49.22%), making it highly valuable for trading signals.│
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Results Visualization

### Actual vs Predicted Price (Test Set)
Comparison plot of actual IHSG prices vs model predictions during the test period (~2023–2025).  
📁 `outputs/plots/actual_vs_predicted.png`

### Feature Importance
Bar chart showing the relative importance score of each technical indicator after optimization.  
📁 `outputs/plots/feature_importance.png`

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
│   ├── pipeline_1/
│   │   ├── XGBoost_GridSearchCV.ipynb # XGBoost predicting Close
│   │   ├── XGBoost_Optuna.ipynb       # XGBoost predicting Close
│   │   ├── SVR_GridSearchCV.ipynb     # SVR predicting Close
│   │   └── SVR_Optuna.ipynb           # SVR predicting Close
│   └── pipeline_2/
│       └── XGBoost_Custom.ipynb       # XGBoost predicting Returns
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

# 3. Open and sequentially run the cells in the desired model notebook (e.g., notebooks/pipeline_1/XGBoost_Optuna.ipynb). Evaluation results and plots will be automatically saved in the `outputs/` folder.
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
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean absolute error |
| **MAPE** | $\frac{100\%}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Mean absolute percentage error — intuitive and easy to interpret |
| **Hit Rate**| $\frac{100\%}{N} \sum I \left( \text{sgn}(\text{Pred}) = \text{sgn}(\text{Actual}) \right)$| Measures how accurately the model predicts the *direction* (up/down) of the market |

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

1. **SVR heavily outperforms XGBoost (Pipeline 1)**: SVR achieves an exceptional MAPE of ~0.82% with an RMSE of ~78 points, far superior to XGBoost's 10% MAPE. This indicates that the margin-based regression of SVR with scaled features is much better suited for capturing the continuous patterns of the stock market index compared to tree-based models when directly predicting stock prices.

2. **Optuna (Bayesian Optimization) speed advantage**: Optuna is significantly faster than GridSearchCV. For XGBoost, it ran in ~40 seconds compared to ~508 seconds for Grid Search.

3. **Pipeline 2 (Custom Return Prediction) finds Edge**: While Pipeline 1 has low error margins for SVR, Pipeline 2 (XGBoost predicting returns rather than direct closing price) yields the highest **Hit Rate (49.22%)**, demonstrating its superior capability in forecasting the direction of daily price movements.

4. **Different strategies via Bayesian Optimization**: For XGBoost, Optuna found a parameter configuration that is **significantly different** from GridSearch — resulting in a shallower model (`max_depth=4`) but with fewer trees (`n_estimators=184`), indicating a very different optimum in the continuous space.

---

## 12. References

1. Zhang, Y. (2023). *Stock Price Prediction Method Based on XGboost Algorithm*. ICBBEM 2022, AHIS 5, pp. 595–603. DOI: [10.2991/978-94-6463-030-5_60](https://doi.org/10.2991/978-94-6463-030-5_60)
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD, pp. 785–794.
3. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD, pp. 2623–2631.
