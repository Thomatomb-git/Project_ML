# IHSG Price Predictor

Welcome to the **IHSG Price Predictor** project! This repository contains a machine learning pipeline and a fully deployed web application designed to forecast the movement of the Jakarta Composite Index (IHSG / `^JKSE`).

## 📄 Baseline Research
The methodology and feature engineering approaches in this project are heavily inspired by academic research on stock price prediction. The baseline reference for this project is based on the paper included in this repository:
[`Stock_Price_Prediction_Method_Based_on_XGboost_Alg.pdf`](./Stock_Price_Prediction_Method_Based_on_XGboost_Alg.pdf)

---

## 🛠️ Data Preprocessing & Pipelines

To experiment with different prediction strategies, the project implements two distinct data preprocessing pipelines:

### Pipeline 1 (Used for Standard XGBoost & SVR Models)
This pipeline focuses on standard technical indicators to predict the **Direct Close Price** (`Target_Close`).
- **Indicators Used:**
  - `EMA_9` (Exponential Moving Average, 9-day)
  - `SMA_5`, `SMA_15`, `SMA_30` (Simple Moving Averages)
  - `RSI` (Relative Strength Index, 14-day)
  - `MACD` & `MACD_signal`

### Pipeline 2 (Used for Custom XGBoost Model)
This pipeline is heavily optimized for momentum and volatility tracking. Instead of predicting the price directly, it predicts the **Percentage Return** (`Return_Target`), which is then converted back to price.
- **Indicators Used:**
  - `Return_Lag_1` through `Return_Lag_5`
  - `RSI_14`
  - `MACD_Hist` (MACD Histogram)
  - `Normalized_ATR` (Average True Range)
  - `Vol_to_SMA_20` (Volume ratio)
  - `Rolling_Mom_5` (Rolling Momentum)
  - Calendar Features (`Day_of_Week`, `Is_Month_End`)

---

## 🔄 Reading Flow

1. **Data Preparation:** `notebooks/Data_Preparation.ipynb` *(Generates the processed datasets for both pipelines)*
2. **Model 1:** `XGBoost Grid` *(Hyperparameter tuning using GridSearchCV)*
3. **Model 2:** `XGBoost Optuna` *(Bayesian optimization using Optuna)*
4. **Model 3:** `XGBoost Custom` *(Uses Pipeline 2 for return prediction)*
5. **Model 4:** `SVR Grid` *(Support Vector Regression tuning)*
6. **Model 5:** `SVR Optuna` *(SVR optimization using Optuna)*

---

## 📊 Model Performance & Best Parameters

Below is the summary of the best hyperparameters found and the resulting evaluation metrics on the test dataset.

| Model | Best Parameters | Test RMSE | Test MAE | Hit Rate (%) |
|-------|-----------------|-----------|----------|--------------|
| **XGBoost Custom** | `n_estimators`: 147, `learning_rate`: 0.00118, `max_depth`: 13, `gamma`: 0.00048 | 0.0098* | 0.0070* | **49.22%** |
| **SVR Grid** | `C`: 1000.0, `epsilon`: 0.001, `gamma`: 0.001, `kernel`: 'rbf' | 77.89 | 59.15 | 47.75% |
| **SVR Optuna** | `C`: 187.19, `epsilon`: 0.012, `gamma`: 0.006, `kernel`: 'rbf' | 110.76 | 82.54 | 46.71% |
| **XGBoost Optuna** | `n_estimators`: 184, `learning_rate`: 0.067, `max_depth`: 4, `gamma`: 0.036 | 916.71 | 781.94 | 46.53% |
| **XGBoost Grid** | `n_estimators`: 400, `learning_rate`: 0.05, `max_depth`: 12, `gamma`: 0.01 | 923.89 | 804.06 | 46.19% |

*\*Note: The RMSE and MAE for the XGBoost Custom model evaluate the Percentage Return error, not the absolute price error, hence the scale difference.*

---

## 🌐 Web Application

The project features a **Premium Web Dashboard** built with FastAPI and Vanilla Web Technologies (HTML/CSS/JS) featuring Glassmorphism UI, Dark Mode, and dynamic Chart.js visualizations.

The dashboard connects directly to live Yahoo Finance data to predict tomorrow's IHSG price and compares historical backtesting metrics visually.

🔗 **Live Deployment:** [https://project-ml-erv3.onrender.com](https://project-ml-erv3.onrender.com)
