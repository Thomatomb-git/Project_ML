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

| Model | Best Parameters |
|-------|-----------------|
| **XGBoost Custom** | `n_estimators`: 135, `learning_rate`: 0.0533, `max_depth`: 9, `gamma`: 0.0016 |
| **XGBoost Optuna** | `n_estimators`: 463, `learning_rate`: 0.0236, `max_depth`: 4, `gamma`: 0.0072 |
| **XGBoost Grid** | `n_estimators`: 400, `learning_rate`: 0.05, `max_depth`: 12, `gamma`: 0.01 |
| **SVR Grid** | `C`: 1000.0, `epsilon`: 0.001, `gamma`: 0.001, `kernel`: 'rbf' |
| **SVR Optuna** | `C`: 81.94, `epsilon`: 0.0092, `gamma`: 0.0052, `kernel`: 'rbf' |

<br>

| Model | MSE (Test) | RMSE (Test) | MAE (Test) | MAPE | Hit Rate (%) |
| :---  | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Custom** | 0,0000979 | 0,0099 | 0,0070 | 112,0417 | **54,58%** |
| **XGBoost Optuna** | 837.352,7826 | 915,0698 | 781,9048 | 10,3236 | 46,89% |
| **XGBoost Grid** | 853.577,9333 | 923,8928 | 804,0676 | 10,6471 | 46,19% |
| **SVR Grid** | 6.066,9661 | 77,8907 | 59,1545 | 0,8161 | 47,75% |
| **SVR Optuna** | 9.495,7411 | 97,4461 | 74,4033 | 1,0030 | 46,54% |

*\*Note: The MSE, MAPE, RMSE and MAE for the XGBoost Custom model evaluate the Percentage Return error, not the absolute price error, hence the scale difference.*

---

## 🌐 Web Application

The project features a **Premium Web Dashboard** built with FastAPI and Vanilla Web Technologies (HTML/CSS/JS) featuring Glassmorphism UI, Dark Mode, and dynamic Chart.js visualizations.

The dashboard connects directly to live Yahoo Finance data to predict tomorrow's IHSG price and compares historical backtesting metrics visually.

🔗 **Live Deployment:** [https://project-ml-erv3.onrender.com](https://project-ml-erv3.onrender.com)
