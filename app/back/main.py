# Jalankan dari root project:
# uvicorn app.back.main:app --reload

"""
Backend FastAPI untuk dashboard prediksi saham IHSG.

Modul ini menyediakan API endpoint untuk:
- Prediksi harga saham menggunakan model XGBoost (Optuna) dan SVR (Optuna)
- Data chart perbandingan aktual vs prediksi (XGBoost & SVR) pada test set
- Perbandingan metrik antara model GridSearch dan Optuna untuk XGBoost & SVR
- Feature importance dari model XGBoost Optuna
"""

import json
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from xgboost import XGBRegressor
import joblib

# ──────────────────────────────────────────────
# Path resolution – semua relatif terhadap root project
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_XGB_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_optuna.json"
MODEL_XGB_GRID_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_gridsearch.json"
MODEL_SVR_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "models" / "svr_optuna.pkl"
MODEL_SVR_GRID_PATH = PROJECT_ROOT / "outputs" / "models" / "svr_gridsearch.pkl"
SCALER_X_PATH = PROJECT_ROOT / "outputs" / "models" / "scalerX.pkl"
SCALER_Y_PATH = PROJECT_ROOT / "outputs" / "models" / "scalery.pkl"

METRICS_XGB_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "optuna_results.json"
METRICS_XGB_GRID_PATH = PROJECT_ROOT / "outputs" / "metrics" / "gridsearch_results.json"
METRICS_SVR_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "optuna_SVR_results.json"
METRICS_SVR_GRID_PATH = PROJECT_ROOT / "outputs" / "metrics" / "gridsearch_SVR_results.json"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ihsg_processed_features.csv"
FRONTEND_DIR = PROJECT_ROOT / "app" / "front"

# Kolom fitur yang digunakan model
FEATURE_COLUMNS = [
    "EMA_9",
    "SMA_5",
    "SMA_15",
    "SMA_30",
    "RSI",
    "MACD",
    "MACD_signal",
]

# ──────────────────────────────────────────────
# Model global – dimuat saat startup
# ──────────────────────────────────────────────
model_xgb_optuna = None
model_xgb_gridsearch = None
model_svr_optuna = None
model_svr_gridsearch = None
scaler_X = None
scaler_y = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model XGBoost dan SVR saat aplikasi pertama kali berjalan."""
    global model_xgb_optuna, model_xgb_gridsearch, model_svr_optuna, model_svr_gridsearch, scaler_X, scaler_y

    try:
        model_xgb_optuna = XGBRegressor()
        model_xgb_optuna.load_model(str(MODEL_XGB_OPTUNA_PATH))
        
        model_xgb_gridsearch = XGBRegressor()
        model_xgb_gridsearch.load_model(str(MODEL_XGB_GRID_PATH))
        model_svr_optuna = joblib.load(str(MODEL_SVR_OPTUNA_PATH))
        model_svr_gridsearch = joblib.load(str(MODEL_SVR_GRID_PATH))
        scaler_X = joblib.load(str(SCALER_X_PATH))
        scaler_y = joblib.load(str(SCALER_Y_PATH))
        print("[INFO] Semua model dan scaler berhasil dimuat.")
    except Exception as e:
        print(f"[WARNING] Gagal memuat model/scaler. Pastikan sudah menjalankan notebook: {e}")

    yield  # Aplikasi berjalan

    # Cleanup (jika diperlukan)
    print("[INFO] Aplikasi dimatikan.")


app = FastAPI(
    title="IHSG Prediction Dashboard API",
    description="API backend untuk dashboard prediksi saham IHSG menggunakan XGBoost dan SVR.",
    version="1.0.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# CORS middleware – izinkan semua origin (untuk development)
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Fungsi utilitas
# ──────────────────────────────────────────────


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung indikator teknikal pada DataFrame yang memiliki kolom 'Close'.
    """
    close = df["Close"]

    df["EMA_9"] = EMAIndicator(close=close, window=9).ema_indicator()
    df["SMA_5"] = SMAIndicator(close=close, window=5).sma_indicator()
    df["SMA_15"] = SMAIndicator(close=close, window=15).sma_indicator()
    df["SMA_30"] = SMAIndicator(close=close, window=30).sma_indicator()
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()

    macd_indicator = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd_indicator.macd()
    df["MACD_signal"] = macd_indicator.macd_signal()

    return df


def _split_data_sequential(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Membagi data secara sekuensial (kronologis): 70% train, 15% val, 15% test.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────


@app.get("/api/predict")
async def predict(model_type: str = "xgboost"):
    """
    Endpoint prediksi harga IHSG hari berikutnya.
    Menggunakan model XGBoost atau SVR berdasarkan parameter model_type.
    """
    try:
        # Ambil data IHSG 3 bulan terakhir dari Yahoo Finance
        df = yf.download("^JKSE", period="3mo")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Tidak bisa mengambil data dari Yahoo Finance. Periksa koneksi internet.",
        )

    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Tidak bisa mengambil data dari Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = _compute_technical_indicators(df)
    df = df.dropna(subset=FEATURE_COLUMNS)

    if df.empty:
        raise HTTPException(status_code=503, detail="Data tidak cukup.")

    last_row = df.iloc[-1]
    features_df = pd.DataFrame([last_row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)
    features = features_df.values

    # Prediksi
    if model_type.lower() == "svr":
        if model_svr_optuna is None or scaler_X is None or scaler_y is None:
             raise HTTPException(status_code=500, detail="SVR model atau scaler belum siap.")
        features_scaled = scaler_X.transform(features_df)
        pred_scaled = model_svr_optuna.predict(features_scaled)
        prediction = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
    else:
        if model_xgb_optuna is None:
             raise HTTPException(status_code=500, detail="XGBoost model belum siap.")
        prediction = float(model_xgb_optuna.predict(features)[0])

    last_close = float(last_row["Close"])
    last_date = str(df.index[-1].strftime("%Y-%m-%d"))

    delta = round(prediction - last_close, 2)
    delta_percent = round((delta / last_close) * 100, 2)

    return {
        "last_close": round(last_close, 2),
        "last_date": last_date,
        "prediction": round(prediction, 2),
        "delta": delta,
        "delta_percent": delta_percent,
        "model_used": model_type.lower()
    }


@app.get("/api/chart-data")
async def chart_data():
    """
    Endpoint data chart perbandingan aktual vs prediksi pada test set (XGB & SVR).
    """
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    _, _, test_df = _split_data_sequential(df)

    X_test_df = test_df[FEATURE_COLUMNS]
    X_test = X_test_df.values
    y_actual = test_df["Target_Close"].values

    dates = [d.strftime("%Y-%m-%d") for d in test_df.index]
    
    response = {
        "dates": dates,
        "actual": [round(float(v), 2) for v in y_actual]
    }
    
    if model_xgb_optuna is not None:
        y_predicted_xgb = model_xgb_optuna.predict(X_test)
        response["predicted_xgb"] = [round(float(v), 2) for v in y_predicted_xgb]
    else:
        response["predicted_xgb"] = []
        
    if model_svr_optuna is not None and scaler_X is not None and scaler_y is not None:
        X_test_scaled = scaler_X.transform(X_test_df)
        y_predicted_svr_scaled = model_svr_optuna.predict(X_test_scaled)
        y_predicted_svr = scaler_y.inverse_transform(y_predicted_svr_scaled.reshape(-1, 1)).flatten()
        response["predicted_svr"] = [round(float(v), 2) for v in y_predicted_svr]
    else:
        response["predicted_svr"] = []

    return response


@app.get("/api/comparison")
async def comparison():
    """
    Endpoint perbandingan metrik antara ke-4 model.
    """
    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {
        "xgboost_gridsearch": load_json(METRICS_XGB_GRID_PATH),
        "xgboost_optuna": load_json(METRICS_XGB_OPTUNA_PATH),
        "svr_gridsearch": load_json(METRICS_SVR_GRID_PATH),
        "svr_optuna": load_json(METRICS_SVR_OPTUNA_PATH),
    }


@app.get("/api/feature-importance")
async def feature_importance():
    """
    Endpoint feature importance dari model XGBoost Optuna.
    (SVR non-linear rbf tidak memiliki feature importance)
    """
    if model_xgb_optuna is None:
        return {"features": [], "scores": []}
        
    importances = model_xgb_optuna.feature_importances_
    feature_score_pairs = list(zip(FEATURE_COLUMNS, importances.tolist()))
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return {
        "features": [pair[0] for pair in feature_score_pairs],
        "scores": [round(pair[1], 6) for pair in feature_score_pairs],
    }


# ──────────────────────────────────────────────
# Static files & frontend – di-mount TERAKHIR agar API routes prioritas
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
