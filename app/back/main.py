# Jalankan dari root project:
# uvicorn app.back.main:app --reload

"""
Backend FastAPI untuk dashboard prediksi saham IHSG.

Modul ini menyediakan API endpoint untuk:
- Prediksi harga saham menggunakan model XGBoost (Optuna)
- Data chart perbandingan aktual vs prediksi pada test set
- Perbandingan metrik antara model GridSearch dan Optuna
- Feature importance dari model Optuna
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

# ──────────────────────────────────────────────
# Path resolution – semua relatif terhadap root project
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_optuna.json"
MODEL_GRIDSEARCH_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_gridsearch.json"
METRICS_OPTUNA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "optuna_results.json"
METRICS_GRIDSEARCH_PATH = PROJECT_ROOT / "outputs" / "metrics" / "gridsearch_results.json"
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
model_optuna: XGBRegressor | None = None
model_gridsearch: XGBRegressor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model XGBoost saat aplikasi pertama kali berjalan."""
    global model_optuna, model_gridsearch

    # Muat model Optuna
    model_optuna = XGBRegressor()
    model_optuna.load_model(str(MODEL_OPTUNA_PATH))
    print(f"[INFO] Model Optuna berhasil dimuat dari {MODEL_OPTUNA_PATH}")

    # Muat model GridSearch
    model_gridsearch = XGBRegressor()
    model_gridsearch.load_model(str(MODEL_GRIDSEARCH_PATH))
    print(f"[INFO] Model GridSearch berhasil dimuat dari {MODEL_GRIDSEARCH_PATH}")

    yield  # Aplikasi berjalan

    # Cleanup (jika diperlukan)
    print("[INFO] Aplikasi dimatikan.")


app = FastAPI(
    title="IHSG Prediction Dashboard API",
    description="API backend untuk dashboard prediksi saham IHSG menggunakan XGBoost.",
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

    Indikator yang dihitung:
    - EMA 9, SMA 5/15/30, RSI 14, MACD & MACD Signal

    Returns:
        DataFrame dengan kolom indikator teknikal ditambahkan.
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

    Returns:
        Tuple berisi (train_df, val_df, test_df).
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
async def predict():
    """
    Endpoint prediksi harga IHSG hari berikutnya.

    Mengambil data 3 bulan terakhir dari Yahoo Finance, menghitung
    indikator teknikal, lalu memprediksi harga Close berikutnya
    menggunakan model XGBoost Optuna.

    Returns:
        JSON berisi last_close, last_date, prediction, delta, delta_percent.
    """
    try:
        # Ambil data IHSG 3 bulan terakhir dari Yahoo Finance
        df = yf.download("^JKSE", period="3mo")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Tidak bisa mengambil data dari Yahoo Finance. Periksa koneksi internet.",
        )

    # Validasi apakah data berhasil diambil
    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail="Tidak bisa mengambil data dari Yahoo Finance. Periksa koneksi internet.",
        )

    # Handle multi-level columns dari yfinance (misal ('Close', '^JKSE'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Hitung indikator teknikal
    df = _compute_technical_indicators(df)

    # Buang baris dengan NaN (akibat window indikator)
    df = df.dropna(subset=FEATURE_COLUMNS)

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Data tidak cukup untuk menghitung indikator teknikal.",
        )

    # Ambil baris terakhir (hari trading terbaru)
    last_row = df.iloc[-1]
    features = last_row[FEATURE_COLUMNS].values.reshape(1, -1)

    # Prediksi
    prediction = float(model_optuna.predict(features)[0])
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
    }


@app.get("/api/chart-data")
async def chart_data():
    """
    Endpoint data chart perbandingan aktual vs prediksi pada test set.

    Memuat data processed, membaginya secara kronologis (70/15/15),
    dan mengembalikan prediksi model pada porsi test set.

    Returns:
        JSON berisi dates, actual, dan predicted arrays.
    """
    # Muat data processed
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)

    # Bagi data: 70% train, 15% val, 15% test
    _, _, test_df = _split_data_sequential(df)

    # Siapkan fitur dan target
    X_test = test_df[FEATURE_COLUMNS].values
    y_actual = test_df["Target_Close"].values

    # Prediksi menggunakan model Optuna
    y_predicted = model_optuna.predict(X_test)

    # Format tanggal sebagai string
    dates = [d.strftime("%Y-%m-%d") for d in test_df.index]

    return {
        "dates": dates,
        "actual": [round(float(v), 2) for v in y_actual],
        "predicted": [round(float(v), 2) for v in y_predicted],
    }


@app.get("/api/comparison")
async def comparison():
    """
    Endpoint perbandingan metrik antara model GridSearch dan Optuna.

    Membaca file JSON metrik dari kedua model dan mengembalikannya.

    Returns:
        JSON berisi objek gridsearch dan optuna dengan metrik masing-masing.
    """
    with open(METRICS_GRIDSEARCH_PATH, "r", encoding="utf-8") as f:
        gridsearch_metrics = json.load(f)

    with open(METRICS_OPTUNA_PATH, "r", encoding="utf-8") as f:
        optuna_metrics = json.load(f)

    return {
        "gridsearch": gridsearch_metrics,
        "optuna": optuna_metrics,
    }


@app.get("/api/feature-importance")
async def feature_importance():
    """
    Endpoint feature importance dari model Optuna.

    Mengambil skor feature importance, memetakan ke nama fitur,
    dan mengurutkan secara descending.

    Returns:
        JSON berisi features (nama fitur) dan scores (skor importance),
        diurutkan dari yang paling penting.
    """
    importances = model_optuna.feature_importances_

    # Gabungkan nama fitur dengan skor importance
    feature_score_pairs = list(zip(FEATURE_COLUMNS, importances.tolist()))

    # Urutkan berdasarkan skor descending
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)

    features_sorted = [pair[0] for pair in feature_score_pairs]
    scores_sorted = [round(pair[1], 6) for pair in feature_score_pairs]

    return {
        "features": features_sorted,
        "scores": scores_sorted,
    }


# ──────────────────────────────────────────────
# Static files & frontend – di-mount TERAKHIR agar API routes prioritas
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
