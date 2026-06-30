from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import glob
import json
from app2.backend.ml_pipeline import download_yfinance_data, pipeline_1_preprocessing, pipeline_2_preprocessing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')

import xgboost
import joblib

# Load models and scalers
models = {}
scalerX = None
scalery = None

try:
    # --- Load XGBoost Models (menggunakan XGBoost Booster) ---
    models['XG_grid'] = xgboost.XGBRegressor()
    models['XG_grid'].load_model(os.path.join(MODELS_DIR, 'XG_grid_model.json'))
    
    models['XG_optuna'] = xgboost.XGBRegressor()
    models['XG_optuna'].load_model(os.path.join(MODELS_DIR, 'XG_optuna_model.json'))
    
    models['XG_custom'] = xgboost.XGBRegressor()
    models['XG_custom'].load_model(os.path.join(MODELS_DIR, 'XG_custom_model.json'))
    
    # --- Load SVR Models & Scalers (menggunakan Joblib) ---
    models['SVR_grid'] = joblib.load(os.path.join(MODELS_DIR, 'SVR_grid_model.joblib'))
    models['SVR_optuna'] = joblib.load(os.path.join(MODELS_DIR, 'SVR_optuna_model.joblib'))
    
    scalerX = joblib.load(os.path.join(MODELS_DIR, 'scalerX.joblib'))
    scalery = joblib.load(os.path.join(MODELS_DIR, 'scalery.joblib'))
    
except Exception as e:
    print(f"Global error loading models: {e}")

@app.get("/api/metrics")
def get_metrics():
    metrics_data = {}
    for filename in os.listdir(METRICS_DIR):
        if filename.endswith(".json"):
            name = filename.replace("_metrics.json", "")
            with open(os.path.join(METRICS_DIR, filename), 'r') as f:
                metrics_data[name] = json.load(f)
    return JSONResponse(content=metrics_data)

@app.get("/api/predict/all")
def predict_all():
    # Fetch Data
    df_raw = download_yfinance_data()
    
    # Preprocess
    features_p1 = pipeline_1_preprocessing(df_raw)
    features_p2 = pipeline_2_preprocessing(df_raw)
    
    current_price = float(df_raw['Close'].iloc[-1])
    historical_prices = df_raw['Close'].tail(30).tolist()
    
    predictions = {
        "current_price": current_price,
        "historical_prices": historical_prices
    }
    
    # 1. SVR Grid (P1 + Scaling)
    try:
        f_p1_scaled = scalerX.transform(features_p1)
        pred_scaled = models['SVR_grid'].predict(f_p1_scaled)
        pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        predictions['SVR_grid'] = float(pred)
    except Exception as e:
        predictions['SVR_grid'] = str(e)
        
    # 2. SVR Optuna (P1 + Scaling)
    try:
        f_p1_scaled = scalerX.transform(features_p1)
        pred_scaled = models['SVR_optuna'].predict(f_p1_scaled)
        pred = scalery.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        predictions['SVR_optuna'] = float(pred)
    except Exception as e:
        predictions['SVR_optuna'] = str(e)
        
    # 3. XG Grid (P1, No Scaling)
    try:
        pred = models['XG_grid'].predict(features_p1)[0]
        predictions['XG_grid'] = float(pred)
    except Exception as e:
        predictions['XG_grid'] = str(e)
        
    # 4. XG Optuna (P1, No Scaling)
    try:
        pred = models['XG_optuna'].predict(features_p1)[0]
        predictions['XG_optuna'] = float(pred)
    except Exception as e:
        predictions['XG_optuna'] = str(e)
        
    # 5. XG Custom (P2, No Scaling)
    try:
        pred_return = models['XG_custom'].predict(features_p2)[0]
        pred_price = current_price * (1 + float(pred_return))
        predictions['XG_custom'] = float(pred_price)
    except Exception as e:
        predictions['XG_custom'] = str(e)
        
    return JSONResponse(content=predictions)

# Mount plots and frontend
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app2.backend.main:app", host="127.0.0.1", port=8000, reload=True)
