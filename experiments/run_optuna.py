import os
import sys
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna

# Menambahkan root direktori ke sys.path agar bisa membaca config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data

# Menonaktifkan log bawaan Optuna agar terminal tidak terlalu penuh,
# kita hanya akan menampilkan ringkasan iterasi yang penting saja.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    print("=== Menjalankan Metode Usulan: XGBoost + Optuna (Bayesian Optimization) ===")
    
    # 1. Memuat data sekuensial yang konsisten
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    # 2. Definisikan Fungsi Objektif untuk Optuna
    def objective(trial):
        # Mengonfigurasi ruang pencarian parameter yang luas dan kontinu
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 16),
            'gamma': trial.suggest_float('gamma', 1e-4, 0.1, log=True),
            # Menambahkan parameter penjinak overfitting (Celah yang tidak ada di paper 2)
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Latih model menggunakan parameter saran dari Bayesian
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Hitung performa pada Validation Set
        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        
        return val_mse

    # 3. Proses Optuna Study (Mencari titik parameter terbaik)
    start_time = time.time()
    
    # Kita arahkan untuk me-minimize skor MSE pada data validasi
    study = optuna.create_study(direction='minimize')
    
    # Menjalankan 50 iterasi pencarian cerdas (bisa disesuaikan)
    n_trials = 50
    print(f"Memulai pencarian Bayesian sebanyak {n_trials} iterasi...")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    execution_time = time.time() - start_time
    print("\n--- Proses Optimasi Cerdas Selesai ---")
    print(f"Waktu Eksekusi Optuna: {execution_time:.2f} detik")
    print(f"Parameter Terbaik Hasil Optuna: {study.best_params}")
    
    # 4. Evaluasi Final Menggunakan Parameter Terbaik Optuna di Test Set (2023-2025)
    print("\nMelakukan evaluasi final pada Test Set...")
    best_params = study.best_params
    best_params['random_state'] = 42 # Pastikan random state tetap terkunci
    
    best_model = XGBRegressor(**best_params, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    test_preds = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    print(f"Hasil Akhir Test Set -> MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f}")
    
    # 5. Menyimpan Hasil Metrik & Model Usulan
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    metrics_results = {
        "method": "Optuna (Bayesian)",
        "execution_time_seconds": execution_time,
        "best_params": best_params,
        "best_validation_mse": float(study.best_value),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse)
    }
    
    with open(os.path.join("outputs", "metrics", "optuna_results.json"), "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    best_model.save_model(os.path.join("outputs", "models", "xgboost_optuna.json"))
    print("Model usulan dan metrik performa berhasil disimpan di folder outputs/")

if __name__ == "__main__":
    main()