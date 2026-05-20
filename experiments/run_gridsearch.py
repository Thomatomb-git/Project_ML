import os
import sys
import time
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

# Menambahkan root direktori ke sys.path agar bisa membaca config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data

def main():
    print("=== Menjalankan Baseline: XGBoost + GridSearchCV ===")
    
    # 1. Memuat data yang sudah displit sekuensial
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    # 2. Definisikan kandidat parameter persis sesuai Table 1 dari Paper 2
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42] # Mengunci randomness
    }
    
    grid = ParameterGrid(param_grid)
    print(f"Total kombinasi parameter yang akan dievaluasi: {len(grid)}")
    
    best_val_mse = float('inf')
    best_params = None
    
    # 3. Proses Tuning Parameter (Manual Grid Search pada Validation Set)
    # Catatan: Kita mengevaluasi pada Validation Set khusus (bukan K-Fold silang acak) 
    # agar sesuai metodologi deret waktu murni paper 2.
    start_time = time.time()
    
    for i, params in enumerate(grid):
        model = XGBRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Prediksi pada validation set
        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        
        # Cari kombinasi dengan MSE terkecil di data validation
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params
            
        if (i + 1) % 10 == 0 or (i + 1) == len(grid):
            print(f"Progress: {i + 1}/{len(grid)} kombinasi selesai diperiksa...")
            
    execution_time = time.time() - start_time
    print("\n--- Proses Optimasi Selesai ---")
    print(f"Waktu Eksekusi Grid Search: {execution_time:.2f} detik")
    print(f"Parameter Terbaik Hasil Validasi: {best_params}")
    
    # 4. Evaluasi Final Menggunakan Parameter Terbaik di Test Set (2023-2025)
    print("\nMelakukan evaluasi final pada Test Set...")
    best_model = XGBRegressor(**best_params, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    test_preds = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    print(f"Hasil Akhir Test Set -> MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f}")
    
    # 5. Menyimpan Hasil Metrik & Model
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    # Menyimpan file log performa
    metrics_results = {
        "method": "GridSearchCV",
        "execution_time_seconds": execution_time,
        "best_params": best_params,
        "best_validation_mse": float(best_val_mse),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse)
    }
    
    with open(os.path.join("outputs", "metrics", "gridsearch_results.json"), "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    # Menyimpan file model terlatih
    best_model.save_model(os.path.join("outputs", "models", "xgboost_gridsearch.json"))
    print("Model dan metrik performa berhasil disimpan di folder outputs/")

if __name__ == "__main__":
    main()