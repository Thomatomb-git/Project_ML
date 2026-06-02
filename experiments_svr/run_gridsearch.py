import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data_scaled

def main():
    print("=== Menjalankan Baseline: SVR + GridSearchCV ===")
    
    # 1. Memuat data yang sudah displit sekuensial
    X_train, y_train, X_val, y_val, X_test, y_test, scalery = load_and_split_data_scaled()
    
    # 2. Definisikan kandidat parameter murni untuk SVR
    param_grid = {
        'C': [1.0, 10.0, 100.0, 1000.0, 10000.0],
        'gamma': [0.001, 0.01, 0.1, 1.0, 10.0],
        'epsilon': [0.0001, 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }
    
    grid = ParameterGrid(param_grid)
    print(f"Total kombinasi parameter yang akan dievaluasi: {len(grid)}")
    
    best_val_mse = float('inf')
    best_params = None
    
    # 3. Proses Tuning Parameter (Manual Grid Search pada Validation Set)
    start_time = time.time()
    
    for i, params in enumerate(grid):
        # SVR tidak menerima argumen n_jobs
        model = SVR(**params, max_iter=50000)
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
    
    # 4. Evaluasi Final Menggunakan Parameter Terbaik di Test Set
    print("\nMelakukan evaluasi final pada Test Set...")
    best_model = SVR(**best_params)
    best_model.fit(X_train, y_train)
    
    test_preds = best_model.predict(X_test)
    test_preds = scalery.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    print(f"Hasil Akhir Test Set -> MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f}")
    
    # 5. Menyimpan Hasil Metrik & Model
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    # Menyimpan file log performa
    metrics_results = {
        "method": "SVR_GridSearchCV",
        "execution_time_seconds": execution_time,
        "best_params": best_params,
        "best_validation_mse": float(best_val_mse),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse)
    }
    
    with open(os.path.join("outputs", "metrics", "gridsearch_SVR_results.json"), "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    model_path = os.path.join("outputs", "models", "svr_gridsearch.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model dan metrik performa berhasil disimpan di folder outputs/ sebagai {model_path}")

if __name__ == "__main__":
    main()