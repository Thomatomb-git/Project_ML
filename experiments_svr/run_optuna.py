import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
import optuna
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data_scaled
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    print("=== Menjalankan Metode Usulan: SVR + Optuna (Bayesian Optimization) ===")
    
    # 1. Memuat data sekuensial yang konsisten
    X_train, y_train, X_val, y_val, X_test, y_test, scalery = load_and_split_data_scaled()
    
    # 2. Definisikan Fungsi Objektif untuk Optuna yang Sesuai SVR
    def objective(trial):
        # Mengonfigurasi ruang pencarian parameter murni untuk SVR
        params = {
            # Memaksa C mencari dari angka 1 sampai 10,000 agar model lebih sensitif terhadap error
            'C': trial.suggest_float('C', 1.0, 1e4, log=True),
            # Memperluas gamma agar radius support vector lebih fleksibel
            'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-4, 0.1, log=True),
            'kernel': 'rbf'
        }
        
        # Latih model SVR (Tanpa n_jobs karena tidak didukung SVR sklearn)
        model = SVR(**params, max_iter=50000)
        model.fit(X_train, y_train)
        
        # Hitung performa pada Validation Set
        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        
        return val_mse

    # 3. Proses Optuna Study (Mencari titik parameter terbaik)
    start_time = time.time()
    
    study = optuna.create_study(direction='minimize')
    
    n_trials = 50
    print(f"Memulai pencarian Bayesian sebanyak {n_trials} iterasi...")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
    
    execution_time = time.time() - start_time
    print("\n--- Proses Optimasi Cerdas Selesai ---")
    print(f"Waktu Eksekusi Optuna: {execution_time:.2f} detik")
    print(f"Parameter Terbaik Hasil Optuna: {study.best_params}")
    
    # 4. Evaluasi Final Menggunakan Parameter Terbaik Optuna di Test Set
    print("\nMelakukan evaluasi final pada Test Set...")
    best_params = study.best_params
    
    # Menginisialisasi SVR dengan parameter terbaik (tanpa n_jobs dan random_state)
    best_model = SVR(**best_params)
    best_model.fit(X_train, y_train)
    
    test_preds = best_model.predict(X_test)
    test_preds = scalery.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    print(f"Hasil Akhir Test Set -> MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f}")
    
    # 5. Menyimpan Hasil Metrik & Model Usulan
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    metrics_results = {
        "method": "SVR_Optuna_(Bayesian)",
        "execution_time_seconds": execution_time,
        "best_params": best_params,
        "best_validation_mse": float(study.best_value),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse)
    }
    
    with open(os.path.join("outputs", "metrics", "optuna_SVR_results.json"), "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    # Menyimpan model SVR menggunakan joblib karena SVR tidak punya .save_model()
    model_path = os.path.join("outputs", "models", "svr_optuna.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model usulan dan metrik performa berhasil disimpan di folder outputs/ sebagai {model_path}")

if __name__ == "__main__":
    main()