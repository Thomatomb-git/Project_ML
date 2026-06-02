import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data_scaled

def main():
    print("=== Memulai Pembuatan Grafik Visualisasi Riset (SVR Mode) ===")
    
    # 1. Memuat data dan model Optuna (Metode Usulan)
    X_train, y_train, X_val, y_val, X_test, y_test, scalery = load_and_split_data_scaled()
    
    model_path = os.path.join("outputs", "models", "svr_optuna.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} tidak ditemukan. Jalankan run_optuna.py dulu!")
        
    # Memuat model SVR menggunakan joblib
    model = joblib.load(model_path)
    
    # Membuat prediksi pada Test Set menggunakan model terbaik
    test_preds = model.predict(X_test)
    test_preds = scalery.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    
    # Menyiapkan folder untuk menyimpan plot gambar
    plot_dir = os.path.join("outputs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Mengatur gaya visualisasi agar terlihat profesional untuk paper akademik
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

    # =========================================================================
    # PLOT 1: Actual vs Predicted Price (Grafik Garis Komparasi Kronologis)
    # =========================================================================
    print("Membuat Plot 1: Actual vs Predicted Price...")
    plt.figure(figsize=(14, 6))
    
    # Plot harga asli (Aktual)
    plt.plot(y_test.index, y_test.values, label="Actual IHSG Price (Truth)", color='#1f77b4', linewidth=1.5)
    # Plot harga tebakan model (Prediksi)
    plt.plot(y_test.index, test_preds, label="SVR + Optuna Prediction", color='#ff7f0e', linestyle='--', linewidth=1.5)
    
    plt.title("IHSG Closing Price Prediction Results on Test Set (2023-2025)")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot1_path = os.path.join(plot_dir, "actual_vs_predicted_SVR.png")
    plt.savefig(plot1_path, dpi=300) # Resolusi 300 DPI standar jurnal ilmiah
    plt.close()
    print(f"Plot 1 berhasil disimpan di: {plot1_path}")

    # =========================================================================
    # PLOT 2: Feature Importance (Menggunakan Permutation Importance untuk SVR RBF)
    # =========================================================================
    print("Membuat Plot 2: Feature Importance via Permutation Importance...")
    
    # Menghitung Permutation Importance pada Validation Set agar tidak bias
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Mengambil nilai mean dari penurunan performa (importances_mean)
    importance_scores = result.importances_mean
    feature_names = X_train.columns
    
    # Membuat DataFrame untuk memudahkan proses pengurutan grafik
    importance_df = pd.DataFrame({
        'Features': feature_names,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Features', data=importance_df, palette="viridis")
    
    plt.title("SVR Feature Importance Scores (via Permutation Importance)")
    plt.xlabel("Mean Importance Drop (MSE)")
    plt.ylabel("Technical Indicators")
    plt.tight_layout()
    
    plot2_path = os.path.join(plot_dir, "feature_importance_SVR.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Plot 2 berhasil disimpan di: {plot2_path}")

    # =========================================================================
    # RINGKASAN DATA AKHIR UNTUK TABEL PAPER
    # =========================================================================
    print("\n=========================================================")
    print("BERIKUT ADALAH DATA UNTUK TABEL HASIL PENELITIAN PAPER ANDA:")
    print("=========================================================")
        
    try:
        with open(os.path.join("outputs", "metrics", "gridsearch_SVR_results.json"), "r") as f:
            gs = json.load(f)
        with open(os.path.join("outputs", "metrics", "optuna_SVR_results.json"), "r") as f:
            op = json.load(f)
            
        gs_time = f"{gs['execution_time_seconds']:.2f} detik"
        op_time = f"{op['execution_time_seconds']:.2f} detik"
        gs_val_mse = f"{gs['best_validation_mse']:.4f}"
        op_val_mse = f"{op['best_validation_mse']:.4f}"
        gs_test_mse = f"{gs['test_mse']:.4f}"
        op_test_mse = f"{op['test_mse']:.4f}"
        gs_test_rmse = f"{gs['test_rmse']:.4f}"
        op_test_rmse = f"{op['test_rmse']:.4f}"
    except Exception:
        # Fallback teks jika file json belum terisi pasca pembersihan
        gs_time = op_time = gs_val_mse = op_val_mse = gs_test_mse = op_test_mse = gs_test_rmse = op_test_rmse = "Data Belum Ada"
        
    print(f"{'Metrik Performa':<25} | {'GridSearchCV (Baseline)':<25} | {'Optuna (Proposed)':<25}")
    print("-" * 83)
    print(f"{'Execution Time (Run)':<25} | {gs_time:<25} | {op_time:<25}")
    print(f"{'Validation MSE':<25} | {gs_val_mse:<25} | {op_val_mse:<25}")
    print(f"{'Test Set MSE':<25} | {gs_test_mse:<25} | {op_test_mse:<25}")
    print(f"{'Test Set RMSE':<25} | {gs_test_rmse:<25} | {op_test_rmse:<25}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()