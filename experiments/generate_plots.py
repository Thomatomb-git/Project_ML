import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBRegressor

# Menambahkan root direktori ke sys.path agar bisa membaca config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data

def main():
    print("=== Memulai Pembuatan Grafik Visualisasi Riset ===")
    
    # 1. Memuat data dan model Optuna (Metode Usulan)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    model_path = os.path.join("outputs", "models", "xgboost_optuna.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} tidak ditemukan. Jalankan run_optuna.py dulu!")
        
    model = XGBRegressor()
    model.load_model(model_path)
    
    # Membuat prediksi pada Test Set menggunakan model terbaik
    test_preds = model.predict(X_test)
    
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
    plt.plot(y_test.index, test_preds, label="XGBoost + Optuna Prediction", color='#ff7f0e', linestyle='--', linewidth=1.5)
    
    plt.title("IHSG Closing Price Prediction Results on Test Set (2023-2025)")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot1_path = os.path.join(plot_dir, "actual_vs_predicted.png")
    plt.savefig(plot1_path, dpi=300) # Resolusi 300 DPI standar jurnal ilmiah
    plt.close()
    print(f"Plot 1 berhasil disimpan di: {plot1_path}")

    # =========================================================================
    # PLOT 2: Feature Importance (Diagram Batang Tingkat Kepentingan Fitur)
    # =========================================================================
    print("Membuat Plot 2: Feature Importance...")
    
    # Mengambil skor pentingnya fitur (F-score/Gain) dari XGBoost
    importance_scores = model.feature_importances_
    feature_names = X_train.columns
    
    # Membuat DataFrame untuk memudahkan proses pengurutan grafik
    importance_df = pd.DataFrame({
        'Features': feature_names,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Features', data=importance_df, palette="viridis")
    
    plt.title("XGBoost Feature Importance Scores (Post-Bayesian Optimization)")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Technical Indicators")
    plt.tight_layout()
    
    plot2_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Plot 2 berhasil disimpan di: {plot2_path}")

    # =========================================================================
    # RINGKASAN DATA AKHIR UNTUK TABEL PAPER
    # =========================================================================
    print("\n=========================================================")
    print("BERIKUT ADALAH DATA UNTUK TABEL HASIL PENELITIAN PAPER ANDA:")
    print("=========================================================")
    
    with open(os.path.join("outputs", "metrics", "gridsearch_results.json"), "r") as f:
        gs = json.load(f)
    with open(os.path.join("outputs", "metrics", "optuna_results.json"), "r") as f:
        op = json.load(f)
        
    gs_time = f"{gs['execution_time_seconds']:.2f} detik"
    op_time = f"{op['execution_time_seconds']:.2f} detik"
    gs_val_mse = f"{gs['best_validation_mse']:.4f}"
    op_val_mse = f"{op['best_validation_mse']:.4f}"
    gs_test_mse = f"{gs['test_mse']:.4f}"
    op_test_mse = f"{op['test_mse']:.4f}"
    gs_test_rmse = f"{gs['test_rmse']:.4f}"
    op_test_rmse = f"{op['test_rmse']:.4f}"

    print(f"{'Metrik Performa':<25} | {'GridSearchCV (Baseline)':<25} | {'Optuna (Proposed)':<25}")
    print("-" * 83)
    print(f"{'Execution Time (Run)':<25} | {gs_time:<25} | {op_time:<25}")
    print(f"{'Validation MSE':<25} | {gs_val_mse:<25} | {op_val_mse:<25}")
    print(f"{'Test Set MSE':<25} | {gs_test_mse:<25} | {op_test_mse:<25}")
    print(f"{'Test Set RMSE':<25} | {gs_test_rmse:<25} | {op_test_rmse:<25}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()