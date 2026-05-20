import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Menambahkan root direktori ke sys.path agar bisa membaca config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import load_and_split_data

def main():
    print("=" * 70)
    print("  CEK PERSENTASE ERROR MODEL PREDIKSI HARGA IHSG")
    print("=" * 70)
    
    # 1. Memuat data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    
    # 2. Memuat model-model yang sudah dilatih
    models = {}
    model_files = {
        "XGBoost + Optuna": os.path.join("outputs", "models", "xgboost_optuna.json"),
        "XGBoost + GridSearch": os.path.join("outputs", "models", "xgboost_gridsearch.json"),
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            model = XGBRegressor()
            model.load_model(path)
            models[name] = model
            print(f"✓ Model '{name}' berhasil dimuat dari {path}")
        else:
            print(f"✗ Model '{name}' tidak ditemukan di {path}, diskip.")
    
    if not models:
        print("Error: Tidak ada model yang ditemukan! Jalankan run_optuna.py / run_gridsearch.py dulu.")
        return
    
    # 3. Hitung metrik persentase error untuk setiap model
    for model_name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"  HASIL ANALISIS: {model_name}")
        print(f"{'=' * 70}")
        
        # Prediksi
        test_preds = model.predict(X_test)
        
        # === MENGHITUNG PERCENTAGE ERROR ===
        # Persentase error per data point: |(prediksi - aktual) / aktual| * 100
        actual = y_test.values
        percentage_errors = np.abs((test_preds - actual) / actual) * 100
        
        # MAPE (Mean Absolute Percentage Error) = rata-rata dari semua % error
        mape = np.mean(percentage_errors)
        
        # Metrik tambahan yang berguna
        median_pe = np.median(percentage_errors)
        max_pe = np.max(percentage_errors)
        min_pe = np.min(percentage_errors)
        std_pe = np.std(percentage_errors)
        
        # Berapa persen data yang errornya < 1%, < 2%, < 5%
        within_1pct = np.mean(percentage_errors < 1) * 100
        within_2pct = np.mean(percentage_errors < 2) * 100
        within_5pct = np.mean(percentage_errors < 5) * 100
        
        print(f"\n📊 RINGKASAN PERSENTASE ERROR:")
        print(f"   MAPE (Rata-rata % Error)  : {mape:.4f}%")
        print(f"   Median % Error            : {median_pe:.4f}%")
        print(f"   Error Terkecil            : {min_pe:.4f}%")
        print(f"   Error Terbesar            : {max_pe:.4f}%")
        print(f"   Standar Deviasi % Error   : {std_pe:.4f}%")
        
        print(f"\n📈 DISTRIBUSI AKURASI:")
        print(f"   Data dengan error < 1%    : {within_1pct:.1f}% dari total")
        print(f"   Data dengan error < 2%    : {within_2pct:.1f}% dari total")
        print(f"   Data dengan error < 5%    : {within_5pct:.1f}% dari total")
        
        print(f"\n💡 INTERPRETASI:")
        print(f"   Model rata-rata meleset {mape:.2f}% dari harga aslinya.")
        if mape < 1:
            print(f"   → Akurasi SANGAT BAIK! Error di bawah 1%.")
        elif mape < 2:
            print(f"   → Akurasi BAIK. Error di bawah 2%.")
        elif mape < 5:
            print(f"   → Akurasi CUKUP. Error di bawah 5%.")
        else:
            print(f"   → Akurasi KURANG. Error di atas 5%, perlu perbaikan model.")
        
        # === MENAMPILKAN BEBERAPA CONTOH PREDIKSI ===
        print(f"\n📋 CONTOH 10 PREDIKSI PERTAMA (Test Set):")
        print(f"   {'Tanggal':<14} | {'Harga Asli':>12} | {'Prediksi':>12} | {'Error (%)':>10}")
        print(f"   {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        
        for i in range(min(10, len(y_test))):
            date_str = str(y_test.index[i].date())
            print(f"   {date_str:<14} | {actual[i]:>12,.2f} | {test_preds[i]:>12,.2f} | {percentage_errors[i]:>9.4f}%")
        
        # === MEMBUAT VISUALISASI ===
        plot_dir = os.path.join("outputs", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        safe_name = model_name.replace(" ", "_").replace("+", "").lower()
        
        # Plot 1: Grafik persentase error sepanjang waktu
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Subplot atas: Persentase error per hari
        axes[0].plot(y_test.index, percentage_errors, color='#e74c3c', alpha=0.7, linewidth=0.8)
        axes[0].axhline(y=mape, color='#2ecc71', linestyle='--', linewidth=2, label=f'MAPE = {mape:.2f}%')
        axes[0].axhline(y=1.0, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.5, label='Batas 1%')
        axes[0].fill_between(y_test.index, 0, percentage_errors, alpha=0.15, color='#e74c3c')
        axes[0].set_title(f'Persentase Error Harian - {model_name}')
        axes[0].set_xlabel('Tanggal')
        axes[0].set_ylabel('Absolute Percentage Error (%)')
        axes[0].legend(loc='upper right')
        
        # Subplot bawah: Histogram distribusi error
        axes[1].hist(percentage_errors, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
        axes[1].axvline(x=mape, color='#e74c3c', linestyle='--', linewidth=2, label=f'MAPE = {mape:.2f}%')
        axes[1].axvline(x=median_pe, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median = {median_pe:.2f}%')
        axes[1].set_title(f'Distribusi Persentase Error - {model_name}')
        axes[1].set_xlabel('Absolute Percentage Error (%)')
        axes[1].set_ylabel('Frekuensi (Jumlah Hari)')
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"percentage_error_{safe_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n📊 Grafik persentase error disimpan di: {plot_path}")

    print(f"\n{'=' * 70}")
    print("  SELESAI! Cek folder outputs/plots/ untuk visualisasinya.")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
