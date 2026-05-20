import os
import glob
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

def create_features_and_targets():
    """
    Membaca data mentah IHSG, menghitung indikator teknikal sebagai fitur,
    membuat target harga esok hari (A+1), dan menyimpan hasilnya ke data/processed/.
    """
    print("=== Memulai Proses Feature Engineering ===")
    
    # 1. Cari file mentah di data/raw/
    raw_path = os.path.join("data", "raw", "ihsg_raw_*.csv")
    files = glob.glob(raw_path)
    
    if not files:
        print("Error: File data mentah tidak ditemukan di data/raw/. Silakan jalankan download_data.py terlebih dahulu.")
        return None
        
    # Ambil file terbaru yang ditemukan
    latest_file = max(files, key=os.path.getctime)
    print(f"Membaca data mentah dari: {latest_file}")
    
    # Membaca data dengan menjadikan kolom 'Date' sebagai index
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True, skiprows=[1])
    
    # Pastikan data terurut secara kronologis dari tanggal terlama ke terbaru
    df = df.sort_index()

    # 2. Perhitungan Indikator Teknikal (Feature Factors)
    print("Menghitung indikator teknikal...")
    
    # Kolom Close digunakan sebagai basis perhitungan utama
    close_series = df['Close']
    
    # a. Exponential Moving Average (EMA 9)
    df['EMA_9'] = EMAIndicator(close=close_series, window=9).ema_indicator()
    
    # b. Simple Moving Average (SMA 5, SMA 15, SMA 30)
    df['SMA_5'] = SMAIndicator(close=close_series, window=5).sma_indicator()
    df['SMA_15'] = SMAIndicator(close=close_series, window=15).sma_indicator()
    df['SMA_30'] = SMAIndicator(close=close_series, window=30).sma_indicator()
    
    # c. Relative Strength Index (RSI 14)
    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
    
    # d. Moving Average Convergence Divergence (MACD & MACD Signal)
    macd_init = MACD(close=close_series, window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd_init.macd()
    df['MACD_signal'] = macd_init.macd_signal()

    # 3. Membuat Variabel Target (Harga Esok Hari / A+1)
    # Kita menggeser harga 'Close' ke atas sebesar 1 baris.
    # Artinya, di baris data hari Senin, kolom 'Target_Close' akan berisi harga asli hari Selasa.
    df['Target_Close'] = df['Close'].shift(-1)
    
    # 4. Pembersihan Data Akhir
    # Karena indikator memerlukan data historis beberapa hari ke belakang (misal SMA 30 butuh 30 hari awal),
    # baris-baris awal pasti akan bernilai NaN (kosong). Baris paling terakhir juga akan NaN untuk Target_Close.
    # Kita harus menghapusnya agar tidak membuat XGBoost error.
    df_clean = df.dropna()
    
    print(f"Jumlah baris awal: {len(df)} -> Jumlah baris bersih setelah transformasi: {len(df_clean)}")

    # 5. Simpan ke data/processed/
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    output_filename = "ihsg_processed_features.csv"
    output_path = os.path.join(processed_dir, output_filename)
    
    df_clean.to_csv(output_path)
    print(f"Data fitur berhasil disimpan di: {output_path}")
    print("=========================================\n")
    
    return output_path

if __name__ == "__main__":
    # Eksekusi fungsi utama jika file ini dijalankan langsung
    create_features_and_targets()

