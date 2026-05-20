import os
import pandas as pd
import yfinance as yf

def download_ihsg_data(ticker="^JKSE", start_date="2010-01-01", end_date="2025-12-31"):
    """
    Mengunduh data historis harian dari Yahoo Finance dan menyimpannya ke folder data/raw/.
    """
    print(f"=== Memulai Pengunduhan Data {ticker} ===")
    print(f"Periode: {start_date} sampai {end_date}")
    
    try:
        # 1. Menarik data menggunakan yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Jaga-jaga jika data kosong karena kesalahan ticker atau koneksi
        if df.empty:
            print("Error: Data tidak ditemukan atau kosong. Periksa koneksi internet atau simbol ticker.")
            return None
            
        print(f"Berhasil mengunduh {len(df)} baris data.")
        
        # 2. Menyiapkan folder penyimpanan jika belum ada
        output_dir = os.path.join("data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. Menentukan nama file keluaran
        filename = f"ihsg_raw_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        file_path = os.path.join(output_dir, filename)
        
        # 4. Menyimpan ke format CSV
        df.to_csv(file_path)
        print(f"Data mentah berhasil disimpan di: {file_path}")
        print("=========================================\n")
        
        return file_path

    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh data: {str(e)}")
        return None

if __name__ == "__main__":
    # Bagian ini akan berjalan jika Anda mengeksekusi langsung file ini di terminal:
    # python src/download_data.py
    
    # Menjalankan fungsi dengan parameter default riset kita
    download_ihsg_data(
        ticker="^JKSE",
        start_date="2010-01-01",
        end_date="2025-12-31"
    )