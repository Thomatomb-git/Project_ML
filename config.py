import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_split_data():
    """
    Memuat data hasil feature engineering dan membaginya menjadi 
    Train (70%), Validation (15%), dan Test (15%) secara sekuensial (kronologis).
    """
    processed_file = os.path.join("data", "processed", "ihsg_processed_features.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"File {processed_file} tidak ditemukan. Jalankan src/features.py dulu!")
        
    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    df = df.sort_index() # Pastikan urutan waktu aman
    
    # Menentukan fitur (X) dan target (y)
    # Fitur sesuai paper 2: EMA_9, SMA_5, SMA_15, SMA_30, RSI, MACD, MACD_signal
    feature_cols = ['EMA_9', 'SMA_5', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']
    target_col = 'Target_Close'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Menghitung indeks batas pemotongan sekuensial
    total_rows = len(df)
    train_end = int(total_rows * 0.70)
    val_end = train_end + int(total_rows * 0.15)
    
    # Pemotongan data kronologis
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_and_split_data_scaled():
    """
    Memuat data hasil feature engineering dan membaginya menjadi 
    Train (70%), Validation (15%), dan Test (15%) secara sekuensial (kronologis).
    """
    processed_file = os.path.join("data", "processed", "ihsg_processed_features.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"File {processed_file} tidak ditemukan. Jalankan src/features.py dulu!")
        
    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    df = df.sort_index() # Pastikan urutan waktu aman
    
    # Menentukan fitur (X) dan target (y)
    # Fitur sesuai paper 2: EMA_9, SMA_5, SMA_15, SMA_30, RSI, MACD, MACD_signal
    feature_cols = ['EMA_9', 'SMA_5', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']
    target_col = 'Target_Close'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Menghitung indeks batas pemotongan sekuensial
    total_rows = len(df)
    train_end = int(total_rows * 0.70)
    val_end = train_end + int(total_rows * 0.15)
    
    # Pemotongan data kronologis
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Inisialisasi scaler terpisah
    scalerX = StandardScaler()
    scalery = StandardScaler()

    # Bungkus kembali numpy array hasil scaling ke dalam Pandas DataFrame / Series aslinya
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
    y_val_scaled = pd.Series(y_val_scaled, index=y_val.index)
    
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test, scalery