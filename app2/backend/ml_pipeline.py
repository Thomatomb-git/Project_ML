import yfinance as yf
import pandas as pd
import ta
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

def download_yfinance_data(ticker="^JKSE", period="90d"):
    """
    Downloads historical stock data from Yahoo Finance.
    We need enough history (e.g. 90 days) to calculate rolling indicators like SMA_30.
    """
    df = yf.download(ticker, period=period)
    
    # If the returned dataframe has multi-level columns (common in newer yfinance versions for multiple tickers), flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index = pd.to_datetime(df.index)
    
    # Keep only required columns, standard yfinance columns
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_to_keep]
    
    return df

def pipeline_1_preprocessing(df: pd.DataFrame):
    """
    Pipeline 1 for SVR and XGBoost Grid/Optuna.
    Target features: Close, High, Low, Open, Volume, EMA_9, SMA_5, SMA_15, SMA_30, RSI, MACD, MACD_signal
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_p1 = df.copy()
    
    close_series = df_p1['Close']
    
    df_p1['EMA_9'] = EMAIndicator(close=close_series, window=9).ema_indicator()
    df_p1['SMA_5'] = SMAIndicator(close=close_series, window=5).sma_indicator()
    df_p1['SMA_15'] = SMAIndicator(close=close_series, window=15).sma_indicator()
    df_p1['SMA_30'] = SMAIndicator(close=close_series, window=30).sma_indicator()
    df_p1['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
    
    macd_init = MACD(close=close_series, window_fast=12, window_slow=26, window_sign=9)
    df_p1['MACD'] = macd_init.macd()
    df_p1['MACD_signal'] = macd_init.macd_signal()
    
    # Drop NaNs created by rolling windows
    df_clean = df_p1.dropna()
    
    # The order of features MUST match the training data (ihsg_processed_1.csv)
    # Price,Close,High,Low,Open,Volume,EMA_9,SMA_5,SMA_15,SMA_30,RSI,MACD,MACD_signal
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'EMA_9', 'SMA_5', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']
    
    # Return the last row as the feature set for tomorrow's prediction
    return df_clean[feature_cols].iloc[[-1]]

def pipeline_2_preprocessing(df: pd.DataFrame):
    """
    Pipeline 2 for Custom XGBoost.
    Target features: Close, Return_Lag_1, Return_Lag_2, Return_Lag_3, Return_Lag_5, RSI_14, MACD_Hist, Normalized_ATR, Vol_to_SMA_20, Rolling_Mom_5, Day_of_Week, Is_Month_End
    """
    df_p2 = df.copy()
    
    # Lags
    df_p2['Return_Lag_1'] = df_p2['Close'].pct_change()
    df_p2['Return_Lag_2'] = df_p2['Return_Lag_1'].shift(1)
    df_p2['Return_Lag_3'] = df_p2['Return_Lag_1'].shift(2)
    df_p2['Return_Lag_5'] = df_p2['Return_Lag_1'].shift(4)
    
    # RSI
    df_p2['RSI_14'] = ta.momentum.rsi(df_p2['Close'], window=14)
    
    # MACD Hist
    df_p2['MACD_Hist'] = ta.trend.macd_diff(df_p2['Close'])
    
    # ATR
    atr_14 = ta.volatility.average_true_range(df_p2['High'], df_p2['Low'], df_p2['Close'], window=14)
    df_p2['Normalized_ATR'] = atr_14 / df_p2['Close'] 
    
    # Volatility
    sma_vol_20 = ta.trend.sma_indicator(df_p2['Volume'], window=20)
    df_p2['Vol_to_SMA_20'] = df_p2['Volume'] / sma_vol_20
    
    # Momentum
    df_p2['Rolling_Mom_5'] = df_p2['Return_Lag_1'].rolling(window=5).sum()
    
    # Calendar features
    df_p2['Day_of_Week'] = df_p2.index.dayofweek
    df_p2['Is_Month_End'] = df_p2.index.is_month_end.astype(int)
    
    df_clean = df_p2.dropna()
    
    feature_cols = [
        'Close', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5', 
        'RSI_14', 'MACD_Hist', 'Normalized_ATR', 'Vol_to_SMA_20', 'Rolling_Mom_5', 
        'Day_of_Week', 'Is_Month_End'
    ]
    
    # Return the last row as the feature set for tomorrow's prediction
    return df_clean[feature_cols].iloc[[-1]]

if __name__ == "__main__":
    # Test script locally
    data = download_yfinance_data()
    print("Pipeline 1 Features:\n", pipeline_1_preprocessing(data).T)
    print("\nPipeline 2 Features:\n", pipeline_2_preprocessing(data).T)
