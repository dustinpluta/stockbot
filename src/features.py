# src/features.py

import pandas as pd
import ta
from config import FEATURE_COLUMNS

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame based on the configured feature list.

    Args:
        df (pd.DataFrame): Stock price data (with Open, High, Low, Close, Volume)

    Returns:
        pd.DataFrame: DataFrame with selected feature columns added
    """
    df = df.copy()
    close = df["Close"].astype(float).squeeze()

    if "rsi" in FEATURE_COLUMNS:
        df["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()

    if "macd" in FEATURE_COLUMNS:
        macd = ta.trend.MACD(close=close)
        df["macd"] = macd.macd()

    if "sma20" in FEATURE_COLUMNS:
        df["sma20"] = close.rolling(window=20).mean()

    # Drop rows with missing values introduced by indicators
    df.dropna(inplace=True)

    return df
