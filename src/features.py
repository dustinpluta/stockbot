# src/features.py

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using an EMA-based method.

    Args:
        series (pd.Series): The price series (e.g., closing prices).
        window (int): The RSI calculation period.

    Returns:
        pd.Series: RSI values, bounded between 0 and 100.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_macd(series: pd.Series, span1: int = 12, span2: int = 26, signal: int = 9) -> pd.Series:
    ema_short = series.ewm(span=span1, adjust=False).mean()
    ema_long = series.ewm(span=span2, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic moving average
    df["sma20"] = df["Close"].rolling(window=20).mean()

    # RSI
    df["rsi"] = compute_rsi(df["Close"])

    # MACD
    df["macd"] = compute_macd(df["Close"])

    # Price returns
    df["return_1h"] = df["Close"].pct_change(1)
    df["return_4h"] = df["Close"].pct_change(4)
    df["log_return_1h"] = np.log(df["Close"] / df["Close"].shift(1))

    # Volatility
    df["volatility_5h"] = df["Close"].rolling(window=5).std()

    # Momentum: price above/below SMA
    df["momentum"] = df["Close"] - df["sma20"]

    # Bollinger Bands
    rolling_std = df["Close"].rolling(window=20).std()
    df["bollinger_upper"] = df["sma20"] + 2 * rolling_std
    df["bollinger_lower"] = df["sma20"] - 2 * rolling_std

    return df
