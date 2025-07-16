# src/preprocessing/features.py
import pandas as pd
import numpy as np

FEATURE_REGISTRY = {}

def register_feature(name):
    """
    Decorator to register a feature-generating function under a given name.
    """
    def decorator(func):
        if name in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' is already registered.")
        FEATURE_REGISTRY[name] = func
        return func
    return decorator

@register_feature("rsi")
def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI) for a given price DataFrame.
    
    Parameters:
        df: DataFrame containing a 'Close' column
        window: Rolling window size for RSI computation

    Returns:
        A DataFrame with one column: 'rsi'
    """
    if "Close" not in df.columns:
        raise ValueError("RSI calculation requires a 'Close' column in the input DataFrame.")

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.DataFrame({"rsi": rsi})

@register_feature("macd")
def macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return pd.DataFrame({"macd": macd_line})

@register_feature("sma20")
def sma20(df):
    sma = df["Close"].rolling(window=20).mean()
    return pd.DataFrame({"sma20": sma})

@register_feature("return_1h")
def return_1h(df):
    ret = df["Close"].pct_change(periods=1)
    return pd.DataFrame({"return_1h": ret})

@register_feature("return_4h")
def return_4h(df):
    ret = df["Close"].pct_change(periods=4)
    return pd.DataFrame({"return_4h": ret})

@register_feature("log_return_1h")
def log_return_1h(df):
    ratio = df["Close"] / df["Close"].shift(1)
    log_ret = np.log(ratio.where(ratio > 0))
    return pd.DataFrame({"log_return_1h": log_ret})

@register_feature("volatility_5h")
def volatility_5h(df):
    vol = df["Close"].pct_change().rolling(window=5).std()
    return pd.DataFrame({"volatility_5h": vol})

@register_feature("momentum")
def momentum(df):
    mom = df["Close"] - df["Close"].shift(5)
    return pd.DataFrame({"momentum": mom})

@register_feature("bollinger_upper")
def bollinger_upper(df, window: int = 20):
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    upper = sma + (2 * std)
    return pd.DataFrame({"bollinger_upper": upper})

@register_feature("bollinger_lower")
def bollinger_lower(df, window: int = 20):
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    lower = sma - (2 * std)
    return pd.DataFrame({"bollinger_lower": lower}) 

@register_feature("Close")
def close_(df):
    return pd.DataFrame(df["Close"], columns=["Close"])

@register_feature("Open")
def open_(df):
    return pd.DataFrame(df["Open"], columns=["Open"])

@register_feature("High")
def high_(df):
    return pd.DataFrame(df["High"], columns=["High"])

@register_feature("Low")
def low_(df):
    return pd.DataFrame(df["Low"], columns=["Low"])

@register_feature("Volume")
def volume_(df):
    return pd.DataFrame(df["Volume"], columns=["Volume"])

