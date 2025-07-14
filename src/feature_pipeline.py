# src/feature_pipeline.py

import pandas as pd
from pathlib import Path
from config import FEATURE_COLUMNS, FEATURE_DIR, LABEL_HORIZON_HOURS, DEBUG
from features import compute_features
from data_fetch import get_historical_data


def label_target(df: pd.DataFrame, horizon: int = LABEL_HORIZON_HOURS) -> pd.Series:
    """
    Label the target: 1 if close price increases within N hours, else 0.
    """
    return (df["Close"].shift(-horizon) > df["Close"]).astype(int)


def process_ticker(ticker: str, period: chr = "60d", debug: bool = DEBUG) -> pd.DataFrame:
    """
    Fetch data, compute features, and label target for a given ticker.

    Returns:
        DataFrame with features + target + 'ticker' column
    """
    try:
        df = get_historical_data(ticker, period=period, interval="1h")
        if debug:
            print(f"[INFO] Fetched {len(df)} rows for {ticker}")

        df = compute_features(df)

        df["target"] = label_target(df)
        df["ticker"] = ticker
        #df = df.dropna()

        if debug:
            print(f"[INFO] Final shape for {ticker}: {df.shape}")

        save_features(df, ticker)
        return df

    except Exception as e:
        print(f"[ERROR] Failed to process {ticker}: {e}")
        return pd.DataFrame()  # fail gracefully


def save_features(df: pd.DataFrame, ticker: str, feature_dir: Path = FEATURE_DIR):
    """
    Save feature DataFrame for a ticker to a .parquet file.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    out_path = feature_dir / f"{ticker}.parquet"
    df.to_parquet(out_path)
    print(f"[SUCCESS] Saved features for {ticker} to {out_path}")
