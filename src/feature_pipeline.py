# src/feature_pipeline.py

import pandas as pd
from pathlib import Path

from data_fetch import get_historical_data
from features import add_technical_indicators
from config import FEATURE_DIR, LABEL_HORIZON_HOURS

def create_labeled_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds a binary label based on whether the close price increases over the next `horizon` hours.

    Args:
        df (pd.DataFrame): Feature-enhanced stock data
        horizon (int): Number of periods (e.g., hours) into the future to predict

    Returns:
        pd.DataFrame: DataFrame with 'target' column added
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)

    # Align to avoid index mismatch
    close_aligned, future_aligned = df["Close"].align(future_close, join="inner")
    df = df.loc[close_aligned.index]
    df["target"] = (future_aligned > close_aligned).astype(int)

    return df

def process_and_save_ticker(ticker: str, period: str = '60d', interval: str = '1h'):
    """
    Fetch data, compute features and target, and save to disk.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (default 60 days)
        interval (str): Interval (default 1 hour)
    """
    print(f"Processing {ticker}...")
    df = get_historical_data(ticker, period=period, interval=interval)
    df = add_technical_indicators(df)
    df = create_labeled_features(df, horizon=LABEL_HORIZON_HOURS)
    df["ticker"] = ticker  # Add ticker column for pooled training

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURE_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=True)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    process_and_save_ticker("AAPL")
