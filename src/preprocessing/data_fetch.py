# src/preprocessing/data_fetch.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

def fetch_stock_data(ticker: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False ,
            auto_adjust=True
        )
        if df.empty:
            return None
        df.index = df.index.tz_convert("UTC").tz_localize(None)  # force UTC then make naive
        df = df.xs(ticker, axis=1, level=1)
        df.columns.name = None
        return df
    except Exception as e:
        print(f"[ERROR] Could not fetch data for {ticker}: {e}")
        return None
