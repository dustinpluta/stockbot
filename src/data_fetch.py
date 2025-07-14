# src/data_fetch.py

import yfinance as yf
import pandas as pd
from datetime import datetime

# src/data_fetch.py

import yfinance as yf
import pandas as pd

def get_historical_data(ticker: str, period: str = '60d', interval: str = '1h') -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance, convert to UTC, and remove timezone.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period (e.g., '60d', '1y')
        interval (str): Data interval (e.g., '1h', '30m', '1d')

    Returns:
        pd.DataFrame: Clean OHLCV data indexed by naive UTC datetime
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Convert index to UTC if it's not already, then remove timezone info
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    df.index = df.index.tz_localize(None)  # remove tz info (naive UTC)
    
    return df

def get_latest_hourly_bar(ticker: str) -> pd.Series:
    """
    Fetch the most recent hourly bar for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        pd.Series: The most recent OHLCV data as a Pandas Series
    """
    df = get_historical_data(ticker, period='1d', interval='1h')
    if not df.empty:
        return df.iloc[-1]
    else:
        raise ValueError(f"No data returned for ticker {ticker}")

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    df = get_historical_data(ticker)
    print(df.tail())

    latest_bar = get_latest_hourly_bar(ticker)
    print("\nLatest hourly bar:")
    print(latest_bar)
