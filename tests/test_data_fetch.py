# scripts/test_data_fetch.py

import sys
import os
from pathlib import Path

# Add src/ to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from data_fetch import get_historical_data, get_latest_hourly_bar

def main():
    ticker = "AAPL"
    
    print("=== Fetching Historical Data ===")
    df = get_historical_data(ticker, period='5d', interval='1h')
    print(df.tail())
    print(f"Returned {len(df)} rows.\n")

    print("=== Fetching Latest Hourly Bar ===")
    latest = get_latest_hourly_bar(ticker)
    print(latest)

if __name__ == "__main__":
    main()
