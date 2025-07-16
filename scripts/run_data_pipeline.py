# scripts/run_data_pipeline.py
import sys
from pathlib import Path
import pandas as pd

import config
from preprocessing.data_fetch import fetch_stock_data
from preprocessing.process_features import process_features
from preprocessing.features import FEATURE_REGISTRY

def run_pipeline(ticker_file: Path, 
                 period: str = None,
                 interval: str = "1h",
                 feature_dir: Path = config.FEATURE_DIR,
                 feature_names: list[str] = FEATURE_REGISTRY.keys(),
                 debug: bool = False):
    with open(ticker_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"Processing features for {len(tickers)} tickers...")
    for ticker in tickers:
        df = fetch_stock_data(ticker)  # uses default 60d, 1h, no end date
        if df is not None and not df.empty:
            process_features(ticker, df, 
                             feature_dir, 
                             save=True,
                             debug=debug)
        else:
            print(f"[WARN] No data for {ticker}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_data_pipeline.py <ticker_file.txt> [time_period]")
        sys.exit(1)

    ticker_file_path = Path(sys.argv[1])
    if not ticker_file_path.exists():
        print(f"[ERROR] Ticker file not found: {ticker_file_path}")
        sys.exit(1)
    time_period = sys.argv[2] if len(sys.argv) == 3 else "60d"
    run_pipeline(ticker_file_path, time_period, debug=False)
