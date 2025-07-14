# scripts/run_feature_pipeline.py

import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from feature_pipeline import process_ticker
from config import FEATURE_DIR

def read_tickers_from_file(filepath: str):
    with open(filepath, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/run_feature_pipeline.py <ticker_file.txt> <time_period>")
        sys.exit(1)

    ticker_file = sys.argv[1]
    period = sys.argv[2]

    if not Path(ticker_file).exists():
        print(f"Error: File '{ticker_file}' not found.")
        sys.exit(1)

    tickers = read_tickers_from_file(ticker_file)

    for ticker in tickers:
        try:
            process_ticker(ticker, period)
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

if __name__ == "__main__":
    main()
