# scripts/run_filter_feature_data.py

import sys
from pathlib import Path
import pandas as pd

from utils.filter_feature_data import filter_feature_data
from preprocessing.features import FEATURE_REGISTRY


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_filter_feature_data.py <tickers.txt> <output.parquet> [start_date] [end_date]")
        sys.exit(1)

    ticker_file = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    start_date = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else None
    end_date = pd.Timestamp(sys.argv[4]) if len(sys.argv) > 4 else None

    # Read tickers
    with open(ticker_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Customize as needed
    feature_dir = Path("data/features")
    features = ['rsi', 'macd', 'return_1h']

    filtered_df = filter_feature_data(
        feature_dir=feature_dir,
        tickers=tickers,
        features=features,
        start_time=start_date,
        end_time=end_date,
        save_path=output_path,
        debug=True
    )


if __name__ == "__main__":
    main()
