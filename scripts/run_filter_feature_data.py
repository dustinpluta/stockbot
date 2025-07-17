# scripts/run_filter_feature_data.py

import sys
from pathlib import Path
import pandas as pd

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.filter_feature_data import filter_feature_data
from preprocessing.features import FEATURE_REGISTRY


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/run_filter_feature_data.py <tickers.txt> <output.parquet> <feature_file.txt> [start_date] [end_date]")
        sys.exit(1)

    ticker_file = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    start_date = pd.Timestamp(sys.argv[4]) if len(sys.argv) > 4 else None
    end_date = pd.Timestamp(sys.argv[5]) if len(sys.argv) > 5 else None

    # Read tickers
    with open(ticker_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Customize as needed
    feature_dir = Path("data/features")
    selected_features_file = Path(sys.argv[3])
    with open(selected_features_file, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]

    filtered_df = filter_feature_data(
        feature_dir=feature_dir,
        tickers=tickers,
        features=selected_features,
        start_time=start_date,
        end_time=end_date,
        save_path=output_path,
        debug=True
    )

if __name__ == "__main__":
    main()
