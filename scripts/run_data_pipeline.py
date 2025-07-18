# scripts/run_data_pipeline.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.data_fetch import fetch_stock_data
from preprocessing.process_features import process_features
from config import SPLIT_BOUNDS, FEATURE_DIR, FEATURE_SETS, PERIOD

def run_pipeline(tickers_file: Path, debug: bool=False):
    tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]

    for ticker in tickers:
        raw_df = fetch_stock_data(ticker, PERIOD)  # full-range OHLCV

        for split_name, (start, end) in SPLIT_BOUNDS.items():
            process_features(
                ticker=ticker,
                raw_df=raw_df,
                feature_columns=FEATURE_SETS["all"],
                start_time=start,
                end_time=end,
                split_name=split_name,
                feature_dir=FEATURE_DIR,
                save=True,
                debug=debug
            )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_data_pipeline.py <tickers.txt>")
        sys.exit(1)
    run_pipeline(Path(sys.argv[1]), debug=True)
