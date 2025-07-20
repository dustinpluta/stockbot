#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal # type: ignore
import sys

# Ensure src/ is on sys.path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import TRAIN_START, TRAIN_END  # Use dates from project config

INPUT_CSV = Path("data/dev/sp500_ticker_start_end.csv")
OUTPUT_FILE = Path("data/tickers/proto_universe.txt")

def load_ticker_dates(input_csv):
    df = pd.read_csv(input_csv, parse_dates=["start_date", "end_date"])
    df["end_date"].fillna(pd.Timestamp("2099-12-31"), inplace=True)
    return df

def filter_by_date_range(df, start_date, end_date):
    mask = (df["start_date"] <= end_date) & (df["end_date"] >= start_date)
    return df.loc[mask, "ticker"].unique()

def calculate_expected_bars(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_minutes = mcal.date_range(schedule, frequency='1D', closed='right')
    return trading_minutes

def check_data_coverage(tickers, start_date, end_date, min_coverage=0.9):
    accepted = []
    expected_bars = calculate_expected_bars(start_date, end_date)
    print(f"{expected_bars[0:9]}")
    total_expected = len(expected_bars)

    print(f"Total expected trading days: {total_expected}")

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
        available = df.dropna().shape[0]

        coverage = available / total_expected if total_expected else 0
        if coverage >= min_coverage:
            accepted.append(ticker)
        print(f"{ticker}: coverage={coverage:.1%} ({available}/{total_expected})")

    return accepted

def save_tickers(tickers, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(sorted(tickers)).to_csv(output_file, index=False, header=False)
    print(f"Saved validated tickers to {output_file}")

def main(min_coverage):
    tickers_df = load_ticker_dates(INPUT_CSV)
    tickers = filter_by_date_range(tickers_df, TRAIN_START, TRAIN_END)
    print(f"Tickers overlapping {TRAIN_START} to {TRAIN_END}: {len(tickers)} found")

    accepted_tickers = check_data_coverage(tickers, TRAIN_START, TRAIN_END, min_coverage=min_coverage)
    print(f"Tickers meeting coverage criteria: {len(accepted_tickers)}")

    save_tickers(accepted_tickers, OUTPUT_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage", type=float, default=0.9, help="Minimum fraction of hourly data required")
    args = parser.parse_args()

    main(args.min_coverage)
