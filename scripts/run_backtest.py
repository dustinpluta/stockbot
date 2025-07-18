#!/usr/bin/env python3
# scripts/run_backtest.py

import sys
import argparse
import joblib
import pandas as pd
from pathlib import Path

# Ensure src/ is on PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import FEATURE_DIR, MODEL_DIR
from preprocessing.filter_feature_data import filter_feature_data
from sim.strategies import basic_buy_strategy
from sim.simulator import PortfolioSimulator

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest a trained model on the test feature split"
    )
    p.add_argument(
        "tickers_file",
        type=Path,
        help="Text file with one ticker per line"
    )
    p.add_argument(
        "model_id",
        help="Model identifier (folder under models/) to load"
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional CSV path to save the trade log"
    )
    return p.parse_args()


def run_backtest(tickers_file: Path, model_id: str, output_file: Path = None):
    # 1) Load tickers
    tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]
    print(f"[INFO] Tick­ers loaded: {len(tickers)} from {tickers_file}")

    # 2) Load model
    model_path = MODEL_DIR / model_id / "model.pkl"
    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found at {model_path}")
    model = joblib.load(model_path)
    print(f"[INFO] Loaded model '{model_id}'")

    # 3) Load test‐split features
    test_feat_dir = FEATURE_DIR / "test"
    df = filter_feature_data(
        feature_dir=test_feat_dir,
        tickers=tickers,
        features=None,      # None = load all features
        start_time=None,
        end_time=None,
        debug=False,
        retain_timestamp=True
    )
    
    if df.empty:
        sys.exit("[ERROR] No test data loaded; check your FEATURE_DIR/test folder")

    # 4) Predict probabilities
    #    Assumes classifier with predict_proba and `feature_names_in_`
    X = df[model.feature_names_in_]
    df["predicted_prob"] = model.predict_proba(X)[:, 1]

    # 5) Generate buy signals
    signals = basic_buy_strategy(df)

    # 6) Run the portfolio simulator
    sim = PortfolioSimulator(initial_budget=1000)
    trade_log = sim.simulate(signals)

    # 7) Report summary & optionally save
    summary = sim.get_summary()
    print("\n[RESULT] Backtest Summary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if output_file:
        trade_log.to_csv(output_file, index=False)
        print(f"[INFO] Trade log saved to {output_file}")


def main():
    args = parse_args()
    run_backtest(args.tickers_file, args.model_id, args.output)


if __name__ == "__main__":
    main()
