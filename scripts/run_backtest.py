#!/usr/bin/env python3
# scripts/run_backtest.py

import sys
import argparse
import yaml
import joblib
import pandas as pd
from pathlib import Path

# Ensure src/ is on PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import FEATURE_DIR, MODEL_DIR
from preprocessing.filter_feature_data import filter_feature_data
from sim.strategies import STRATEGY_REGISTRY
from sim.simulate import BacktestSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest using a simulation config YAML"
    )
    parser.add_argument(
        "sim_config",
        type=Path,
        help="Path to the simulation config YAML file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional CSV path to save the trade log"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_backtest(sim_cfg: dict, output_file: Path = None):
    # Unpack sim config
    model_id      = sim_cfg["model_id"]
    tickers_file  = Path(sim_cfg["tickers_file"])
    buy_strategy = sim_cfg["buy_strategy"]
    buy_params  = sim_cfg.get("buy_params", {})
    sell_strategy = sim_cfg["sell_strategy"]
    sell_params  = sim_cfg.get("sell_params", {})
    budget        = sim_cfg.get("initial_budget", 1000.0)
    cooldown      = sim_cfg.get("cooldown_hours", sell_params.get("hold_hours", 3))
    feature_split = sim_cfg.get("feature_split", "test")

    # New: date-range fields
    start_date = sim_cfg.get("start_date")
    end_date   = sim_cfg.get("end_date")

    print(f"[INFO] Backtest window: {start_date} → {end_date}")

    # 1) Load tickers
    tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]
    print(f"[INFO] Loaded {len(tickers)} tickers from {tickers_file}")

    # 2) Load model
    model_path = MODEL_DIR / model_id / "model.pkl"
    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found at {model_path}")
    model = joblib.load(model_path)
    print(f"[INFO] Loaded model '{model_id}'")

    # 3) Load feature data for the specified split & date window
    df = filter_feature_data(
        feature_dir=FEATURE_DIR / feature_split,
        tickers=tickers,
        features=None,
        start_time=start_date,
        end_time=end_date,
        debug=False,
        retain_timestamp=True
    )
    if df.empty:
        sys.exit(f"[ERROR] No data in feature split '{feature_split}' "
                 f"for dates {start_date} → {end_date}")

    # 4) Prepare for prediction
    required_feats = list(model.feature_names_in_)
    df = df.dropna(subset=required_feats)
    X = df[required_feats]

    model_cfg = load_config(MODEL_DIR / model_id / "config.yaml")
    regression_model = model_cfg["regression_model"]
    # 5) Score
    if regression_model:
        df["score"] = model.predict(X)
    else:
        df["score"] = model.predict_proba(X)[:, 1]

    # 6) Strategy
    if buy_strategy not in STRATEGY_REGISTRY:
        sys.exit(f"[ERROR] Unknown strategy '{buy_strategy}'")
    if sell_strategy not in STRATEGY_REGISTRY:
        sys.exit(f"[ERROR] Unknown strategy '{sell_strategy}'")
    buy_fn = STRATEGY_REGISTRY[buy_strategy]
    sell_fn = STRATEGY_REGISTRY[sell_strategy]
    print(f"[INFO] Buy strategy: '{buy_strategy}' with params {buy_params}")
    print(f"[INFO] Sell strategy: '{sell_strategy}' with params {sell_params}")
    sim = BacktestSimulator(initial_budget=budget,
                        cooldown_hours=cooldown)
    trade_log, summary = sim.run(df, 
                                 buy_fn, buy_params,
                                 sell_fn, sell_params)

    # 7) Report
    print("\n[RESULT] Backtest Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 8) Save if requested
    if output_file:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        trade_log.to_csv(output_file, index=False)
        print(f"[INFO] Trade log saved to {output_file}")


def main():
    args = parse_args()
    sim_cfg = load_config(args.sim_config)
    run_backtest(sim_cfg, args.output)

if __name__ == "__main__":
    main()
