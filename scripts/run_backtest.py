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
from sim.simulator import PortfolioSimulator


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


def load_sim_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_backtest(sim_cfg: dict, output_file: Path = None):
    # Unpack sim config
    model_id      = sim_cfg["model_id"]
    tickers_file  = Path(sim_cfg["tickers_file"])
    strategy_name = sim_cfg["strategy"]
    strat_params  = sim_cfg.get("strategy_params", {})
    budget        = sim_cfg.get("initial_budget", 1000.0)
    cooldown      = sim_cfg.get("cooldown_hours", strat_params.get("hold_hours", 3))
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

    # 5) Score
    df["predicted_prob"] = model.predict_proba(X)[:, 1]

    # 6) Strategy
    if strategy_name not in STRATEGY_REGISTRY:
        sys.exit(f"[ERROR] Unknown strategy '{strategy_name}'")
    strat_fn = STRATEGY_REGISTRY[strategy_name]
    print(f"[INFO] Applying strategy '{strategy_name}' with params {strat_params}")
    signals = strat_fn(df, **strat_params)

    # 7) Simulate
    sim = PortfolioSimulator(initial_budget=budget, cooldown_hours=cooldown)
    trade_log = sim.simulate(signals, price_col="Close")

    # 8) Report
    summary = sim.get_summary()
    print("\n[RESULT] Backtest Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 9) Save if requested
    if output_file:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        trade_log.to_csv(output_file, index=False)
        print(f"[INFO] Trade log saved to {output_file}")


def main():
    args = parse_args()
    sim_cfg = load_sim_config(args.sim_config)
    run_backtest(sim_cfg, args.output)


if __name__ == "__main__":
    main()
