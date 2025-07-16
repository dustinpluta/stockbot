# src/model/model_runner.py
import xgboost as xgb
import pandas as pd
from utils.filter_feature_data import filter_feature_data
from config import FEATURE_SETS
from pathlib import Path
import joblib
import json
import yaml # type: ignore

def run_model_training(config: dict):
    model_id = config["model_id"]
    output_dir = Path("models") / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tickers
    with open(config["tickers_file"], "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Load filtered data
    features = FEATURE_SETS[config["feature_set"]]
    df = filter_feature_data(
        feature_dir=Path("data/features"),
        tickers=tickers,
        features=features,
        start_time=config["train_start"],
        end_time=config["test_end"]
    )

    df = df.dropna()
    X = df[features]
    y = df["target"]

    # Train/test split
    split_date = pd.to_datetime(config["test_start"])
    X_train = X[df.index < split_date]
    y_train = y[df.index < split_date]
    X_test = X[df.index >= split_date]
    y_test = y[df.index >= split_date]

    # Train model
    model = xgb.XGBClassifier(**config["xgboost_params"])
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, output_dir / "model.pkl")

    # Save config and basic metrics
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    accuracy = float((model.predict(X_test) == y_test).mean())
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

    print(f"[{model_id}] Model trained and saved. Accuracy: {accuracy:.4f}")
