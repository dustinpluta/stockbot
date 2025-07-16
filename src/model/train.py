import sys
import yaml
import pandas as pd
from pathlib import Path
import joblib
import json

from config import FEATURE_SETS
from utils.filter_feature_data import filter_feature_data
from model.labeling import get_label_function
from model.split import split_data
import xgboost as xgb


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_from_config(config: dict):
    model_id = config["model_id"]
    output_dir = Path("models") / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tickers
    with open(config["tickers_file"], "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Select features
    feature_list = FEATURE_SETS[config["feature_set"]]

    # Load full feature data (unlabeled)
    df = filter_feature_data(
        feature_dir=Path("data/features"),
        tickers=tickers,
        features=feature_list + ["Close"],  # Include Close for labeling
        start_time=config["train_start"],
        end_time=config["test_end"],
    ).dropna()

    # Split before labeling
    train_df, test_df = split_data(df, config["train_start"], config["train_end"], config["test_start"])

    # Apply labeling separately to avoid leakage
    label_func = get_label_function(config["label_method"])
    train_df["target"] = label_func(train_df)
    test_df["target"] = label_func(test_df)

    # Drop rows with missing labels (due to shift(-3))
    train_df = train_df.dropna(subset=["target"])
    test_df = test_df.dropna(subset=["target"])

    # Select model features
    model_features = [f for f in feature_list if f in df.columns]
    train_df = train_df[model_features + ["target"]]
    test_df = test_df[model_features + ["target"]]

    # Extract X/y
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    print(f"[INFO] Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # Train model
    model = xgb.XGBClassifier(**config["xgboost_params"])
    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, output_dir / "model.pkl")
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Save simple metrics
    accuracy = float((model.predict(X_test) == y_test).mean())
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    print(f"[{model_id}] Accuracy: {accuracy:.4f} — model saved to {output_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/model/train.py <config_file.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    train_from_config(config)


if __name__ == "__main__":
    main()
