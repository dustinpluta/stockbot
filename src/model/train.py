import sys
import yaml # type: ignore
import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from config import FEATURE_SETS
from utils.filter_feature_data import filter_feature_data
from model.labeling import get_label_function
from model.split import split_data
from model.metrics import METRIC_REGISTRY


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, X_eval, y_eval) -> dict:
    """Evaluate the trained model using the registered metrics."""
    y_pred = model.predict(X_eval)
    results = {}
    for name, func in METRIC_REGISTRY.items():
        try:
            results[name] = float(func(y_eval, y_pred))
        except Exception as e:
            results[name] = f"Error: {e}"
    return results


def save_model_results(
    output_dir: Path,
    model_id: str,
    metrics: dict,
    y_eval: pd.Series,
    feature_names: list,
    data_type: str = "test"
):
    """Save evaluation summary including metrics, label distribution, and features."""
    summary = {
        "model_id": model_id,
        "data_type": data_type,
        "num_rows": len(y_eval),
        "label_distribution": {
            "0": int((y_eval == 0).sum()),
            "1": int((y_eval == 1).sum())
        },
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{model_id}] {data_type.capitalize()} Summary:")
    print(f"  Label counts — 0s: {summary['label_distribution']['0']}, 1s: {summary['label_distribution']['1']}")
    print(f"  Features used ({summary['num_features']}): {', '.join(feature_names)}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
     
def train_from_config(config: dict):
    model_id = config["model_id"]
    output_dir = Path("models") / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tickers
    with open(config["tickers_file"], "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Select features
    feature_list = FEATURE_SETS[config["feature_set"]]

    # Load data
    df = filter_feature_data(
        feature_dir=Path("data/features"),
        tickers=tickers,
        features=feature_list + ["Close"],
        start_time=config["train_start"],
        end_time=config["test_end"],
    ).dropna()

    # Split before labeling
    train_df, test_df = split_data(df, config["train_start"], config["train_end"], config["test_start"])

    # Apply label
    label_func = get_label_function(config["label_method"])
    train_df["target"] = label_func(train_df)
    test_df["target"] = label_func(test_df)

    # Drop missing labels
    train_df = train_df.dropna(subset=["target"])
    test_df = test_df.dropna(subset=["target"])

    # Extract model features
    model_features = [f for f in feature_list if f in train_df.columns]
    X_train = train_df[model_features]
    y_train = train_df["target"]
    X_test = test_df[model_features]
    y_test = test_df["target"]

    print(f"[INFO] Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # Train
    model = XGBClassifier(**config["xgboost_params"])
    model.fit(X_train, y_train)

    # Save model + config
    joblib.dump(model, output_dir / "model.pkl")
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Evaluate and save results
    metrics = evaluate_model(model, X_test, y_test)
    save_model_results(output_dir, model_id, metrics, y_test, model_features, data_type="test")


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/model/train.py <config_file.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    train_from_config(config)


if __name__ == "__main__":
    main()
