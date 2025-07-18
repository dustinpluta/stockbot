#!/usr/bin/env python3
# src/model/train.py

import logging
import sys
from pathlib import Path
import joblib
import yaml

from config import FEATURE_SETS, FEATURE_DIR, MODEL_DIR
from preprocessing.filter_feature_data import filter_feature_data
from model.labeling import get_label_function
from model.save_results import save_results
from model.registry import MODEL_REGISTRY
from model.utils import parse_args, load_config, evaluate_model

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_from_config(config: dict) -> None:
    """
    Core training logic:
      1. Load feature DataFrames for train/validate/test splits
      2. Label each split
      3. Invoke the registered trainer for model_type
      4. Persist model, config, and evaluation metrics
    """
    model_id = config["model_id"]
    output_dir = MODEL_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting training for model '%s'", model_id)

    # Load ticker list
    tickers_file = Path(config["tickers_file"])
    tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]
    logger.info("Loaded %d tickers from %s", len(tickers), tickers_file)

    # Determine feature list
    feature_list = FEATURE_SETS[config["feature_set"]]
    logger.info("Feature set '%s' → %d features", config["feature_set"], len(feature_list))

    # Load each split's data
    dfs = {}
    for split in ("train", "validate"):
        df = filter_feature_data(
            feature_dir=FEATURE_DIR / split,
            tickers=tickers,
            features=feature_list + ["Close"],
            start_time=None,
            end_time=None,
            debug=False
        )
        if df.empty:
            logger.error("No data for split '%s'", split)
            sys.exit(1)
        dfs[split] = df
        logger.info("Loaded %d rows for split '%s'", len(df), split)

    # Label and clean each split
    label_fn = get_label_function(config["label_method"])
    for split, df in dfs.items():
        df["target"] = label_fn(df)
        df.dropna(subset=["target"] + feature_list, inplace=True)
        dfs[split] = df
        logger.info("After labeling, '%s' has %d rows", split, len(df))

    # Prepare train/validate/test arrays
    X_train, y_train = dfs["train"][feature_list], dfs["train"]["target"]
    X_val,   y_val   = dfs["validate"][feature_list], dfs["validate"]["target"]
    #X_test,  y_test  = dfs["test"][feature_list], dfs["test"]["target"]

    # Train via registry
    model_type = config["model_type"].lower()
    trainer = MODEL_REGISTRY.get(model_type)
    if not trainer:
        logger.error("Unknown model_type '%s'. Valid: %s", model_type, list(MODEL_REGISTRY))
        sys.exit(1)

    logger.info("Training model '%s'...", model_type)
    model = trainer(X_train, y_train, X_val, y_val, config.get("model_params", {}))

    # Persist model and config
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config))
    logger.info("Model saved to %s", model_path)

    # Evaluate on test set
    metrics = evaluate_model(model, X_val, y_val)
    save_results(
        output_dir=output_dir,
        model_id=model_id,
        metrics=metrics,
        y_eval=y_val,
        feature_names=feature_list,
        data_type="Validate"
    )
    logger.info("Training complete for '%s'", model_id)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(args.config_file)
    train_from_config(config)


if __name__ == "__main__":
    main()
