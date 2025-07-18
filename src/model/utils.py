#!/usr/bin/env python3
# src/model/utils.py

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd

from model.metrics import METRIC_REGISTRY


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for train.py.
    """
    parser = argparse.ArgumentParser(
        description="Train a model from a YAML config file"
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to model config YAML (e.g. config_xgb.yaml)"
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load and return a YAML configuration as a dict.
    """
    with path.open("r") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Compute all metrics registered in METRIC_REGISTRY on (X, y).
    Returns a dict of metric_name -> value.
    """
    y_pred = model.predict(X)
    results: Dict[str, float] = {}
    for name, fn in METRIC_REGISTRY.items():
        try:
            results[name] = float(fn(y, y_pred))
        except Exception as e:
            results[name] = f"Error: {e}"
    return results
