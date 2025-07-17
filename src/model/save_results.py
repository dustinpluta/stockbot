# src/model/save_results.py

import json
from pathlib import Path
import pandas as pd

def save_results(
    output_dir: Path,
    model_id: str,
    metrics: dict,
    y_eval: pd.Series,
    feature_names: list[str],
    data_type: str = "test"
) -> None:
    """
    Save a summary of model evaluation to a JSON file.

    The JSON will include:
      - model_id
      - data_type ("train", "test", "validation", etc.)
      - num_rows
      - label_distribution (counts of 0s and 1s)
      - num_features
      - feature_names
      - metrics (registered performance metrics)

    Prints a human-readable summary to stdout as well.
    """
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

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"[{model_id}] {data_type.capitalize()} Summary:")
    print(f"  Total rows: {summary['num_rows']}")
    print(f"  Label counts — 0s: {summary['label_distribution']['0']}, "
          f"1s: {summary['label_distribution']['1']}")
    print(f"  Features used ({summary['num_features']}): {', '.join(feature_names)}")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
