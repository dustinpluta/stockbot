# scripts/summarize_feature_files.py

import sys
from pathlib import Path
import pandas as pd

# Ensure src/ is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import FEATURE_DIR


def summarize_feature_files(feature_dir: Path = FEATURE_DIR):
    """
    Print a summary of each .parquet feature file in the given directory.

    Args:
        feature_dir (Path): Path to the directory containing .parquet files.
    """
    files = sorted(feature_dir.glob("*.parquet"))

    if not files:
        print(f"[INFO] No .parquet files found in {feature_dir}")
        return

    print(f"Found {len(files)} feature file(s) in {feature_dir}\n")

    for file in files:
        ticker = file.stem.upper()
        try:
            df = pd.read_parquet(file)

            print(f"--- {ticker} ---")
            print(f"Rows: {len(df):,}")
            print(f"Columns: {len(df.columns)}")
            if isinstance(df.index, pd.DatetimeIndex):
                print(f"Date Range: {df.index.min()} → {df.index.max()}")

            missing_counts = df.isna().sum()
            missing_counts = missing_counts[missing_counts > 0]

            if not missing_counts.empty:
                print("Missing Values:")
                for col, count in missing_counts.items():
                    print(f"  {col}: {count}")
            else:
                print("Missing Values: None")

            feature_cols = sorted(set(df.columns) - {"target", "ticker"})
            print(f"Feature Columns ({len(feature_cols)}): {feature_cols}")
            print()

        except Exception as e:
            print(f"[ERROR] Failed to read {file.name}: {e}\n")


if __name__ == "__main__":
    summarize_feature_files()
