# src/utils.py

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from config import FEATURE_COLUMNS, FEATURE_DIR

# src/utils.py

def concatenate_feature_files(feature_dir: Path = FEATURE_DIR, debug: bool = True) -> pd.DataFrame:
    """
    Loads and concatenates all .parquet feature files from a directory into a single DataFrame.

    Args:
        feature_dir: Path to the directory containing feature .parquet files.
        debug: If True, print per-ticker and summary diagnostics.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all features.
    """
    all_dfs = []
    failed = []

    for file in sorted(feature_dir.glob("*.parquet")):
        ticker = file.stem.upper()
        try:
            df = pd.read_parquet(file)
            n_rows = len(df)

            if debug:
                print(f"--- {ticker} ---")
                print(f"Rows: {n_rows}")
                n_missing = df.isna().sum().sum()
                if n_missing > 0:
                    missing_cols = df.columns[df.isna().any()].tolist()
                    print(f"[WARNING] Missing values: {n_missing} in columns {missing_cols}")
                else:
                    print("[OK] No missing values")

            all_dfs.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to load {file.name}: {e}")
            failed.append(file.name)

    if not all_dfs:
        raise RuntimeError("No feature files could be loaded.")

    combined = pd.concat(all_dfs).sort_index()

    if debug:
        print("\n=== Overall Summary ===")
        print(f"Total rows: {len(combined)}")
        print(f"Tickers: {combined['ticker'].nunique()}")
        print(f"Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"Feature columns: {sorted(set(combined.columns) - {'target', 'ticker'})}")

    return combined

def train_test_split_from_features(
    df: pd.DataFrame,
    split_time: pd.Timestamp,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a pooled feature DataFrame into train/test sets based on time.

    Args:
        df (pd.DataFrame): Output from concatenate_feature_files()
        split_time (pd.Timestamp): Cutoff for splitting (naive UTC assumed)
        debug (bool): If True, print info about the splits

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError("Expected timezone-naive datetime index")

    split_time = pd.to_datetime(split_time)

    train_df = df[df.index < split_time]
    test_df = df[df.index >= split_time]

    if debug:
        print(f"\n[DEBUG] Train/Test Split at {split_time}")
        print(f"  Train set: {len(train_df)} rows")
        print(f"    Date range: {train_df.index.min()} to {train_df.index.max()}")
        print(f"    Tickers: {sorted(train_df['ticker'].unique())}")
        print(f"  Test set:  {len(test_df)} rows")
        print(f"    Date range: {test_df.index.min()} to {test_df.index.max()}")
        print(f"    Tickers: {sorted(test_df['ticker'].unique())}")

    return train_df, test_df

def prepare_train_test_data(
    feature_dir: Path = FEATURE_DIR,
    split_time: pd.Timestamp = pd.Timestamp("2025-06-01"),
    tickers: Optional[list[str]] = None,
    dropna: bool = True,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads features, splits into train/test sets, and returns X/y for each.

    Args:
        feature_dir (Path): Directory of .parquet feature files
        split_time (pd.Timestamp): Timestamp to split on (naive UTC)
        tickers (list[str], optional): Tickers to include
        dropna (bool): Drop missing values before splitting
        debug (bool): Print summaries for each step

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    df = concatenate_feature_files(
        feature_dir=feature_dir,
        debug=debug
    )

    train_df, test_df = train_test_split_from_features(df, split_time=split_time, debug=debug)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test
