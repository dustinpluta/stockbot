# src/utils.py

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from config import FEATURE_COLUMNS, FEATURE_DIR

# src/utils.py

def concatenate_feature_files(
    feature_dir: Path = FEATURE_DIR,
    tickers: Optional[list[str]] = None,
    dropna: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Load and concatenate multiple .parquet feature files.

    Args:
        feature_dir (Path): Path to directory containing .parquet files
        tickers (list[str], optional): List of tickers to load (all if None)
        dropna (bool): Whether to drop rows with missing values
        debug (bool): If True, print summary stats per ticker

    Returns:
        pd.DataFrame: Combined feature data
    """
    dfs = []
    files = list(feature_dir.glob("*.parquet"))

    for file in files:
        ticker = file.stem
        if tickers and ticker not in tickers:
            continue

        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"[SKIP] Failed to load {ticker}: {e}")
            continue

        if debug:
            print(f"\n[DEBUG] {ticker} feature summary:")
            print(f"  Total rows:       {len(df)}")
            print(f"  Unique timestamps: {df.index.nunique()}")
            num_missing = df.isnull().sum().sum()
            print(f"  Total missing values: {num_missing}")
            if num_missing > 0:
                missing_by_col = df.isnull().sum()
                missing_cols = missing_by_col[missing_by_col > 0]
                print(f"  Features with missing values:")
                for col, count in missing_cols.items():
                    print(f"    {col}: {count}")

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid feature files found.")

    df_all = pd.concat(dfs)

    if dropna:
        df_all.dropna(inplace=True)

    # Standardize index to naive UTC datetime
    df_all.index = df_all.index.tz_localize(None)

    return df_all

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
    split_time: pd.Timestamp = pd.Timestamp("2024-12-01"),
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
        tickers=tickers,
        dropna=dropna,
        debug=debug
    )

    train_df, test_df = train_test_split_from_features(df, split_time=split_time, debug=debug)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test
