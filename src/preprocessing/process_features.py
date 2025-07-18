# src/preprocessing/process_features.py

from pathlib import Path
import pandas as pd
from preprocessing.features import FEATURE_REGISTRY

def process_features(
    ticker: str,
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    start_time: str,
    end_time: str,
    split_name: str,
    feature_dir: Path,
    save: bool = True,
    debug: bool = False
) -> pd.DataFrame | None:
    """
    Compute and optionally save selected features for a given ticker over a specific time window.

    Args:
        ticker: Stock symbol
        raw_df: Full raw OHLCV DataFrame (indexed by datetime)
        feature_columns: List of feature names to compute (must be in FEATURE_REGISTRY)
        start_time: Inclusive start of window (ISO string or any parsable date)
        end_time:   Exclusive end of window
        split_name: Name of the data split (e.g. 'train', 'validate', 'test')
        feature_dir: Base directory where features are saved
        save: Whether to write out a parquet file
        debug: If True, print detailed logs

    Returns:
        DataFrame of computed features (index = timestamps in [start_time, end_time)),
        or None if processing failed or no data in window.
    """
    try:
        # 1) Slice the raw data for this split
        df = raw_df.copy()
        df.index = pd.to_datetime(df.index)
        start = pd.to_datetime(start_time)
        end   = pd.to_datetime(end_time)
        window_df = df[(df.index >= start) & (df.index < end)]
        
        if window_df.empty:
            if debug:
                print(f"[WARNING][{split_name}] No data for {ticker} in window {start_time} → {end_time}")
            return None

        # 2) Compute each feature with validation
        computed = []
        for feature in feature_columns:
            if debug:
                print(f"[{split_name}][{ticker}] Computing feature: {feature}")

            if feature not in FEATURE_REGISTRY:
                raise ValueError(f"Feature '{feature}' not registered.")

            func = FEATURE_REGISTRY[feature]
            result = func(window_df)

            # Validate return type
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"Feature function '{feature}' must return a pd.DataFrame.")

            # Validate row count
            if result.shape[0] != window_df.shape[0]:
                raise ValueError(
                    f"Feature '{feature}' returned {result.shape[0]} rows, expected {window_df.shape[0]}"
                )

            # Validate index alignment
            if not result.index.equals(window_df.index):
                raise ValueError(f"Feature '{feature}' index does not align with raw data index.")

            computed.append(result)

        # 3) Concatenate all features into one DataFrame
        features_df = pd.concat(computed, axis=1)

        # 4) Save to split-specific folder if requested
        if save:
            out_dir = feature_dir / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{ticker}.parquet"
            features_df.to_parquet(path)
            if debug:
                print(f"[{split_name}][{ticker}] Saved features to {path}")

        return features_df

    except Exception as e:
        print(f"[ERROR][{split_name}][{ticker}] {e}")
        return None
