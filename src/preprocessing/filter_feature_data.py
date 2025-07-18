import pandas as pd
from pathlib import Path
from typing import Optional, Sequence


def filter_feature_data(
    feature_dir: Path,
    tickers: Sequence[str],
    features: Optional[Sequence[str]] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    save_path: Optional[Path] = None,
    debug: bool = False,
    retain_timestamp: bool = False
) -> pd.DataFrame:
    """
    Load and filter feature data from individual ticker Parquet files.

    Parameters:
        feature_dir: Path to folder containing individual .parquet files
        tickers: list of tickers to include
        features: list of features to retain (if None, keep all)
        start_time: optional datetime lower bound
        end_time: optional datetime upper bound
        save_path: optional path to save the concatenated filtered DataFrame
        debug: print summary of result
        retain_timestamp: if True, reset the datetime index into a column "timestamp"

    Returns:
        Concatenated and filtered DataFrame with a 'ticker' column.
        If retain_timestamp=True, includes a "timestamp" column.
    """
    all_dfs = []

    for ticker in tickers:
        path = feature_dir / f"{ticker}.parquet"
        if not path.exists():
            print(f"[WARNING] Missing file for {ticker}, skipping.")
            continue

        df = pd.read_parquet(path)
        df = df.drop(columns=[ticker], errors="ignore")

        # Time filter
        if start_time or end_time:
            df = df.loc[
                (df.index >= (start_time or df.index.min())) &
                (df.index <= (end_time or df.index.max()))
            ]

        # Feature filter
        if features:
            keep = [col for col in df.columns if col in features]
            df = df[keep]

        df = df.copy()
        df["ticker"] = ticker
        all_dfs.append(df)

    result = pd.concat(all_dfs).sort_index()

    # Optionally reset index to a timestamp column
    if retain_timestamp:
        result["timestamp"] = result.index
        result = result.reset_index(drop=True)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(save_path)
        print(f"[INFO] Saved filtered feature data to {save_path}")

    if debug:
        print(f"[SUMMARY] {len(result)} rows across {len(all_dfs)} tickers")
        print(f" - Time range: {result.index.min()} → {result.index.max()}")
        print(f" - Columns: {result.columns.tolist()}")

    return result
