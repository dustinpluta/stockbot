# tests/test_feature_coverage_and_consistency.py

import sys
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from config import FEATURE_COLUMNS, FEATURE_DIR


def has_all_features(df: pd.DataFrame, ticker: str, feature_list: list[str]) -> bool:
    print(f"  [Check] Feature coverage for {ticker}")
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"    [FAIL] Missing features: {missing}")
        return False
    print(f"    [OK] All required features present")
    return True

def check_index_properties(df: pd.DataFrame, ticker: str) -> bool:
    print(f"  [Check] Index properties for {ticker}")
    idx = df.index
    success = True

    if not isinstance(idx, pd.DatetimeIndex):
        print(f"    [FAIL] Index is not a DatetimeIndex")
        success = False
    if idx.tz is not None:
        print(f"    [FAIL] Index is timezone-aware (expected naive UTC)")
        success = False
    if not idx.is_monotonic_increasing:
        print(f"    [FAIL] Index is not sorted")
        success = False
    if not idx.is_unique:
        print(f"    [FAIL] Index has duplicate timestamps")
        success = False

    if success:
        print(f"    [OK] Index is datetime-naive, sorted, and unique")

    return success

def drop_non_feature_columns(df: pd.DataFrame, ignore: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in ignore if c in df.columns], errors='ignore')

def run_consistency_check(baseline_df: pd.DataFrame, other_df: pd.DataFrame, ticker: str) -> bool:
    print(f"  [Check] Schema consistency for {ticker}")
    success = True

    if list(other_df.columns) != list(baseline_df.columns):
        print(f"    [FAIL] Column mismatch")
        print(f"      Expected: {list(baseline_df.columns)}")
        print(f"      Got     : {list(other_df.columns)}")
        success = False
    else:
        print(f"    [OK] Columns match")

    if not other_df.dtypes.equals(baseline_df.dtypes):
        print(f"    [FAIL] Dtype mismatch")
        print(other_df.dtypes)
        success = False
    else:
        print(f"    [OK] Dtypes match")

    if other_df.isnull().any().any():
        print(f"    [FAIL] Missing values found")
        success = False
    else:
        print(f"    [OK] No missing values")

    return success

def main():
    parquet_files = list(FEATURE_DIR.glob("*.parquet"))
    ignore_cols = ["ticker"]
    failed_tickers = []

    for base_file in parquet_files:
        base_ticker = base_file.stem
        print(f"\n=== Attempting baseline: {base_ticker} ===")

        base_df = pd.read_parquet(base_file)

        if not has_all_features(base_df, base_ticker, FEATURE_COLUMNS):
            print(f"Skipping {base_ticker} as baseline (feature check failed).")
            continue

        if not check_index_properties(base_df, base_ticker):
            print(f"Skipping {base_ticker} as baseline (index check failed).")
            continue

        base_df_clean = drop_non_feature_columns(base_df, ignore_cols)

        for file in parquet_files:
            ticker = file.stem
            df = pd.read_parquet(file)
            df_clean = drop_non_feature_columns(df, ignore_cols)

            print(f"\n--- Validating {ticker} ---")
            all_passed = True

            if not has_all_features(df, ticker, FEATURE_COLUMNS):
                all_passed = False
            if not check_index_properties(df, ticker):
                all_passed = False
            if not run_consistency_check(base_df_clean, df_clean, ticker):
                all_passed = False

            print(f"=== {ticker} check: {'[PASS]' if all_passed else '[FAIL]'} ===")

            if not all_passed:
                failed_tickers.append(ticker)

        break  # only use the first valid baseline

    print("\n=== Final Summary ===")
    if failed_tickers:
        print(f"[FAIL] The following tickers failed consistency checks: {', '.join(failed_tickers)}")
    else:
        print("[PASS] All feature files passed consistency and coverage checks.")

if __name__ == "__main__":
    main()
