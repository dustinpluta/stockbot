from pathlib import Path
import pandas as pd
from preprocessing.features import FEATURE_REGISTRY


def process_features(
    ticker: str,
    df: pd.DataFrame,
    feature_dir: Path,
    save: bool = True,
    debug: bool = False
) -> pd.DataFrame | None:
    """
    Compute and optionally save selected features for a given ticker.

    Returns the resulting DataFrame if successful, else None.
    """
    try:
        computed = []
        for feature in FEATURE_REGISTRY.keys():
            if debug:
                print(f"[INFO] Computing: {feature}")
            if feature not in FEATURE_REGISTRY:
                raise ValueError(f"Feature '{feature}' not found in registry.")
            func = FEATURE_REGISTRY[feature]
            result = func(df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"[ERROR] Feature {feature} did not return a DataFrame.")
            if result.shape[0] != df.shape[0]:
                raise ValueError(
                    f"[ERROR] Feature {feature} returned {result.shape[0]} rows; expected {df.shape[0]}"
                )
            computed.append(result)
        
        features = pd.concat(computed, axis=1)
        if save:
            feature_dir.mkdir(parents=True, exist_ok=True)
            features.to_parquet(feature_dir / f"{ticker}.parquet")
            print(f"[SUCCESS] Saved features for {ticker}")

        return features

    except Exception as e:
        print(f"[ERROR] Failed to process {ticker}: {e}")
        return None
