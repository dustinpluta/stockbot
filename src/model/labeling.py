import pandas as pd

def label_increase_3h(df: pd.DataFrame) -> pd.Series:
    """
    Binary classification label:
    1 if the Close price increases 3 hours later, 0 otherwise.
    """
    if "Close" not in df.columns:
        raise ValueError("Close column required for labeling.")
    future_close = df["Close"].shift(-3)
    return (future_close > df["Close"]).astype(int)

LABELING_FUNCTIONS = {
    "increase_3h": label_increase_3h,
}

def get_label_function(method: str):
    if method not in LABELING_FUNCTIONS:
        raise ValueError(f"Unknown label method: {method}")
    return LABELING_FUNCTIONS[method]
