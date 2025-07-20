# src/model/labeling.py

import pandas as pd
from typing import Callable, Dict

LABEL_REGISTRY: Dict[str, Callable] = {}

def register_label(name: str):
    """Decorator to register a labeling function."""
    def decorator(fn: Callable):
        LABEL_REGISTRY[name] = fn
        return fn
    return decorator

def get_label_function(name: str) -> Callable:
    """Fetch a registered labeling function by name."""
    try:
        return LABEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Label method '{name}' is not registered.")

@register_label("binary_return_3h")
def binary_return_3h(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """
    Binary label: 1 if Close_t+horizon > Close_t, else 0.
    """
    return (df["Close"].shift(-horizon) > df["Close"]).astype(int)

@register_label("return_3h")
def return_3h(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """
    Continuous label: (Close_t+horizon / Close_t) - 1
    """
    return (df["Close"].shift(-horizon) / df["Close"]) - 1
