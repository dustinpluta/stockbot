# src/sim/strategies.py

import pandas as pd
from datetime import timedelta
from typing import Callable, Dict

# --- Strategy Registry ---

STRATEGY_REGISTRY: Dict[str, Callable] = {}

def register_strategy(name: str):
    """
    Decorator to register a strategy function under a given name.
    The strategy function must have signature: (df: pd.DataFrame, **kwargs) -> pd.DataFrame.
    """
    def decorator(fn: Callable):
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{name}' is already registered.")
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


# --- Strategy Implementations ---

@register_strategy("basic_buy")
def basic_buy_strategy(
    df: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Strategy #1: Buy top_k tickers by predicted probability each timestamp.
    End-of-day sells are handled by the simulator.
    """
    df = df.copy()
    df["action"] = None

    for ts, group in df.groupby("timestamp"):
        top = group.nlargest(top_k, "score")
        df.loc[top.index, "action"] = "buy"

    return df


@register_strategy("hold_n_hours")
def hold_n_hours_strategy(
    df: pd.DataFrame,
    hold_hours: int = 3,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Strategy #2: Buy top_k as in basic_buy, then sell each position
    exactly hold_hours later (if that timestamp exists).
    """
    df = basic_buy_strategy(df, top_k=top_k)
    df = df.copy()

    for idx, row in df[df["action"] == "buy"].iterrows():
        sell_time = row["timestamp"] + timedelta(hours=hold_hours)
        mask = (df["timestamp"] == sell_time) & (df["ticker"] == row["ticker"])
        df.loc[mask, "action"] = "sell"

    return df


@register_strategy("threshold_hold")
def threshold_time_exit_strategy(
    df: pd.DataFrame,
    threshold: float = 0.7,
    hold_hours: int = 3,
    score: str = "predicted_prob"
) -> pd.DataFrame:
    """
    Strategy #3: Buy when predicted probability exceeds threshold,
    then sell hold_hours later if that timestamp exists.
    """
    df = df.copy()
    df["action"] = None

    buys = df[df[score] > threshold]
    df.loc[buys.index, "action"] = "buy"

    for idx, row in buys.iterrows():
        sell_time = row["timestamp"] + timedelta(hours=hold_hours)
        mask = (df["timestamp"] == sell_time) & (df["ticker"] == row["ticker"])
        df.loc[mask, "action"] = "sell"

    return df
