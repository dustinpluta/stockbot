# src/sim/strategies.py

import pandas as pd
from datetime import timedelta
from typing import Callable, Dict, Any, List

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

@register_strategy("cooldown_sell")
def cooldown_sell_strategy(
    hour_df: pd.DataFrame,
    holdings: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
    cooldown: timedelta
) -> List[str]:
    """
    Liquidate any position held for >= cooldown hours.
    Returns a list of tickers to sell.
    """
    ts = hour_df["timestamp"].iloc[0]
    return [
        ticker for ticker, info in holdings.items()
        if ts - info["buy_time"] >= cooldown
    ]

@register_strategy("first_hour_equal_allocation")
def first_hour_equal_allocation_strategy(
    df: pd.DataFrame,
    budget: float,
    params: Dict[str, Any],
    price_col: str = "Close",
    score: str = "score",
    target_hour: int = 13
) -> pd.DataFrame:
    """
    In the first trading hour only, buy the top_k tickers by score,
    allocating budget equally across them.
    Otherwise, no action.

    Args:
      df: hour‐slice DataFrame with columns ['timestamp','ticker',price_col,score_col]
      budget: current available cash
      top_k: how many names to buy
      price_col: price column name
      score_col: model score column
    Returns:
      DataFrame with added 'action' and 'quantity'
    """
    df = df.copy()
    df["action"]   = None
    df["quantity"] = 0

    # Detect the first hour of the day: on Yahoo data this is hour==10 (10:30 bar)
    if df["timestamp"].dt.hour.iloc[0] == target_hour:
        # pick top_k by score
        top = df.nlargest(params["top_k"], score)
        if not top.empty:
            alloc = min([1000 / params["top_k"], budget])
            for idx, row in top.iterrows():
                print(row)
                print(alloc)
                price = row[price_col]
                qty   = int(alloc // price)
                print(qty)
                if qty > 0:
                    df.at[idx, "action"]   = "buy"
                    df.at[idx, "quantity"] = qty

    return df
