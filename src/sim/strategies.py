# src/sim/strategy.py
import pandas as pd

def basic_buy_strategy(df: pd.DataFrame, top_k: int = 3, prob_col: str = "predicted_prob") -> pd.DataFrame:
    """
    Generate a basic buy strategy signal: at each timestamp, buy top-k stocks by predicted probability.

    Parameters:
        df (pd.DataFrame): Must contain columns ['ticker', 'timestamp', 'predicted_prob']
        top_k (int): Number of stocks to "buy" per time step
        prob_col (str): Name of the column containing predicted probabilities

    Returns:
        pd.DataFrame: A DataFrame with the same index, including a new column 'action' with 'buy' or None.
    """
    if not {'ticker', 'Datetime', "predicted_prob"}.issubset(df.columns):
        raise ValueError("Required columns missing: 'ticker', 'timestamp', predicted probability column.")

    df_sorted = df.sort_values(by=['Datetime', 'predicted_prob'], ascending=[True, False]).copy()
    df_sorted['action'] = None

    # For each timestamp, mark top_k as 'buy'
    for ts, group in df_sorted.groupby('Datetime'):
        top_indices = group.head(top_k).index
        df_sorted.loc[top_indices, 'action'] = 'buy'

    return df_sorted
