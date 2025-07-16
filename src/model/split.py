import pandas as pd

def split_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str
):
    """
    Split a feature DataFrame into raw train and test subsets (no labels).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)

    train_df = df[(df.index >= train_start) & (df.index < train_end)]
    test_df = df[df.index >= test_start]

    return train_df, test_df
