# src/model/registry.py

from typing import Callable, Dict
import pandas as pd

# Model classes
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """
    Decorator to register a model-specific training function.
    The function signature must be:
      fn(X_train, y_train, X_val, y_val, params) -> fitted_model
    """
    def decorator(fn: Callable):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


@register_model("xgboost")
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict
) -> XGBClassifier:
    model = XGBClassifier(**params)
    esr = params.get("early_stopping_rounds")
    if esr:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=esr,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    return model


@register_model("random_forest")
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict
) -> RandomForestClassifier:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


@register_model("logistic_regression")
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict
) -> LogisticRegression:
    # Drop any NaNs in features or target
    df = pd.concat([X_train, y_train.rename("target")], axis=1).dropna()
    y_clean = df["target"]
    X_clean = df.drop(columns="target")
    # Impute any remaining holes (just in case)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    model = LogisticRegression(**params)
    model.fit(X_imputed, y_clean)
    return model
