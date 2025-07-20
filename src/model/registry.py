# src/model/registry.py

from typing import Callable, Dict
import pandas as pd
import inspect

# Existing imports...
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """
    Decorator to register a model-specific training function.
    Signature: fn(X_train, y_train, X_val, y_val, params) -> fitted_model
    """
    def decorator(fn: Callable):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


# ——— Classification trainers ———

@register_model("xgboost")
def train_xgboost_classifier(X_train, y_train, X_val, y_val, params):
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, 
                eval_set=[(X_val, y_val)],
                verbose=False)
    return model

@register_model("random_forest")
def train_random_forest_classifier(X_train, y_train, X_val, y_val, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

@register_model("logistic_regression")
def train_logistic_regression(X_train, y_train, X_val, y_val, params):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    df = pd.concat([X_train, y_train.rename("target")], axis=1).dropna()
    Xc = df.drop(columns="target")
    yc = df["target"]
    imputer = SimpleImputer(strategy="mean")
    Xi = pd.DataFrame(imputer.fit_transform(Xc),
                      columns=Xc.columns, index=Xc.index)
    model = LogisticRegression(**params)
    model.fit(Xi, yc)
    return model

# ——— Regression trainers ———

@register_model("xgboost_regressor")
def train_xgboost_regressor(X_train, y_train, X_val, y_val, params):
    model = XGBRegressor(**params)
    model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
    return model

@register_model("random_forest_regressor")
def train_random_forest_regressor(X_train, y_train, X_val, y_val, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

@register_model("linear_regression")
def train_linear_regression(X_train, y_train, X_val, y_val, params):
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model
