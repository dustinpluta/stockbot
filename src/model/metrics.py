# src/model/metrics.py

from typing import Callable, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

METRIC_REGISTRY: Dict[str, Callable] = {}

def register_metric(name: str):
    """Decorator to register an evaluation metric."""
    def decorator(fn: Callable):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

# ——— Classification metrics ———
@register_metric("accuracy")
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

@register_metric("f1")
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

@register_metric("recall")
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

@register_metric("precision")
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

# ——— Regression metrics ———
@register_metric("mse")
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

@register_metric("mae")
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

@register_metric("r2")
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
