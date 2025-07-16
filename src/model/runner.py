# src/model.py

import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import FEATURE_DIR, MODEL_DIR
from utils import prepare_train_test_data

def train_xgboost_with_cv(
    model_name: str = "xgb_model_cv",
    feature_dir: Path = FEATURE_DIR,
    split_time: str = "2025-06-01",
    tickers: list[str] = None,
    save_model: bool = True,
    debug: bool = False,
    cv_folds: int = 5,
    scoring: str = "accuracy"
) -> xgb.XGBClassifier:
    """
    Train an XGBoost model with cross-validated hyperparameter tuning.

    Args:
        model_name (str): Name to save the best model under
        feature_dir (Path): Feature file directory
        split_time (str): Date string for train/test split
        tickers (list[str]): Optional list of tickers
        save_model (bool): Save best model to disk
        debug (bool): Print training stats
        cv_folds (int): Number of CV folds
        scoring (str): Metric to optimize (e.g., 'accuracy', 'f1')

    Returns:
        Trained best XGBClassifier
    """
    X_train, y_train, X_test, y_test = prepare_train_test_data(
        feature_dir=feature_dir,
        split_time=split_time,
        tickers=tickers,
        debug=debug
    )

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }

    base_model = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    print("\n[INFO] Starting cross-validation...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n=== Cross-Validation Results ===")
    print(f"Best Score ({scoring}): {grid_search.best_score_:.3f}")
    print(f"Best Params: {grid_search.best_params_}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{model_name}.json"
        best_model.save_model(model_path)
        print(f"[INFO] Best model saved to {model_path}")

    return best_model

if __name__ == "__main__":
    train_xgboost_with_cv(
        model_name="xgb_cv_2025_06_01",
        tickers=["AAPL", "GOOG", "MSFT"],
        cv_folds=5
    )
