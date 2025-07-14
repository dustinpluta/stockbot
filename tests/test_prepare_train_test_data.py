import sys
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils import prepare_train_test_data

X_train, y_train, X_test, y_test = prepare_train_test_data(
    split_time="2025-06-01",
    tickers=["AAPL", "GOOG", "MSFT"],
    debug=True
)
