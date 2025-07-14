# src/config.py

from pathlib import Path

# List of features used for training and prediction
FEATURE_COLUMNS = [
    "Close", "High", "Low", "Open", "Volume",
    "rsi", "macd", "sma20",
    "return_1h", "return_4h", "log_return_1h",
    "volatility_5h", "momentum",
    "bollinger_upper", "bollinger_lower"
]

# Label settings
LABEL_HORIZON_HOURS = 1  # how far ahead we predict

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"

# Training settings
TEST_SIZE = 0.2  # fraction of data used for testing
RANDOM_STATE = 42

# Global debug flag
DEBUG = True
