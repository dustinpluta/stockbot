# src/config.py

from pathlib import Path

# Label settings
LABEL_HORIZON_HOURS = 1  # how far ahead we predict

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"

# Training settings
TEST_SIZE = 0.2  # fraction of data used for testing
RANDOM_STATE = 42

# Time‐series split definitions (UTC timestamps or ISO strings)
TRAIN_START   = "2024-07-01"
TRAIN_END     = "2025-01-01"
VALID_START   = "2025-01-02"
VALID_END     = "2025-05-01"
TEST_START    = "2025-05-02"
TEST_END      = "2025-07-01"

PERIOD = "390d"

# Derived if you like:
SPLIT_BOUNDS = {
    "train": (TRAIN_START, TRAIN_END),
    "validate": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}

# Feature sets
FEATURE_SETS = {
    "all": [  # default
        "rsi", "macd", "sma20", "return_1h", "volatility_5h", "momentum",
        "bollinger_upper", "bollinger_lower", "log_return_1h", "return_4h",
        "Close", "Open", "High", "Low", "Volume"
    ],
    "minimal": ["rsi", "return_1h", "Close", "Volume"],
    "price_only": ["Close", "Open", "High", "Low"],
    "momentum_focus": ["rsi", "macd", "momentum", "sma20", "volatility_5h"]
}

# Global debug flag
DEBUG = True
