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

# Global debug flag
DEBUG = True
