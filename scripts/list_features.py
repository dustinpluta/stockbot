# scripts/list_features.py

from pathlib import Path
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.features import FEATURE_REGISTRY

def list_features():
    print("Available features:")
    for name in FEATURE_REGISTRY:
        print("-", name)

if __name__ == "__main__":
    list_features()
