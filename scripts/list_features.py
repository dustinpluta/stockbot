# scripts/list_features.py

from pathlib import Path

from preprocessing.features import FEATURE_REGISTRY

def list_features():
    print("Available features:")
    for name in FEATURE_REGISTRY:
        print("-", name)

if __name__ == "__main__":
    list_features()
