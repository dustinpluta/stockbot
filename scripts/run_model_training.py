# scripts/run_model_training.py

import sys
import os
import yaml

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model.train import train_from_config

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_model_training.py <config_file.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)
    train_from_config(config)
