#!/usr/bin/env python3
import argparse
import sys
import yaml
from pathlib import Path

# Ensure src/ is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.train import train_from_config

def main():
    parser = argparse.ArgumentParser(
        description="CI runner: train models from config and exit non-zero on any failure"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "config",
        nargs="?",
        help="Path to a single model config YAML file"
    )
    args = parser.parse_args()

    config_file = Path(args.config)

    if not config_file.is_file():
        print(f"[ERROR] Config file not found: {config_file}", file=sys.stderr)
    try:
        cfg = yaml.safe_load(config_file.read_text())
        train_from_config(cfg)
    except Exception as e:
        print(f"[ERROR] Training failed for {config_file}:\n{e}", file=sys.stderr)

if __name__ == "__main__":
    main()
