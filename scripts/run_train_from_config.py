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
        "--all",
        action="store_true",
        help="Run training for all YAML configs in src/model/configs"
    )
    group.add_argument(
        "config",
        nargs="?",
        help="Path to a single model config YAML file"
    )
    args = parser.parse_args()

    if args.all:
        cfg_dir = Path(__file__).resolve().parents[1] / "src" / "model" / "configs"
        config_files = sorted(cfg_dir.glob("*.yaml"))
        if not config_files:
            print(f"[ERROR] No config files found in {cfg_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        config_files = [Path(args.config)]

    overall_ok = True
    for cfg_path in config_files:
        print(f"\n=== CI: Training config {cfg_path} ===")
        if not cfg_path.is_file():
            print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
            overall_ok = False
            continue

        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            train_from_config(cfg)
        except Exception as e:
            print(f"[ERROR] Training failed for {cfg_path}:\n{e}", file=sys.stderr)
            overall_ok = False

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
