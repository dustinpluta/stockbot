# tests/test_concatenate_feature_files.py

import sys
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils import concatenate_feature_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python tests/test_concatenate_feature_files.py <feature_dir>")
        sys.exit(1)

    feature_dir = Path(sys.argv[1])
    if not feature_dir.exists() or not feature_dir.is_dir():
        print(f"[ERROR] Directory does not exist: {feature_dir}")
        sys.exit(1)

    print(f"\n=== Running concatenate_feature_files on {feature_dir} ===")
    try:
        df = concatenate_feature_files(feature_dir=feature_dir, debug=True)
        print(f"\n[PASS] Successfully concatenated features ({len(df)} total rows).")
    except Exception as e:
        print(f"\n[FAIL] Error during concatenation: {e}")

if __name__ == "__main__":
    main()
