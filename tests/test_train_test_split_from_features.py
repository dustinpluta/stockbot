import sys
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils import concatenate_feature_files, train_test_split_from_features

df = concatenate_feature_files(feature_dir=Path("data/features"))
train_df, test_df = train_test_split_from_features(df, split_time="2025-05-01", debug=True)
