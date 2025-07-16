#!/bin/bash

# Root directory (customize if running from elsewhere)
ROOT_DIR=$(pwd)

echo "📁 Creating new project structure..."

# Top-level structure
mkdir -p "$ROOT_DIR"/{config,data/features,data/models,notebooks,scripts,tests}

# src submodules
mkdir -p "$ROOT_DIR"/src/{core,pipeline,model,sim,utils}

# Move existing code if present
echo "📦 Porting existing code..."

# Move config.py
if [ -f "$ROOT_DIR/src/config.py" ]; then
  mv "$ROOT_DIR/src/config.py" "$ROOT_DIR/config/config.py"
fi

# Move core modules
for f in data_fetch.py features.py split.py label.py; do
  if [ -f "$ROOT_DIR/src/$f" ]; then
    mv "$ROOT_DIR/src/$f" "$ROOT_DIR/src/core/$f"
  fi
done

# Move pipeline code
if [ -f "$ROOT_DIR/src/feature_pipeline.py" ]; then
  mv "$ROOT_DIR/src/feature_pipeline.py" "$ROOT_DIR/src/pipeline/feature_pipeline.py"
fi

# Move model training logic if exists
if [ -f "$ROOT_DIR/src/model.py" ]; then
  mv "$ROOT_DIR/src/model.py" "$ROOT_DIR/src/model/runner.py"
fi

# Move utility files if any
for f in utils.py summarize.py; do
  if [ -f "$ROOT_DIR/src/$f" ]; then
    mv "$ROOT_DIR/src/$f" "$ROOT_DIR/src/utils/$f"
  fi
done

# Make starter scripts if missing
touch "$ROOT_DIR"/scripts/{run_feature_pipeline.py,train_model.py,predict_live.py,simulate_trading_day.py}
touch "$ROOT_DIR"/src/sim/{backtest.py,trade_engine.py}
touch "$ROOT_DIR"/src/model/{metrics.py,tuner.py}
touch "$ROOT_DIR"/src/utils/{io.py,tickers.py,time.py}

echo "✅ Structure setup complete."
