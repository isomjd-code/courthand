#!/bin/bash

# Simple script to resume Pylaia model training from where it left off
#
# Usage:
#   ./resume_training.sh [model_version]
#   
#   If model_version is not provided, defaults to v13.
#   Example: ./resume_training.sh 13

# Get model version from argument or default to 13
MODEL_VERSION=${1:-13}

# Model directory
BASE_MODEL_DIR="/home/qj/projects/latin_bho/bootstrap_training_data/pylaia_models"
MODEL_DIR="${BASE_MODEL_DIR}/model_v${MODEL_VERSION}"
PYLAIA_ENV="$HOME/projects/pylaia-env/bin/activate"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if train_config.yaml exists
TRAIN_CONFIG="$MODEL_DIR/train_config.yaml"
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "❌ Error: Training config not found: $TRAIN_CONFIG"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_DIR/model" ]; then
    echo "❌ Error: Model file not found: $MODEL_DIR/model"
    exit 1
fi

# Check for checkpoints
EXPERIMENT_DIR="$MODEL_DIR/experiment"
if [ -d "$EXPERIMENT_DIR" ]; then
    CHECKPOINT_COUNT=$(find "$EXPERIMENT_DIR" -name "*.ckpt" | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "✅ Found $CHECKPOINT_COUNT checkpoint(s) in experiment directory"
    fi
fi

# Update train_config.yaml to set resume: true
echo "📝 Updating training config to resume training..."
if grep -q "resume: false" "$TRAIN_CONFIG"; then
    sed -i 's/resume: false/resume: true/' "$TRAIN_CONFIG"
    echo "✅ Updated config: resume: true"
elif grep -q "resume: true" "$TRAIN_CONFIG"; then
    echo "✅ Config already set to resume: true"
else
    echo "⚠️  Warning: Could not find 'resume:' setting in config"
fi

# Set environment variables (like train_model.sh)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 4096 2>/dev/null || true

# Activate Pylaia environment if it exists
if [ -f "$PYLAIA_ENV" ]; then
    source "$PYLAIA_ENV"
else
    echo "⚠️  Warning: Pylaia environment not found at $PYLAIA_ENV"
    echo "   Attempting to run without explicit activation..."
fi

# Run training
echo ""
echo "🚀 Starting training..."
echo "   Model directory: $MODEL_DIR"
echo "   Config file: $TRAIN_CONFIG"
echo ""

cd "$MODEL_DIR" || exit 1
pylaia-htr-train-ctc --config "$TRAIN_CONFIG"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "❌ Training exited with code $EXIT_CODE"
fi

exit $EXIT_CODE

