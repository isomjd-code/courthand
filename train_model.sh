#!/bin/bash

# --- 1. PREVENT WINDOW FROM CLOSING ---
function pause_on_exit {
    EXIT_CODE=$?
    echo ""
    echo "----------------------------------------------------"
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Script CRASHED with exit code $EXIT_CODE."
        echo "Check the error message above to see what went wrong."
    else
        echo "✅ Script COMPLETED successfully."
    fi
    echo "Press Enter to close this window..."
    read
}
trap pause_on_exit EXIT

echo "--- Starting Pylaia Training (Final Fix) ---"

# --- 2. ENVIRONMENT & DIRECTORIES ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 4096 2>/dev/null || true

DATA_DIR="pylaia_dataset_output"
MODEL_DIR="model_courthand_3090C"

# Ensure directories exist
if [ ! -d "$MODEL_DIR" ]; then mkdir -p "$MODEL_DIR"; fi
if [ ! -d "$DATA_DIR" ]; then echo "Error: Data directory '$DATA_DIR' not found."; exit 1; fi

# Copy syms if missing
if [ ! -f "${MODEL_DIR}/syms.txt" ]; then
    if [ -f "${DATA_DIR}/syms.txt" ]; then
        cp "${DATA_DIR}/syms.txt" "${MODEL_DIR}/syms.txt"
    else
        echo "Error: Could not find 'syms.txt' in data directory."
        exit 1
    fi
fi

# --- 3. RESUME LOGIC ---
if [ -f "${MODEL_DIR}/model" ]; then
    echo "ℹ️  Existing model found. RESUMING training."
    DO_CREATE_MODEL=false
    RESUME_FLAG="true"
else
    echo "🆕 No existing model found. CREATING new 'High-Res' model."
    if [ -d "${MODEL_DIR}/experiment" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mv "${MODEL_DIR}/experiment" "${MODEL_DIR}/experiment_backup_${TIMESTAMP}" 2>/dev/null || true
    fi
    DO_CREATE_MODEL=true
    RESUME_FLAG="false"
fi

# --- 4. CREATE CONFIGURATION FILES ---

# 4A. Model Architecture Config
# FIX: explicitly provided 'cnn_dropout' and 'cnn_stride' as lists of length 5
# to match 'cnn_num_features'. All lists must be exactly the same length.
echo "Writing working_create_config.yaml..."
cat <<EOF > working_create_config.yaml
syms: ${MODEL_DIR}/syms.txt
fixed_input_height: 128
adaptive_pooling: avg

common:
  train_path: ./${MODEL_DIR}
  model_filename: model

logging:
  level: INFO
  filepath: train-crnn.log
  overwrite: ${DO_CREATE_MODEL}

crnn:
  # 5-Layer Deep CNN (Len 5)
  cnn_num_features: [16, 32, 64, 128, 256]
  cnn_kernel_size: [3, 3, 3, 3, 3]
  cnn_stride: [1, 1, 1, 1, 1]
  cnn_dilation: [1, 1, 1, 1, 1]
  cnn_activation: [LeakyReLU, LeakyReLU, LeakyReLU, LeakyReLU, LeakyReLU]
  cnn_batchnorm: [true, true, true, true, true]
  cnn_poolsize: [2, 2, 2, 2, 0]
  # Explicit dropout list is required to match dimensions:
  cnn_dropout: [0.0, 0.0, 0.0, 0.0, 0.0]
  
  rnn_type: LSTM
  rnn_layers: 3
  rnn_units: 512
  rnn_dropout: 0.5
  lin_dropout: 0.5
EOF

# 4B. Training Config
echo "Writing working_train_config.yaml..."
cat <<EOF > working_train_config.yaml
syms: ${MODEL_DIR}/syms.txt
img_dirs:
  - ${DATA_DIR}/images
  - ${DATA_DIR}/
tr_txt_table: ${DATA_DIR}/train.txt
va_txt_table: ${DATA_DIR}/val.txt

data:
  batch_size: 24
  num_workers: 4
  color_mode: L

train:
  delimiters: ["<space>"]
  early_stopping_patience: 40
  checkpoint_k: 5
  resume: ${RESUME_FLAG}
  augment_training: true

common:
  train_path: ./${MODEL_DIR}
  model_filename: model
  monitor: va_cer

trainer:
  max_epochs: 500
  accelerator: gpu
  devices: [0]
  precision: 16

optimizer:
  name: Adam
  learning_rate: 0.0001
EOF

# --- 5. EXECUTE PYLAIA ---

if [ "$DO_CREATE_MODEL" = true ]; then
    echo "🔨 Creating Model Architecture..."
    # Using config only prevents command line arg parsing errors
    pylaia-htr-create-model --config working_create_config.yaml
else
    echo "⏭️  Skipping model creation."
fi

echo "🚀 Starting Training Loop..."
pylaia-htr-train-ctc --config working_train_config.yaml

echo "🎉 Training Finished!"