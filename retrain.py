import os
import subprocess
import sys
import time

# ================= CONFIGURATION =================
# Update these paths to match your specific environment
PROJECT_ROOT = "/home/qj/projects/latin_bho"
PYLAIA_ENV_PATH = "/home/qj/projects/pylaia-env/bin/activate"

MODEL_DIR = "/home/qj/projects/latin_bho/bootstrap_training_data/pylaia_models/model_v13"
DATASET_DIR = "/home/qj/projects/latin_bho/bootstrap_training_data/datasets/dataset_v13"

CREATE_CONFIG_PATH = os.path.join(MODEL_DIR, "create_config.yaml")
TRAIN_CONFIG_PATH = os.path.join(MODEL_DIR, "train_config.yaml")
# =================================================

def write_configs(syms_path, img_dir, train_txt, val_txt):
    # --- 1. Architecture Config (Reverted to the Winning 5-Layer Style) ---
    with open(CREATE_CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"""syms: {syms_path}
fixed_input_height: 128
adaptive_pooling: avg
common:
  train_path: {MODEL_DIR}
  model_filename: model
crnn:
  # 5 Layers (Matches Config 1) - Better for complex scripts
  cnn_num_features: [16, 32, 64, 128, 256]
  cnn_kernel_size: [3, 3, 3, 3, 3]
  cnn_stride: [1, 1, 1, 1, 1]
  cnn_dilation: [1, 1, 1, 1, 1]
  cnn_activation: [LeakyReLU, LeakyReLU, LeakyReLU, LeakyReLU, LeakyReLU]
  cnn_batchnorm: [true, true, true, true, true]
  # Aggressive pooling [2,2] everywhere helps the LSTM converge faster
  cnn_poolsize: [2, 2, 2, 2, 2] 
  cnn_dropout: [0.0, 0.0, 0.0, 0.0, 0.0]
  rnn_type: LSTM
  rnn_layers: 3
  rnn_units: 512
  rnn_dropout: 0.5
  lin_dropout: 0.5
""")

    # --- 2. Training Config ---
    with open(TRAIN_CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"""syms: {syms_path}
img_dirs: [{img_dir}]
tr_txt_table: {train_txt}
va_txt_table: {val_txt}

common:
  train_path: {MODEL_DIR}
  model_filename: model
  monitor: va_cer
  checkpoint: null

data:
  # Increased to 16 for stability. If OOM error, lower to 8.
  batch_size: 16 
  color_mode: L
  num_workers: 8

optimizer:
  learning_rate: 0.0003
  name: Adam

scheduler:
  active: true
  monitor: va_cer
  patience: 20

# ENABLE GPU HERE
trainer:
  gpus: 1

train:
  # INCREASED: Must be higher than scheduler patience (20)
  # This gives the model 30 epochs to recover after LR drops.
  early_stopping_patience: 50 
  augment_training: true

logging:
  level: INFO
  filepath: train-crnn.log
  overwrite: true
""")
    print("Generated separate config files for Creation and Training.")

def run_command(cmd, step_name):
    print("\n" + "=" * 60)
    print(f"STEP: {step_name}")
    print(f"CMD: {cmd}")
    print("=" * 60)
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        executable='/bin/bash',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=PROJECT_ROOT
    )

    for line in process.stdout:
        print(line, end='')

    return_code = process.wait()
    if return_code != 0:
        print(f"\nFAILURE: {step_name} failed with code {return_code}")
        sys.exit(return_code)
    print(f"\nSUCCESS: {step_name} completed.")

def main():
    syms_file = os.path.join(DATASET_DIR, "syms.txt")
    train_file = os.path.join(DATASET_DIR, "train.txt")
    val_file = os.path.join(DATASET_DIR, "val.txt")
    img_dir = os.path.join(DATASET_DIR, "images")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    write_configs(syms_file, img_dir, train_file, val_file)

    # Note: We don't need to re-run create-model if it already succeeded, 
    # but running it again is harmless (it just overwrites the architecture file).
    cmd_create = (
        f"source {PYLAIA_ENV_PATH} && "
        f"pylaia-htr-create-model --config '{CREATE_CONFIG_PATH}'"
    )
    run_command(cmd_create, "Create Model Architecture")

    cmd_train = (
        f"ulimit -n 4096 2>/dev/null || true && "
        f"source {PYLAIA_ENV_PATH} && "
        f"pylaia-htr-train-ctc --config '{TRAIN_CONFIG_PATH}'"
    )
    run_command(cmd_train, "Train Model")

if __name__ == "__main__":
    main()