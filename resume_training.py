#!/usr/bin/env python3
"""
Simple script to resume Pylaia model training from where it left off.

Usage:
    python resume_training.py [model_version]
    
    If model_version is not provided, defaults to v13.
    Example: python resume_training.py 13
"""

import argparse
import os
import subprocess
import sys

# Base directory for models
BASE_MODEL_DIR = "/home/qj/projects/latin_bho/bootstrap_training_data/pylaia_models"
PYLAIA_ENV = os.path.expanduser("~/projects/pylaia-env/bin/activate")

def main():
    """Resume training from the specified model directory."""
    parser = argparse.ArgumentParser(
        description="Resume Pylaia model training from where it left off"
    )
    parser.add_argument(
        "model_version",
        type=int,
        nargs="?",
        default=13,
        help="Model version number to resume (default: 13)"
    )
    args = parser.parse_args()
    
    # Build model directory path
    model_dir = os.path.join(BASE_MODEL_DIR, f"model_v{args.model_version}")
    
    # Convert Windows path to Linux path if provided
    if model_dir.startswith("\\\\wsl.localhost\\") or model_dir.startswith("\\\\wsl$\\"):
        # Convert \\wsl.localhost\Ubuntu\home\qj\... to /home/qj/...
        model_dir = model_dir.replace("\\", "/")
        if "Ubuntu" in model_dir:
            model_dir = model_dir.split("Ubuntu")[-1]
        elif "wsl.localhost" in model_dir:
            model_dir = "/" + model_dir.split("wsl.localhost/")[-1]
    
    # Ensure absolute path
    model_dir = os.path.abspath(model_dir)
    train_config_path = os.path.join(model_dir, "train_config.yaml")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Check if train_config.yaml exists
    if not os.path.exists(train_config_path):
        print(f"❌ Error: Training config not found: {train_config_path}")
        sys.exit(1)
    
    # Check if model file exists
    model_file = os.path.join(model_dir, "model")
    if not os.path.exists(model_file):
        print(f"❌ Error: Model file not found: {model_file}")
        sys.exit(1)
    
    # Check for checkpoints
    experiment_dir = os.path.join(model_dir, "experiment")
    has_checkpoints = False
    if os.path.exists(experiment_dir):
        checkpoint_files = [f for f in os.listdir(experiment_dir) if f.endswith('.ckpt')]
        has_checkpoints = len(checkpoint_files) > 0
        if has_checkpoints:
            print(f"✅ Found {len(checkpoint_files)} checkpoint(s) in experiment directory")
        else:
            print(f"ℹ️  No checkpoints found in experiment directory - will start training from scratch")
    else:
        print(f"ℹ️  Experiment directory not found - will start training from scratch")
    
    # Update train_config.yaml based on whether checkpoints exist
    if has_checkpoints:
        print(f"📝 Updating training config to resume training...")
        with open(train_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Replace resume: false with resume: true
        if 'resume: false' in config_content:
            config_content = config_content.replace('resume: false', 'resume: true')
            with open(train_config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print("✅ Updated config: resume: true")
        elif 'resume: true' in config_content:
            print("✅ Config already set to resume: true")
        else:
            print("⚠️  Warning: Could not find 'resume:' setting in config, but continuing anyway...")
    else:
        print(f"📝 Updating training config to start fresh training...")
        with open(train_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Replace resume: true with resume: false
        if 'resume: true' in config_content:
            config_content = config_content.replace('resume: true', 'resume: false')
            with open(train_config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print("✅ Updated config: resume: false")
        elif 'resume: false' in config_content:
            print("✅ Config already set to resume: false")
        else:
            print("⚠️  Warning: Could not find 'resume:' setting in config, but continuing anyway...")
    
    # Check if Pylaia environment exists
    if not os.path.exists(PYLAIA_ENV):
        print(f"⚠️  Warning: Pylaia environment not found at {PYLAIA_ENV}")
        print("   Attempting to run without explicit activation...")
        activate_cmd = ""
    else:
        activate_cmd = f"source {PYLAIA_ENV} && "
    
    # Build training command
    # Set ulimit like train_model.sh does (increase file descriptor limit)
    cmd = (
        f"ulimit -n 4096 2>/dev/null || true && "
        f"{activate_cmd}"
        f"pylaia-htr-train-ctc --config '{train_config_path}'"
    )
    
    print(f"\n🚀 Starting training...")
    print(f"   Model directory: {model_dir}")
    print(f"   Config file: {train_config_path}")
    print(f"   Resuming from checkpoints: {has_checkpoints}")
    print()
    
    # Run training command
    try:
        # Use bash to run the command (handles source and && properly)
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=model_dir,
            check=False  # Don't raise on non-zero exit, let user see the output
        )
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            return 0
        else:
            print(f"\n❌ Training exited with code {result.returncode}")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running training command: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

