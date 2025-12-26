#!/usr/bin/env python3
"""Script to permanently delete the output_old directory."""
import shutil
import os
import sys

target_dir = "cp40_processing/output_old"

if os.path.exists(target_dir):
    print(f"Deleting {target_dir}...")
    try:
        shutil.rmtree(target_dir)
        print(f"Successfully deleted {target_dir}")
    except Exception as e:
        print(f"Error deleting {target_dir}: {e}")
        sys.exit(1)
else:
    print(f"Directory {target_dir} does not exist")
    sys.exit(0)

