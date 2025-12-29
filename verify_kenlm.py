#!/usr/bin/env python3
"""Quick verification script for KenLM configuration."""

import os
import sys

# Simulate what settings.py does to get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KENLM_MODEL_PATH = os.path.join(BASE_DIR, "kenlm_model", "kenlm_model_3gram.arpa")
KENLM_MODEL_WEIGHT = 1.5
KENLM_USE_BINARY = False

print('=== KenLM Configuration Check ===')
print(f'BASE_DIR: {BASE_DIR}')
print(f'KENLM_MODEL_PATH: {KENLM_MODEL_PATH}')
print(f'KENLM_MODEL_WEIGHT: {KENLM_MODEL_WEIGHT}')
print(f'KENLM_USE_BINARY: {KENLM_USE_BINARY}')
print()
print(f'ARPA model exists: {os.path.exists(KENLM_MODEL_PATH)}')

binary_path = KENLM_MODEL_PATH.replace('.arpa', '.klm')
print(f'Binary model path: {binary_path}')
print(f'Binary model exists: {os.path.exists(binary_path)}')

# Check file sizes
if os.path.exists(KENLM_MODEL_PATH):
    size_mb = os.path.getsize(KENLM_MODEL_PATH) / (1024*1024)
    print(f'ARPA model size: {size_mb:.2f} MB')
if os.path.exists(binary_path):
    size_mb = os.path.getsize(binary_path) / (1024*1024)
    print(f'Binary model size: {size_mb:.2f} MB')

print()
print('=== Model will be used if: ===')
print(f'  - KENLM_MODEL_PATH is set: {"✓ YES" if KENLM_MODEL_PATH else "✗ NO"}')
print(f'  - Model file exists: {"✓ YES" if os.path.exists(KENLM_MODEL_PATH) else "✗ NO"}')
print()
if KENLM_MODEL_PATH and os.path.exists(KENLM_MODEL_PATH):
    print('✓ KenLM IS configured and will be used for PyLaia decoding!')
else:
    print('✗ KenLM will NOT be used - check path and file existence.')

