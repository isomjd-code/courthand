#!/usr/bin/env python3
"""
Training script for ScrabbleGAN using Latin BHO dataset.

This script:
1. Converts PyLaia format to LMDB if needed
2. Trains the ScrabbleGAN model

Usage:
    python train_scrabblegan.py --convert-data  # Convert data first
    python train_scrabblegan.py --train         # Train the model
    python train_scrabblegan.py --convert-data --train  # Do both
"""

import os
import sys
import argparse
import subprocess

def convert_data(input_dir='bootstrap_training_data/datasets/dataset_v22', output_dir='Datasets/LatinBHO'):
    """Convert PyLaia format to LMDB format."""
    print("=" * 60)
    print("Converting PyLaia format to LMDB format...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        print(f"\nConverting {split} split...")
        cmd = [
            sys.executable,
            'convert_to_lmdb.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--split', split
        ]
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"Error converting {split} split")
            return False
    
    print("\n" + "=" * 60)
    print("Data conversion completed!")
    print("=" * 60)
    return True

def train_model(dataname='LatinBHOtrH32', name_prefix='latin_bho', **kwargs):
    """Train the ScrabbleGAN model."""
    print("=" * 60)
    print("Starting ScrabbleGAN training...")
    print("=" * 60)
    
    # Change to scrabblegan directory
    scrabblegan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrabblegan')
    
    # Build training command
    cmd = [
        sys.executable,
        'train.py',
        '--dataname', dataname,
        '--name_prefix', name_prefix,
        '--dataset_mode', 'text',
        '--model', 'ScrabbleGAN',
        '--input_nc', '1',  # Grayscale
        '--resolution', '32',
        # Note: --labeled uses action='store_false', so we don't pass it
        # (default is labeled=True, which is what we want)
    ]
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {scrabblegan_dir}")
    
    result = subprocess.run(cmd, cwd=scrabblegan_dir)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Train ScrabbleGAN on Latin BHO dataset')
    parser.add_argument('--convert-data', action='store_true',
                        help='Convert PyLaia format to LMDB format')
    parser.add_argument('--train', action='store_true',
                        help='Train the ScrabbleGAN model')
    parser.add_argument('--input-dir', type=str, default='bootstrap_training_data/datasets/dataset_v22',
                        help='Input directory with PyLaia format data')
    parser.add_argument('--output-dir', type=str, default='Datasets/LatinBHO',
                        help='Output directory for LMDB files')
    parser.add_argument('--dataname', type=str, default='LatinBHOtrH32',
                        help='Dataset name (default: LatinBHOtrH32)')
    parser.add_argument('--name-prefix', type=str, default='latin_bho',
                        help='Experiment name prefix')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: use ScrabbleGAN default)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: use ScrabbleGAN default)')
    
    args = parser.parse_args()
    
    success = True
    
    if args.convert_data:
        success = convert_data(args.input_dir, args.output_dir)
        if not success:
            print("Data conversion failed. Exiting.")
            sys.exit(1)
    
    if args.train:
        train_kwargs = {}
        if args.epochs:
            train_kwargs['niter'] = args.epochs
            train_kwargs['niter_decay'] = 0
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        
        success = train_model(
            dataname=args.dataname,
            name_prefix=args.name_prefix,
            **train_kwargs
        )
        if not success:
            print("Training failed. Exiting.")
            sys.exit(1)
    
    if not args.convert_data and not args.train:
        parser.print_help()
        print("\nPlease specify --convert-data and/or --train")

if __name__ == '__main__':
    main()

