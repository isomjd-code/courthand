#!/usr/bin/env python3
"""
Resume ScrabbleGAN training with a larger batch size or start a fresh experiment.

This script:
1. (Optional) Copies model weights from old checkpoint (with old batch size) to new checkpoint directory
2. Resumes training with the new batch size, or starts fresh if --continue_train is not set

Usage:
    # Resume from checkpoint with larger batch size:
    python resume_with_larger_batch.py --old_batch_size 8 --new_batch_size 16 --model_name latin_bho_LatinBHOtrH32_GANres32 --continue_train
    
    # Start fresh experiment with lower learning rate:
    python resume_with_larger_batch.py --new_batch_size 16 --model_name latin_bho_LatinBHOtrH32_GANres32 --D_lr 0.00005 --name_suffix low_lr
"""

import os
import sys
import argparse
import shutil
import torch
from pathlib import Path

# Add scrabblegan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrabblegan'))


def copy_checkpoint_weights(old_checkpoint_dir, new_checkpoint_dir, epoch='latest'):
    """
    Copy model weights from old checkpoint to new checkpoint directory.
    
    Args:
        old_checkpoint_dir: Path to old checkpoint directory
        new_checkpoint_dir: Path to new checkpoint directory (will be created)
        epoch: Which epoch to copy ('latest' or specific epoch number)
    """
    print(f"Copying weights from {old_checkpoint_dir} to {new_checkpoint_dir}...")
    
    # Create new checkpoint directory
    os.makedirs(new_checkpoint_dir, exist_ok=True)
    
    # Determine checkpoint file names
    if epoch == 'latest':
        checkpoint_files = [
            'latest_net_G.pth',
            'latest_net_D.pth',
            'latest_net_OCR.pth'
        ]
    else:
        checkpoint_files = [
            f'{epoch}_net_G.pth',
            f'{epoch}_net_D.pth',
            f'{epoch}_net_OCR.pth'
        ]
    
    # Copy model weights
    copied_files = []
    for filename in checkpoint_files:
        old_path = os.path.join(old_checkpoint_dir, filename)
        new_path = os.path.join(new_checkpoint_dir, filename)
        
        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            copied_files.append(filename)
            print(f"  ✓ Copied {filename}")
        else:
            print(f"  ⚠ {filename} not found in old checkpoint")
    
    # Copy other important files
    other_files = ['loss_log.txt', 'train_opt.txt']
    for filename in other_files:
        old_path = os.path.join(old_checkpoint_dir, filename)
        new_path = os.path.join(new_checkpoint_dir, filename)
        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            print(f"  ✓ Copied {filename}")
    
    # Copy web directory if it exists
    old_web = os.path.join(old_checkpoint_dir, 'web')
    new_web = os.path.join(new_checkpoint_dir, 'web')
    if os.path.exists(old_web):
        if os.path.exists(new_web):
            shutil.rmtree(new_web)
        shutil.copytree(old_web, new_web)
        print(f"  ✓ Copied web directory")
    
    if not copied_files:
        print("  ❌ No checkpoint files found to copy!")
        return False
    
    print(f"\n✅ Successfully copied {len(copied_files)} model files")
    return True


def verify_checkpoint_compatibility(old_checkpoint_dir, new_batch_size):
    """
    Verify that the old checkpoint can be used with new batch size.
    Note: Batch size doesn't affect model architecture, so this should always work.
    """
    checkpoint_file = os.path.join(old_checkpoint_dir, 'latest_net_G.pth')
    if not os.path.exists(checkpoint_file):
        print(f"⚠ Warning: Cannot verify compatibility - {checkpoint_file} not found")
        return True
    
    try:
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in state_dict.values()):,}")
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def build_training_command(dataname, name_prefix, old_batch_size, new_batch_size, 
                          resolution=32, continue_train=True, **kwargs):
    """Build the training command with new batch size."""
    scrabblegan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrabblegan')
    
    cmd = [
        sys.executable,
        'train.py',
        '--dataname', dataname,
        '--name_prefix', name_prefix,
        '--dataset_mode', 'text',
        '--model', 'ScrabbleGAN',
        '--input_nc', '1',
        '--resolution', str(resolution),
        '--batch_size', str(new_batch_size),
    ]
    
    if continue_train:
        cmd.append('--continue_train')
    
    # Add discriminator fix parameters
    if 'D_lr' in kwargs:
        cmd.extend(['--D_lr', str(kwargs['D_lr'])])
    if 'num_critic_train' in kwargs:
        cmd.extend(['--num_critic_train', str(kwargs['num_critic_train'])])
    
    # Add any other parameters
    for key, value in kwargs.items():
        if key not in ['D_lr', 'num_critic_train'] and value is not None:
            cmd_key = key.replace('_', '-')
            cmd.extend([f'--{cmd_key}', str(value)])
    
    return cmd, scrabblegan_dir


def main():
    parser = argparse.ArgumentParser(
        description='Resume ScrabbleGAN training with larger batch size'
    )
    parser.add_argument('--old_batch_size', type=int, default=8,
                        help='Old batch size (default: 8)')
    parser.add_argument('--new_batch_size', type=int, required=True,
                        help='New batch size to use')
    parser.add_argument('--model_name', type=str, 
                        default='latin_bho_LatinBHOtrH32_GANres32',
                        help='Model name without batch size suffix')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Image resolution (default: 32)')
    parser.add_argument('--dataname', type=str, default='LatinBHOtrH32',
                        help='Dataset name')
    parser.add_argument('--name_prefix', type=str, default='latin_bho',
                        help='Experiment name prefix')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Which epoch to copy (default: latest)')
    parser.add_argument('--D_lr', type=float, default=0.0004,
                        help='Discriminator learning rate (default: 0.0004)')
    parser.add_argument('--num_critic_train', type=int, default=2,
                        help='Number of critic training steps (default: 2)')
    parser.add_argument('--continue_train', action='store_true',
                        help='Continue training from checkpoint (default: False, start fresh)')
    parser.add_argument('--name_suffix', type=str, default=None,
                        help='Optional suffix to add to experiment name (e.g., "low_lr" or "v2")')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without actually doing it')
    parser.add_argument('--skip_copy', action='store_true',
                        help='Skip copying weights (assumes already copied)')
    
    args = parser.parse_args()
    
    # Build checkpoint directory names
    scrabblegan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrabblegan')
    
    # Build name_prefix with optional suffix (this will be used in training command)
    name_prefix = args.name_prefix
    if args.name_suffix:
        name_prefix = f'{args.name_prefix}_{args.name_suffix}'
    
    # Build expected checkpoint directory name (matches how ScrabbleGAN constructs it)
    # Format: name_prefix_dataname_GANres{resolution}_bs{batch_size}
    old_checkpoint_dir = os.path.join(
        scrabblegan_dir, 
        'checkpoints', 
        f'{args.model_name}_bs{args.old_batch_size}'
    )
    new_checkpoint_name = f'{name_prefix}_{args.dataname}_GANres{args.resolution}_bs{args.new_batch_size}'
    new_checkpoint_dir = os.path.join(
        scrabblegan_dir,
        'checkpoints',
        new_checkpoint_name
    )
    
    print("=" * 60)
    if args.continue_train:
        print("Resume Training with Larger Batch Size")
    else:
        print("Start Fresh Training Experiment")
    print("=" * 60)
    print(f"Old batch size: {args.old_batch_size}")
    print(f"New batch size: {args.new_batch_size}")
    print(f"Continue from checkpoint: {args.continue_train}")
    print(f"Discriminator learning rate: {args.D_lr}")
    if args.name_suffix:
        print(f"Experiment name suffix: {args.name_suffix}")
    print(f"Old checkpoint: {old_checkpoint_dir}")
    print(f"New checkpoint: {new_checkpoint_dir}")
    print()
    
    # Verify old checkpoint exists only if continuing training
    if args.continue_train and not args.skip_copy:
        if not os.path.exists(old_checkpoint_dir):
            print(f"❌ Error: Old checkpoint directory not found: {old_checkpoint_dir}")
            sys.exit(1)
    
    # Verify checkpoint compatibility only if continuing
    if args.continue_train and not args.skip_copy:
        if not verify_checkpoint_compatibility(old_checkpoint_dir, args.new_batch_size):
            print("❌ Checkpoint compatibility check failed")
            sys.exit(1)
    
    # Copy weights if not skipping and continuing training
    if args.continue_train and not args.skip_copy:
        if args.dry_run:
            print(f"[DRY RUN] Would copy weights from {old_checkpoint_dir} to {new_checkpoint_dir}")
        else:
            success = copy_checkpoint_weights(old_checkpoint_dir, new_checkpoint_dir, args.epoch)
            if not success:
                print("❌ Failed to copy weights")
                sys.exit(1)
    else:
        print("⏭ Skipping weight copy (--skip_copy specified)")
    
    # Build training command
    train_kwargs = {
        'D_lr': args.D_lr,
        'num_critic_train': args.num_critic_train
    }
    
    cmd, work_dir = build_training_command(
        dataname=args.dataname,
        name_prefix=name_prefix,
        old_batch_size=args.old_batch_size,
        new_batch_size=args.new_batch_size,
        resolution=args.resolution,
        continue_train=args.continue_train,
        **train_kwargs
    )
    
    print("\n" + "=" * 60)
    print("Training Command")
    print("=" * 60)
    print(f"Working directory: {work_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would execute training command above")
        return
    
    # Ask for confirmation
    response = input("Start training with new batch size? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Execute training
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(cmd, cwd=work_dir)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    main()

