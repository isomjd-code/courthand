#!/usr/bin/env python3
"""
Diagnostic and fix script for discriminator issues in ScrabbleGAN training.

This script helps diagnose why the discriminator loss is 0 and provides fixes.
"""

import argparse
import sys
import os

# Add scrabblegan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrabblegan'))

def diagnose_discriminator(checkpoint_dir):
    """Diagnose discriminator state from checkpoint."""
    import torch
    
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_net_D.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Discriminator checkpoint not found: {checkpoint_path}")
        return
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"\nDiscriminator checkpoint loaded from: {checkpoint_path}")
    print(f"Number of parameters: {sum(p.numel() for p in state_dict.values())}")
    
    # Check for any NaN or Inf values
    has_nan = False
    has_inf = False
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"  WARNING: NaN found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  WARNING: Inf found in {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN or Inf values found")
    
    # Check parameter magnitudes
    print("\nParameter statistics:")
    for name, param in list(state_dict.items())[:5]:  # First 5 layers
        print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, "
              f"min={param.min().item():.6f}, max={param.max().item():.6f}")


def create_training_fix():
    """Create a modified training script with discriminator fixes."""
    
    fix_content = '''# Discriminator Fixes for ScrabbleGAN Training

## Issue
Discriminator loss is consistently 0.000, which means:
- dis_real >= 1 (discriminator correctly identifies real images)
- dis_fake <= -1 (discriminator correctly identifies fake images)
- No gradients flowing to discriminator (loss is already optimal)

## Possible Causes
1. Discriminator is too strong/confident
2. Generator is too weak (producing obviously fake images)
3. Learning rate imbalance
4. Training frequency imbalance

## Fixes to Try

### Fix 1: Increase Discriminator Learning Rate
Add to training command:
```bash
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --D_lr 0.0004
```
(Default is 0.0002, doubling it may help)

### Fix 2: Train Discriminator More Often
Add to training command:
```bash
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --num_critic_train 2
```
(Default is 4, reducing to 2 means D trains twice as often)

### Fix 3: Add Label Smoothing (requires code modification)
Modify scrabblegan/util/util.py loss_hinge_dis function to add label smoothing.

### Fix 4: Check Discriminator Outputs
Add debugging to see actual discriminator outputs:
- If dis_real >> 1 and dis_fake << -1, discriminator is too confident
- If outputs are near 0, there might be a different issue

### Fix 5: Reduce Generator Learning Rate
If generator is too weak:
```bash
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --G_lr 0.0001
```

### Fix 6: Use Different GAN Mode
Try Wasserstein GAN:
```bash
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --gan_mode wgangp
```

## Recommended Approach
1. First try Fix 1 (increase D_lr to 0.0004)
2. If that doesn't help, try Fix 2 (reduce num_critic_train to 2)
3. Monitor if discriminator loss starts showing non-zero values
4. If still 0, the discriminator might be working correctly but the generator needs improvement
'''
    
    with open('DISCRIMINATOR_FIX_GUIDE.md', 'w') as f:
        f.write(fix_content)
    
    print("Created DISCRIMINATOR_FIX_GUIDE.md with recommendations")


def main():
    parser = argparse.ArgumentParser(description='Diagnose and fix discriminator issues')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='scrabblegan/checkpoints/latin_bho_LatinBHOtrH32_GANres32_bs8',
                        help='Path to checkpoint directory')
    parser.add_argument('--create_fix_guide', action='store_true',
                        help='Create a guide with fix recommendations')
    
    args = parser.parse_args()
    
    if args.create_fix_guide:
        create_training_fix()
    
    if os.path.exists(args.checkpoint_dir):
        diagnose_discriminator(args.checkpoint_dir)
    else:
        print(f"Checkpoint directory not found: {args.checkpoint_dir}")
        print("Creating fix guide instead...")
        create_training_fix()


if __name__ == '__main__':
    main()

