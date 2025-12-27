#!/bin/bash
# Start fresh ScrabbleGAN training with balanced learning rates
# This script:
# 1. Starts G and D from scratch (random initialization)
# 2. Loads OCR from previous checkpoint
# 3. Uses balanced learning rates (D_lr = G_lr = 0.0002)

cd scrabblegan

python train.py \
    --dataname LatinBHOtrH32 \
    --name_prefix latin_bho \
    --dataset_mode text \
    --model ScrabbleGAN \
    --input_nc 1 \
    --resolution 32 \
    --batch_size 16 \
    --G_lr 0.0002 \
    --D_lr 0.0002 \
    --OCR_lr 0.0002 \
    --num_critic_train 1 \
    --G_depth 2 \
    --num_accumulations 1 \
    --save_epoch_freq 5 \
    --print_freq 100 \
    --OCR_init ./checkpoints/ocr_only \
    --G_init N02 \
    --D_init N02

