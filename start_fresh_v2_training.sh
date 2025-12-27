#!/bin/bash
# Start fresh ScrabbleGAN training (v2) with new Generator/Discriminator
# This script:
# 1. Starts G and D from scratch (random initialization with N02)
# 2. Loads OCR from pre-trained checkpoint (ocr_only directory)
# 3. Uses balanced learning rates (G_lr = D_lr = 0.0002)
# 4. Uses weakened discriminator (num_critic_train 1) to prevent collapse
# 5. Uses deeper generator (G_depth 2)

cd scrabblegan

/home/qj/projects/pylaia-env/bin/python train.py \
    --dataname LatinBHOtrH32 \
    --name_prefix latin_bho_v2 \
    --dataset_mode text \
    --model ScrabbleGAN \
    --input_nc 1 \
    --resolution 32 \
    --batch_size 16 \
    --num_accumulations 1 \
    --G_lr 0.0002 \
    --D_lr 0.0002 \
    --OCR_lr 0.0002 \
    --num_critic_train 1 \
    --G_depth 2 \
    --save_epoch_freq 5 \
    --print_freq 100 \
    --OCR_init ./checkpoints/ocr_only \
    --G_init N02 \
    --D_init N02

