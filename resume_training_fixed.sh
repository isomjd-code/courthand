#!/bin/bash
# Resume training with correct parameters to match existing checkpoint

cd scrabblegan

python train.py \
    --dataname LatinBHOtrH32 \
    --name_prefix latin_bho \
    --dataset_mode text \
    --model ScrabbleGAN \
    --input_nc 1 \
    --resolution 32 \
    --continue_train \
    --D_lr 0.0004 \
    --num_critic_train 2

