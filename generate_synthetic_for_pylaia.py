#!/usr/bin/env python3
"""
Generate synthetic handwritten text images from ScrabbleGAN and convert to PyLaia format.

This script:
1. Generates images from a trained ScrabbleGAN model
2. Converts them to PyLaia dataset format (images + text files)

Usage:
    python generate_synthetic_for_pylaia.py --model_name latin_bho_LatinBHOtrH32_GANres32_bs8 --epoch latest --n_images 10000 --output_dir synthetic_pylaia_data
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import uuid

# Add scrabblegan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrabblegan'))

from options.test_options import TestOptions
from models import create_model
from util.util import prepare_z_y


def tokenize_text(text):
    """
    Convert text to PyLaia tokenized format.
    Example: "hello world" -> "h e l l o <space> w o r l d"
    """
    tokens = []
    for char in text:
        if char == ' ':
            tokens.append('<space>')
        else:
            tokens.append(char)
    return ' '.join(tokens)


def generate_images(model, opt, n_images, words_list=None):
    """
    Generate synthetic images from ScrabbleGAN model.
    
    Args:
        model: Trained ScrabbleGAN model
        opt: Test options
        n_images: Number of images to generate
        words_list: Optional list of specific words to generate (if None, samples from lexicon)
    
    Returns:
        List of (image, text) tuples
    """
    model.eval()
    results = []
    
    # Prepare noise vectors
    model.z, model.label_fake = prepare_z_y(
        opt.batch_size, opt.dim_z, len(model.lex),
        device=model.device, fp16=opt.G_fp16
    )
    
    print(f"Generating {n_images} images...")
    
    with torch.no_grad():
        for i in tqdm(range(n_images)):
            # Sample words from lexicon or use provided words
            if words_list is None:
                model.label_fake.sample_()
                words = [model.lex[int(j)] for j in model.label_fake]
            else:
                # Use specific words (cycle through if needed)
                word_idx = i % len(words_list)
                words = [words_list[word_idx]]
            
            # Generate image
            model.forward(words=words)
            
            # Convert to PIL Image
            fake_img = model.fake.data.cpu().numpy().squeeze(0).squeeze(0)
            # Normalize to 0-255
            if fake_img.max() <= 1.0:
                fake_img = (fake_img * 255).astype(np.uint8)
            else:
                fake_img = fake_img.astype(np.uint8)
            
            # Convert to PIL Image (grayscale)
            img = Image.fromarray(fake_img, mode='L')
            
            # Get text
            text = words[0] if isinstance(words, list) else words
            
            results.append((img, text))
    
    return results


def save_pylaia_format(images_texts, output_dir, split='train'):
    """
    Save images and text in PyLaia format.
    
    Args:
        images_texts: List of (image, text) tuples
        output_dir: Output directory
        split: Dataset split (train/val/test)
    """
    images_dir = os.path.join(output_dir, 'images', split)
    os.makedirs(images_dir, exist_ok=True)
    
    # Create text file
    text_file = os.path.join(output_dir, f'{split}.txt')
    
    print(f"Saving {len(images_texts)} images to {images_dir}...")
    
    with open(text_file, 'w', encoding='utf-8') as f:
        for idx, (img, text) in enumerate(tqdm(images_texts)):
            # Generate unique filename
            filename = f"synth_{uuid.uuid4().hex[:8]}.png"
            image_path = os.path.join(images_dir, filename)
            
            # Save image
            img.save(image_path)
            
            # Tokenize text
            tokenized = tokenize_text(text)
            
            # Write to text file (PyLaia format: relative_path tokenized_text)
            relative_path = f"{split}/{filename}"
            f.write(f"{relative_path} {tokenized}\n")
    
    print(f"Saved {len(images_texts)} images and text file: {text_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic images from ScrabbleGAN for PyLaia training')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the trained ScrabbleGAN model (e.g., latin_bho_LatinBHOtrH32_GANres32_bs8)')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Epoch to load (default: latest)')
    parser.add_argument('--n_images', type=int, default=10000,
                        help='Number of images to generate (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='synthetic_pylaia_data',
                        help='Output directory for PyLaia format data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split name (default: train)')
    parser.add_argument('--words_file', type=str, default=None,
                        help='Optional: File with words to generate (one per line). If not provided, samples from lexicon.')
    
    args = parser.parse_args()
    
    # Set up test options
    scrabblegan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrabblegan')
    os.chdir(scrabblegan_dir)
    
    # Create test options
    opt = TestOptions().parse()
    opt.name = args.model_name
    opt.epoch = args.epoch
    opt.model = 'ScrabbleGAN'
    opt.dataset_mode = 'text'
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = 1
    
    # Load model
    print(f"Loading model: {args.model_name}, epoch: {args.epoch}")
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Load words list if provided
    words_list = None
    if args.words_file and os.path.exists(args.words_file):
        with open(args.words_file, 'r', encoding='utf-8') as f:
            words_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(words_list)} words from {args.words_file}")
    
    # Generate images
    images_texts = generate_images(model, opt, args.n_images, words_list)
    
    # Change back to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Save in PyLaia format
    save_pylaia_format(images_texts, args.output_dir, args.split)
    
    print(f"\n✅ Generated {len(images_texts)} synthetic images in PyLaia format!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Images: {args.output_dir}/images/{args.split}/")
    print(f"   Text file: {args.output_dir}/{args.split}.txt")
    print(f"\nYou can now use this data to augment your PyLaia training dataset.")


if __name__ == '__main__':
    main()

