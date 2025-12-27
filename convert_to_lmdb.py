#!/usr/bin/env python3
"""
Convert PyLaia format dataset to ScrabbleGAN LMDB format.

Usage:
    python convert_to_lmdb.py --input_dir pylaia_dataset_output --output_dir Datasets/LatinBHO --split train
    python convert_to_lmdb.py --input_dir pylaia_dataset_output --output_dir Datasets/LatinBHO --split val
    python convert_to_lmdb.py --input_dir pylaia_dataset_output --output_dir Datasets/LatinBHO --split test
"""

import os
import sys
import argparse
import lmdb
import io
from PIL import Image
from tqdm import tqdm
import shutil


def detokenize_text(tokenized_text):
    """
    Convert tokenized text to regular text.
    Example: "h e l l o <space> w o r l d" -> "hello world"
    """
    tokens = tokenized_text.split()
    result = []
    for token in tokens:
        if token == '<space>':
            result.append(' ')
        else:
            result.append(token)
    return ''.join(result)


def check_image_valid(image_bin):
    """Check if image binary data is valid."""
    if image_bin is None:
        return False
    try:
        image_buf = io.BytesIO(image_bin)
        img = Image.open(image_buf)
        img.verify()
        return True
    except:
        return False


def write_cache(env, cache):
    """Write cache to LMDB."""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(k, str):
                k = k.encode('utf-8')
            if isinstance(v, str):
                v = v.encode('utf-8')
            txn.put(k, v)


def convert_to_lmdb(input_dir, output_dir, split='train', img_height=32):
    """
    Convert PyLaia format to LMDB format.
    
    Args:
        input_dir: Directory containing train.txt, val.txt, test.txt and images/ folder
        output_dir: Output directory for LMDB files
        split: Which split to convert (train, val, test)
        img_height: Target image height (default 32 for ScrabbleGAN)
    """
    # Read the text file
    text_file = os.path.join(input_dir, f'{split}.txt')
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    print(f"Reading {text_file}...")
    image_path_list = []
    label_list = []
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split into image path and tokenized text
            parts = line.split(' ', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line[:100]}")
                continue
            
            image_rel_path, tokenized_text = parts
            # Detokenize the text
            label = detokenize_text(tokenized_text)
            
            # Construct full image path
            image_path = os.path.join(input_dir, 'images', image_rel_path)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image_path_list.append(image_path)
            label_list.append(label)
    
    print(f"Found {len(image_path_list)} samples")
    
    # Create output directory
    output_path = os.path.join(output_dir, split)
    if os.path.exists(output_path):
        print(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Create LMDB environment
    env = lmdb.open(output_path, map_size=1099511627776)  # 1TB max size
    cache = {}
    cnt = 1
    n_samples = len(image_path_list)
    
    print(f"Converting {n_samples} samples to LMDB...")
    for i in tqdm(range(n_samples)):
        image_path = image_path_list[i]
        label = label_list[i]
        
        try:
            # Load and process image
            im = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to target height while maintaining aspect ratio
            width, height = im.size
            if height != img_height:
                new_width = int(width * img_height / height)
                im = im.resize((new_width, img_height), Image.LANCZOS)
            
            # Convert to TIFF format (as expected by ScrabbleGAN)
            img_byte_arr = io.BytesIO()
            im.save(img_byte_arr, format='TIFF')
            word_bin = img_byte_arr.getvalue()
            
            # Check if image is valid
            if not check_image_valid(word_bin):
                print(f"Warning: Invalid image at {image_path}")
                continue
            
            # Store in cache
            image_key = 'image-%09d' % cnt
            label_key = 'label-%09d' % cnt
            
            cache[image_key] = word_bin
            cache[label_key] = label.encode('utf-8')
            
            # Write cache every 1000 samples
            if cnt % 1000 == 0:
                write_cache(env, cache)
                cache = {}
                print(f'Written {cnt} / {n_samples}')
            
            cnt += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Write remaining cache and num-samples
    n_samples = cnt - 1
    cache['num-samples'] = str(n_samples).encode('utf-8')
    write_cache(env, cache)
    env.close()
    
    print(f'Created LMDB dataset with {n_samples} samples at {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyLaia format to ScrabbleGAN LMDB format')
    parser.add_argument('--input_dir', type=str, default='bootstrap_training_data/datasets/dataset_v22',
                        help='Input directory containing train.txt, val.txt, test.txt and images/')
    parser.add_argument('--output_dir', type=str, default='Datasets/LatinBHO',
                        help='Output directory for LMDB files')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True,
                        help='Which split to convert')
    parser.add_argument('--img_height', type=int, default=32,
                        help='Target image height (default: 32)')
    
    args = parser.parse_args()
    
    convert_to_lmdb(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split=args.split,
        img_height=args.img_height
    )


if __name__ == '__main__':
    main()

