#!/usr/bin/env python3
"""
Generate synthetic handwritten text images from ScrabbleGAN using names from cp40_records.db.

This script:
1. Queries the database for forenames, surnames, latinized forenames, and placenames with frequencies
2. Samples names according to their actual frequencies
3. Generates handwritten images using trained ScrabbleGAN model
4. Saves in PyLaia format

Usage:
    python generate_synthetic_names.py --n_images 10000 --output_dir synthetic_names_data
"""

import os
import sys
import argparse
import sqlite3
import random
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import uuid
from collections import Counter

# Add scrabblegan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scrabblegan'))

from options.test_options import TestOptions
from models import create_model
from util.util import prepare_z_y


def tokenize_text(text):
    """Convert text to PyLaia tokenized format."""
    tokens = []
    for char in text:
        if char == ' ':
            tokens.append('<space>')
        else:
            tokens.append(char)
    return ' '.join(tokens)


def get_names_with_frequencies(db_path='cp40_records.db'):
    """
    Query database for names with their frequencies.
    
    Returns:
        dict with keys: 'forenames', 'surnames', 'latin_forenames', 'placenames'
        Each is a list of (name, frequency) tuples
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    results = {}
    
    print("Querying database for names with frequencies...")
    
    # 1. Forenames (English) - try forenames table first, else extract from persons
    print("  - Forenames...")
    try:
        # Try forenames table (if it exists)
        cursor = conn.execute("""
            SELECT f.english_name, f.frequency as freq
            FROM forenames f
            WHERE f.frequency > 0
            ORDER BY f.frequency DESC
        """)
        results['forenames'] = [(row['english_name'], row['freq']) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Fallback: extract first word from person names
        print("    (forenames table not found, extracting from person names)")
        cursor = conn.execute("""
            SELECT 
                TRIM(SUBSTR(p.name, 1, INSTR(p.name || ' ', ' ') - 1)) as forename,
                COUNT(ep.entry_id) as freq
            FROM persons p
            JOIN entry_persons ep ON p.id = ep.person_id
            WHERE p.name IS NOT NULL AND p.name != ''
            GROUP BY forename
            HAVING freq > 0 AND forename IS NOT NULL AND forename != ''
            ORDER BY freq DESC
        """)
        results['forenames'] = [(row['forename'], row['freq']) for row in cursor.fetchall()]
    
    # 2. Surnames with frequencies
    print("  - Surnames...")
    cursor = conn.execute("""
        SELECT s.surname, COUNT(ps.person_id) as freq
        FROM surnames s
        JOIN person_surnames ps ON s.id = ps.surname_id
        GROUP BY s.id, s.surname
        HAVING freq > 0
        ORDER BY freq DESC
    """)
    results['surnames'] = [(row['surname'], row['freq']) for row in cursor.fetchall()]
    
    # 3. Latinized forenames (from forename_latin_forms if table exists)
    print("  - Latinized forenames...")
    try:
        cursor = conn.execute("""
            SELECT flf.latin_abbreviated, COUNT(DISTINCT f.id) as freq
            FROM forename_latin_forms flf
            JOIN forenames f ON flf.forename_id = f.id
            WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            GROUP BY flf.latin_abbreviated
            HAVING freq > 0
            ORDER BY freq DESC
        """)
        results['latin_forenames'] = [(row['latin_abbreviated'], row['freq']) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        print("    (forename_latin_forms table not found, skipping)")
        results['latin_forenames'] = []
    
    # 4. Placenames with frequencies
    print("  - Placenames...")
    cursor = conn.execute("""
        SELECT pl.name, COUNT(epl.entry_id) as freq
        FROM places pl
        JOIN entry_places epl ON pl.id = epl.place_id
        GROUP BY pl.id, pl.name
        HAVING freq > 0
        ORDER BY freq DESC
    """)
    results['placenames'] = [(row['name'], row['freq']) for row in cursor.fetchall()]
    
    conn.close()
    
    # Print statistics
    print(f"\nFound:")
    print(f"  Forenames: {len(results['forenames'])} unique names")
    print(f"  Surnames: {len(results['surnames'])} unique names")
    print(f"  Latin forenames: {len(results['latin_forenames'])} unique names")
    print(f"  Placenames: {len(results['placenames'])} unique names")
    
    return results


def create_weighted_name_list(name_freq_list, category_name):
    """
    Create a list of names weighted by frequency for sampling.
    
    Args:
        name_freq_list: List of (name, frequency) tuples
        category_name: Name of category for logging
    
    Returns:
        List of names (with duplicates according to frequency)
    """
    if not name_freq_list:
        return []
    
    weighted_list = []
    total_freq = sum(freq for _, freq in name_freq_list)
    
    print(f"\n{category_name}:")
    print(f"  Total frequency: {total_freq}")
    print(f"  Top 10: {[name for name, _ in name_freq_list[:10]]}")
    
    # Create weighted list (names appear frequency times)
    for name, freq in name_freq_list:
        weighted_list.extend([name] * freq)
    
    return weighted_list


def sample_names_by_frequency(name_data, n_images, proportions=None):
    """
    Sample names according to their frequencies.
    
    Args:
        name_data: Dict with name categories and frequency lists
        n_images: Number of images to generate
        proportions: Dict with proportions for each category (default: equal)
    
    Returns:
        List of (name, category) tuples
    """
    if proportions is None:
        # Default: equal proportions
        proportions = {
            'forenames': 0.25,
            'surnames': 0.25,
            'latin_forenames': 0.25,
            'placenames': 0.25
        }
    
    # Create weighted lists for each category
    weighted_forenames = create_weighted_name_list(name_data['forenames'], 'Forenames')
    weighted_surnames = create_weighted_name_list(name_data['surnames'], 'Surnames')
    weighted_latin = create_weighted_name_list(name_data['latin_forenames'], 'Latin Forenames')
    weighted_places = create_weighted_name_list(name_data['placenames'], 'Placenames')
    
    # Calculate counts for each category
    n_forenames = int(n_images * proportions['forenames'])
    n_surnames = int(n_images * proportions['surnames'])
    n_latin = int(n_images * proportions['latin_forenames'])
    n_places = int(n_images * proportions['placenames'])
    n_remaining = n_images - n_forenames - n_surnames - n_latin - n_places
    
    # Sample names
    sampled = []
    
    if weighted_forenames:
        sampled.extend([(random.choice(weighted_forenames), 'forename') for _ in range(n_forenames)])
    if weighted_surnames:
        sampled.extend([(random.choice(weighted_surnames), 'surname') for _ in range(n_surnames)])
    if weighted_latin:
        sampled.extend([(random.choice(weighted_latin), 'latin_forename') for _ in range(n_latin)])
    if weighted_places:
        sampled.extend([(random.choice(weighted_places), 'placename') for _ in range(n_places)])
    
    # Fill remaining with random category
    if n_remaining > 0:
        all_weighted = []
        if weighted_forenames:
            all_weighted.extend([(random.choice(weighted_forenames), 'forename') for _ in range(n_remaining // 4)])
        if weighted_surnames:
            all_weighted.extend([(random.choice(weighted_surnames), 'surname') for _ in range(n_remaining // 4)])
        if weighted_latin:
            all_weighted.extend([(random.choice(weighted_latin), 'latin_forename') for _ in range(n_remaining // 4)])
        if weighted_places:
            all_weighted.extend([(random.choice(weighted_places), 'placename') for _ in range(n_remaining - 3*(n_remaining//4))])
        sampled.extend(all_weighted)
    
    # Shuffle
    random.shuffle(sampled)
    
    return sampled


def generate_images_from_names(model, opt, name_list):
    """
    Generate synthetic images from list of names.
    
    Args:
        model: Trained ScrabbleGAN model
        opt: Test options
        name_list: List of (name, category) tuples
    
    Returns:
        List of (image, text, category) tuples
    """
    model.eval()
    results = []
    
    # Prepare noise vectors
    model.z, model.label_fake = prepare_z_y(
        opt.batch_size, opt.dim_z, len(model.lex),
        device=model.device, fp16=opt.G_fp16
    )
    
    print(f"\nGenerating {len(name_list)} images...")
    
    with torch.no_grad():
        for name, category in tqdm(name_list):
            # Generate image for this specific name
            words = [name]
            
            try:
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
                
                results.append((img, name, category))
            except Exception as e:
                print(f"Error generating image for '{name}': {e}")
                continue
    
    return results


def save_pylaia_format(images_texts_categories, output_dir, split='train'):
    """Save images and text in PyLaia format with category metadata."""
    images_dir = os.path.join(output_dir, 'images', split)
    os.makedirs(images_dir, exist_ok=True)
    
    # Create text file
    text_file = os.path.join(output_dir, f'{split}.txt')
    
    # Create metadata file
    metadata_file = os.path.join(output_dir, f'{split}_metadata.json')
    import json
    metadata = []
    
    print(f"Saving {len(images_texts_categories)} images to {images_dir}...")
    
    with open(text_file, 'w', encoding='utf-8') as f:
        for idx, (img, text, category) in enumerate(tqdm(images_texts_categories)):
            # Generate unique filename
            filename = f"synth_{category}_{uuid.uuid4().hex[:8]}.png"
            image_path = os.path.join(images_dir, filename)
            
            # Save image
            img.save(image_path)
            
            # Tokenize text
            tokenized = tokenize_text(text)
            
            # Write to text file (PyLaia format)
            relative_path = f"{split}/{filename}"
            f.write(f"{relative_path} {tokenized}\n")
            
            # Save metadata
            metadata.append({
                'filename': filename,
                'text': text,
                'category': category,
                'index': idx
            })
    
    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(images_texts_categories)} images and text file: {text_file}")
    print(f"Saved metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic name images from ScrabbleGAN using cp40_records.db'
    )
    parser.add_argument('--model_name', type=str, 
                        default='latin_bho_LatinBHOtrH32_GANres32_bs8',
                        help='Name of the trained ScrabbleGAN model')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Epoch to load (default: latest)')
    parser.add_argument('--n_images', type=int, required=True,
                        help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='synthetic_names_data',
                        help='Output directory for PyLaia format data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split name (default: train)')
    parser.add_argument('--db_path', type=str, default='cp40_records.db',
                        help='Path to cp40_records.db database')
    parser.add_argument('--forename_prop', type=float, default=0.25,
                        help='Proportion of forenames (default: 0.25)')
    parser.add_argument('--surname_prop', type=float, default=0.25,
                        help='Proportion of surnames (default: 0.25)')
    parser.add_argument('--latin_prop', type=float, default=0.25,
                        help='Proportion of latinized forenames (default: 0.25)')
    parser.add_argument('--placename_prop', type=float, default=0.25,
                        help='Proportion of placenames (default: 0.25)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate proportions
    total_prop = args.forename_prop + args.surname_prop + args.latin_prop + args.placename_prop
    if abs(total_prop - 1.0) > 0.01:
        print(f"Warning: Proportions sum to {total_prop}, normalizing...")
        args.forename_prop /= total_prop
        args.surname_prop /= total_prop
        args.latin_prop /= total_prop
        args.placename_prop /= total_prop
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Query database for names
    if not os.path.exists(args.db_path):
        print(f"Error: Database not found: {args.db_path}")
        sys.exit(1)
    
    name_data = get_names_with_frequencies(args.db_path)
    
    # Sample names according to frequencies
    proportions = {
        'forenames': args.forename_prop,
        'surnames': args.surname_prop,
        'latin_forenames': args.latin_prop,
        'placenames': args.placename_prop
    }
    
    name_list = sample_names_by_frequency(name_data, args.n_images, proportions)
    
    print(f"\nSampled {len(name_list)} names:")
    category_counts = Counter(cat for _, cat in name_list)
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
    # Load ScrabbleGAN model
    scrabblegan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scrabblegan')
    original_dir = os.getcwd()
    os.chdir(scrabblegan_dir)
    
    print(f"\nLoading ScrabbleGAN model: {args.model_name}, epoch: {args.epoch}")
    opt = TestOptions().parse()
    opt.name = args.model_name
    opt.epoch = args.epoch
    opt.model = 'ScrabbleGAN'
    opt.dataset_mode = 'text'
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = 1
    
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    # Generate images
    images_texts_categories = generate_images_from_names(model, opt, name_list)
    
    # Change back to project root
    os.chdir(original_dir)
    
    # Save in PyLaia format
    save_pylaia_format(images_texts_categories, args.output_dir, args.split)
    
    print(f"\n✅ Generated {len(images_texts_categories)} synthetic name images!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Images: {args.output_dir}/images/{args.split}/")
    print(f"   Text file: {args.output_dir}/{args.split}.txt")
    print(f"   Metadata: {args.output_dir}/{args.split}_metadata.json")


if __name__ == '__main__':
    main()

