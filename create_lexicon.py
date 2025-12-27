#!/usr/bin/env python3
"""
Create a lexicon file from the dataset by extracting unique words.

Usage:
    python create_lexicon.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_file Datasets/Lexicon/latin_bho_words.txt
"""

import os
import argparse
from collections import Counter
import re

def detokenize_text(tokenized_text):
    """Convert tokenized text to regular text."""
    tokens = tokenized_text.split()
    result = []
    for token in tokens:
        if token == '<space>':
            result.append(' ')
        else:
            result.append(token)
    return ''.join(result)

def extract_words_from_dataset(input_dir, splits=['train', 'val', 'test']):
    """Extract all unique words from the dataset."""
    all_words = Counter()
    
    for split in splits:
        text_file = os.path.join(input_dir, f'{split}.txt')
        if not os.path.exists(text_file):
            print(f"Warning: {text_file} not found, skipping...")
            continue
        
        print(f"Processing {split}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                
                _, tokenized_text = parts
                text = detokenize_text(tokenized_text)
                
                # Split text into words (split on spaces and punctuation)
                # Keep words that contain at least one letter
                words = re.findall(r'\b\w+\b', text)
                for word in words:
                    if any(c.isalpha() for c in word):  # Only count words with at least one letter
                        all_words[word.lower()] += 1
                
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num} lines...")
    
    # Sort by frequency (most common first), then alphabetically
    sorted_words = sorted(all_words.items(), key=lambda x: (-x[1], x[0]))
    
    print(f"\nTotal unique words: {len(all_words)}")
    print(f"Total word occurrences: {sum(all_words.values())}")
    print(f"\nTop 20 most common words:")
    for word, count in sorted_words[:20]:
        print(f"  {word}: {count}")
    
    return [word for word, _ in sorted_words]

def main():
    parser = argparse.ArgumentParser(description='Create lexicon from dataset')
    parser.add_argument('--input_dir', type=str, default='bootstrap_training_data/datasets/dataset_v22',
                        help='Input directory with train.txt, val.txt, test.txt')
    parser.add_argument('--output_file', type=str, default='Datasets/Lexicon/latin_bho_words.txt',
                        help='Output lexicon file path')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Which splits to process')
    
    args = parser.parse_args()
    
    # Extract words
    words = extract_words_from_dataset(args.input_dir, args.splits)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # Write lexicon file
    print(f"\nWriting lexicon to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
    
    print(f"Lexicon created with {len(words)} unique words")

if __name__ == '__main__':
    main()

