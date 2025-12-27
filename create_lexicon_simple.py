#!/usr/bin/env python3
"""
Quick script to create a lexicon from the dataset.
"""

import os
import re
from collections import Counter

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

# Read dataset
input_dir = 'bootstrap_training_data/datasets/dataset_v22'
all_words = Counter()

for split in ['train', 'val', 'test']:
    text_file = os.path.join(input_dir, f'{split}.txt')
    if not os.path.exists(text_file):
        continue
    
    print(f"Processing {split}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            _, tokenized_text = parts
            text = detokenize_text(tokenized_text)
            
            # Extract words (split on spaces and common punctuation)
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if any(c.isalpha() for c in word) and len(word) < 20:
                    all_words[word.lower()] += 1

# Get unique words sorted by frequency
unique_words = sorted(set(all_words.keys()))

print(f"\nFound {len(unique_words)} unique words")
print(f"Top 20: {unique_words[:20]}")

# Write lexicon file
output_file = 'Datasets/Lexicon/latin_bho_words.txt'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    for word in unique_words:
        f.write(word + '\n')

print(f"Lexicon written to {output_file} with {len(unique_words)} words")

