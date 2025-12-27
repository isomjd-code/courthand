#!/usr/bin/env python3
"""Extract all unique characters from the dataset to create a complete alphabet."""

import os
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

def extract_alphabet(input_dir='bootstrap_training_data/datasets/dataset_v22', splits=['train', 'val', 'test']):
    """Extract all unique characters from the dataset."""
    all_chars = Counter()
    
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
                
                for char in text:
                    all_chars[char] += 1
                
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num} lines...")
    
    # Sort by frequency (most common first), then by character code
    sorted_chars = sorted(all_chars.items(), key=lambda x: (-x[1], ord(x[0])))
    
    print(f"\nTotal unique characters: {len(all_chars)}")
    print(f"\nCharacter frequency:")
    for char, count in sorted_chars:
        char_repr = repr(char) if char in [' ', '\n', '\t'] else char
        print(f"  '{char_repr}': {count}")
    
    # Create alphabet string (sorted by frequency, most common first)
    alphabet = ''.join([char for char, _ in sorted_chars])
    
    print(f"\nAlphabet string ({len(alphabet)} chars):")
    print(f'alphabetLatinBHO = """{alphabet}"""')
    
    # Also create a version sorted alphabetically for readability
    alphabet_sorted = ''.join(sorted(set(all_chars.keys()), key=lambda x: (x.isdigit(), x.isupper(), x.lower(), x)))
    print(f"\nAlphabet sorted ({len(alphabet_sorted)} chars):")
    print(f'alphabetLatinBHO = """{alphabet_sorted}"""')
    
    return alphabet, alphabet_sorted

if __name__ == '__main__':
    alphabet, alphabet_sorted = extract_alphabet()
    print(f"\nRecommended: Use the frequency-sorted version for better performance")

