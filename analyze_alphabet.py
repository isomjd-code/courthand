#!/usr/bin/env python3
"""Analyze the dataset to extract unique characters for the alphabet."""

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

def analyze_dataset(input_dir, splits=['train', 'val', 'test']):
    """Analyze dataset to extract all unique characters."""
    all_chars = Counter()
    
    for split in splits:
        text_file = os.path.join(input_dir, f'{split}.txt')
        if not os.path.exists(text_file):
            print(f"Warning: {text_file} not found, skipping...")
            continue
        
        print(f"Analyzing {split}...")
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
                
                for char in text:
                    all_chars[char] += 1
    
    # Sort by frequency (most common first)
    sorted_chars = sorted(all_chars.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal unique characters: {len(all_chars)}")
    print(f"\nCharacter frequency (top 50):")
    for char, count in sorted_chars[:50]:
        print(f"  '{char}': {count}")
    
    # Create alphabet string (sorted by frequency)
    alphabet = ''.join([char for char, _ in sorted_chars])
    
    print(f"\nAlphabet string ({len(alphabet)} chars):")
    print(f'alphabetLatinBHO = """{alphabet}"""')
    
    return alphabet

if __name__ == '__main__':
    input_dir = 'pylaia_dataset_output'
    alphabet = analyze_dataset(input_dir)

