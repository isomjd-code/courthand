#!/usr/bin/env python3
"""
Generate all n-grams of length 1-15 with at least 1400 occurrences from bootstrap training data.
"""

import os
import argparse
from collections import Counter
from typing import List, Tuple


def extract_ngrams(text: str, n: int) -> List[str]:
    """
    Extract all n-grams of length n from text.
    Counting rules:
    - Regular spaces " " don't count towards character count
    - "<space>" token counts as a single character
    - All other characters count as single character
    
    Args:
        text: Input text string
        n: N-gram length (number of content tokens, excluding regular spaces)
        
    Returns:
        List of n-gram strings
    """
    if n <= 0:
        return []
    
    # Tokenize text: treat anything in angle brackets as a single token
    # Skip regular spaces (they don't count)
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == '<':
            # Find the closing '>'
            end = text.find('>', i)
            if end != -1:
                # Extract the token including brackets (e.g., "<space>")
                token = text[i:end+1]
                tokens.append(token)
                i = end + 1
            else:
                # No closing bracket, treat '<' as a regular character
                tokens.append(text[i])
                i += 1
        elif text[i] == ' ':
            # Regular space - skip it (doesn't count)
            i += 1
        else:
            # Regular character - count it
            tokens.append(text[i])
            i += 1
    
    if len(tokens) < n:
        return []
    
    # Extract n-grams from tokens (only content tokens, no regular spaces)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ''.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def load_training_text(dataset_path: str, skip_augmented: bool = True) -> List[str]:
    """
    Load all training text lines from the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        skip_augmented: If True, skip lines from augmented images (containing "aug" in filename)
        
    Returns:
        List of text lines
    """
    # Try train.txt first (has image paths), fall back to train_text.txt
    train_file = os.path.join(dataset_path, 'train.txt')
    train_text_file = os.path.join(dataset_path, 'train_text.txt')
    
    lines = []
    
    if os.path.exists(train_file):
        # Read from train.txt (format: train/uuid.png <space> text)
        print(f"Reading from {train_file}...")
        skipped_count = 0
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
                
                image_path = parts[0]
                text = parts[1]
                
                # Skip augmented images if requested
                if skip_augmented and 'aug' in image_path.lower():
                    skipped_count += 1
                    continue
                
                lines.append(text)
        
        if skip_augmented and skipped_count > 0:
            print(f"Skipped {skipped_count} lines from augmented images")
    
    elif os.path.exists(train_text_file):
        # Fall back to train_text.txt (text only, no image paths)
        print(f"Reading from {train_text_file}...")
        print("Warning: train_text.txt doesn't contain image paths, cannot filter augmented images")
        with open(train_text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    lines.append(line)
    
    else:
        raise FileNotFoundError(
            f"Training file not found. Tried: {train_file} and {train_text_file}"
        )
    
    return lines


def generate_important_ngrams(
    dataset_path: str,
    max_n: int = 15,
    min_frequency: int = 1400,
    skip_augmented: bool = True
) -> List[Tuple[str, int, int]]:
    """
    Generate all n-grams of length 1 to max_n with at least min_frequency occurrences.
    
    Args:
        dataset_path: Path to dataset directory
        max_n: Maximum n-gram length (default: 15)
        min_frequency: Minimum frequency threshold (default: 1400)
        skip_augmented: If True, skip lines from augmented images (default: True)
        
    Returns:
        List of tuples (ngram, length, frequency), sorted by length then frequency
    """
    print(f"Loading training text from {dataset_path}...")
    lines = load_training_text(dataset_path, skip_augmented=skip_augmented)
    print(f"Loaded {len(lines)} training lines")
    
    all_ngrams = []
    
    print(f"\nExtracting all n-grams (n=1 to {max_n}) with frequency >= {min_frequency}...")
    for n in range(1, max_n + 1):
        print(f"  Processing {n}-grams...", end=' ', flush=True)
        ngram_counter = Counter()
        
        for line in lines:
            ngrams = extract_ngrams(line, n)
            ngram_counter.update(ngrams)
        
        # Filter by minimum frequency and add to list
        filtered_count = 0
        for ngram, freq in ngram_counter.items():
            if freq >= min_frequency:
                all_ngrams.append((ngram, n, freq))
                filtered_count += 1
        
        print(f"found {len(ngram_counter)} unique {n}-grams, {filtered_count} with freq >= {min_frequency}")
    
    print(f"\nTotal unique n-grams found: {len(all_ngrams)}")
    
    # Sort by: 1) length (descending), 2) frequency (descending)
    print("Sorting n-grams by length, then frequency...")
    all_ngrams.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print(f"\nGenerated {len(all_ngrams)} n-grams total (length <= {max_n}, frequency >= {min_frequency})")
    
    return all_ngrams


def main():
    parser = argparse.ArgumentParser(
        description='Generate the most important n-grams from bootstrap training data'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='bootstrap_training_data/datasets/dataset_v22',
        help='Path to dataset directory (default: bootstrap_training_data/datasets/dataset_v22)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='important_ngrams.txt',
        help='Output file path (default: important_ngrams.txt)'
    )
    parser.add_argument(
        '--max-n',
        type=int,
        default=15,
        help='Maximum n-gram length (default: 15)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=1400,
        help='Minimum frequency threshold (default: 1400)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'csv', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--include-augmented',
        action='store_true',
        help='Include lines from augmented images (default: skip augmented images)'
    )
    
    args = parser.parse_args()
    
    # Generate n-grams
    ngrams = generate_important_ngrams(
        args.dataset_path,
        max_n=args.max_n,
        min_frequency=args.min_frequency,
        skip_augmented=not args.include_augmented
    )
    
    # Check if we have any n-grams
    if not ngrams:
        print(f"\n⚠ Warning: No n-grams found with frequency >= {args.min_frequency}")
        print("The output file will not be created.")
        return
    
    # Write output
    output_path = os.path.abspath(args.output)
    print(f"\nWriting results to {output_path}...")
    
    try:
        if args.format == 'text':
            with open(args.output, 'w', encoding='utf-8') as f:
                for ngram, length, freq in ngrams:
                    # Escape special characters for readability
                    ngram_display = repr(ngram) if any(c in ngram for c in ['\n', '\t', '\r']) else ngram
                    f.write(f"{ngram_display}\t{length}\t{freq}\n")
        
        elif args.format == 'csv':
            import csv
            with open(args.output, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ngram', 'length', 'frequency'])
                for ngram, length, freq in ngrams:
                    writer.writerow([ngram, length, freq])
        
        elif args.format == 'json':
            import json
            output_data = [
                {'ngram': ngram, 'length': length, 'frequency': freq}
                for ngram, length, freq in ngrams
            ]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully wrote {len(ngrams)} n-grams to {output_path}")
        
        # Verify file was created
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            print(f"✓ File verified: {file_size} bytes")
        else:
            print(f"⚠ Warning: File {args.output} was not created!")
            
    except Exception as e:
        print(f"✗ Error writing output file: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    length_counts = {}
    for _, length, _ in ngrams:
        length_counts[length] = length_counts.get(length, 0) + 1
    
    for length in sorted(length_counts.keys(), reverse=True):
        print(f"  {length}-grams: {length_counts[length]}")
    
    total_freq = sum(freq for _, _, freq in ngrams)
    print(f"\nTotal frequency: {total_freq}")
    print(f"Average frequency: {total_freq / len(ngrams):.2f}")


if __name__ == '__main__':
    main()

