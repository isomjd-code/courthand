#!/usr/bin/env python3
"""
Generate random text using the saved Markov model.

Usage:
    python generate_text_from_markov.py [--num N] [--min-words N] [--max-words N]
"""

import json
import argparse
from pathlib import Path

try:
    import markovify
except ImportError:
    print("Error: markovify is not installed. Install it with: pip install markovify")
    raise


# Import the custom LatinText class
from build_markov_model import LatinText


def load_model(model_path: Path):
    """Load the saved Markov model from JSON file."""
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        model_json_content = json.load(f)
    
    # The file contains a JSON string (since to_json() returns a string and we saved it)
    # Parse it to get the dictionary
    if isinstance(model_json_content, str):
        model_dict = json.loads(model_json_content)
    else:
        model_dict = model_json_content
    
    # Reconstruct the model
    state_size = model_dict['state_size']
    # Chain.from_json() expects a dict (not a JSON string)
    chain = markovify.Chain.from_json(model_dict['chain'])
    parsed_sentences = model_dict['parsed_sentences']
    
    model = LatinText(chain, parsed_sentences, state_size)
    
    print("Model loaded successfully!")
    return model


def generate_text(model: LatinText, num_sentences: int = 5, 
                  min_words: int = None, max_words: int = None,
                  tries: int = 50):
    """
    Generate random text from the Markov model.
    
    Args:
        model: The loaded Markov model
        num_sentences: Number of sentences to generate
        min_words: Minimum words per sentence (None = no limit)
        max_words: Maximum words per sentence (None = no limit)
        tries: Maximum number of attempts to generate valid sentences
    """
    sentences = []
    
    for i in range(num_sentences):
        for attempt in range(tries):
            try:
                sentence = model.make_sentence()
                if sentence:
                    word_count = len(sentence.split())
                    # Check word limits if specified
                    if min_words and word_count < min_words:
                        continue
                    if max_words and word_count > max_words:
                        continue
                    sentences.append(sentence)
                    break
            except Exception as e:
                if attempt == tries - 1:
                    print(f"Warning: Failed to generate sentence {i+1}: {e}")
                continue
        else:
            print(f"Warning: Could not generate valid sentence {i+1} after {tries} attempts")
    
    return sentences


def main():
    parser = argparse.ArgumentParser(description='Generate random text from Markov model')
    parser.add_argument('--model', type=str, default='markov_model.json',
                       help='Path to the saved Markov model JSON file')
    parser.add_argument('--num', type=int, default=5,
                       help='Number of sentences to generate (default: 5)')
    parser.add_argument('--min-words', type=int, default=None,
                       help='Minimum words per sentence (default: no limit)')
    parser.add_argument('--max-words', type=int, default=None,
                       help='Maximum words per sentence (default: no limit)')
    parser.add_argument('--tries', type=int, default=50,
                       help='Maximum attempts per sentence (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
    
    # Load model
    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    model = load_model(model_path)
    
    # Generate text
    print(f"\nGenerating {args.num} sentence(s)...")
    sentences = generate_text(
        model, 
        num_sentences=args.num,
        min_words=args.min_words,
        max_words=args.max_words,
        tries=args.tries
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Generated Text:")
    print('='*60)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    print('='*60)
    print(f"\nGenerated {len(sentences)} sentence(s)")


if __name__ == "__main__":
    main()
