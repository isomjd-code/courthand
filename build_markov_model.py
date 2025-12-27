#!/usr/bin/env python3
"""
Extract merged_text from all step2a_merged.json files and build a Markov model.

This script:
1. Recursively finds all step2a_merged.json files in cp40_processing/output
2. Extracts the "merged_text" field from each file
3. Builds a Markov model using markovify based on word tokens (space-separated)
4. Saves the model to disk
"""

import json
import os
from pathlib import Path

try:
    import markovify
except ImportError:
    print("Error: markovify is not installed. Install it with: pip install markovify")
    raise


def find_all_step2a_files(base_dir: Path) -> list[Path]:
    """Find all step2a_merged.json files recursively."""
    files = list(base_dir.rglob("step2a_merged.json"))
    print(f"Found {len(files)} step2a_merged.json files")
    return files


def extract_merged_texts(file_paths: list[Path], min_words: int = 2) -> list[str]:
    """Extract merged_text from all JSON files, filtering out texts that are too short."""
    texts = []
    skipped = 0
    filtered = 0
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'merged_text' in data and data['merged_text']:
                    text = data['merged_text'].strip()
                    # Filter out texts that are too short
                    word_count = len(text.split())
                    if word_count >= min_words:
                        texts.append(text)
                    else:
                        filtered += 1
                else:
                    skipped += 1
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            skipped += 1
    
    print(f"Extracted {len(texts)} merged texts ({skipped} files skipped, {filtered} filtered as too short)")
    return texts


class LatinText(markovify.Text):
    """
    Custom markovify.Text class for continuous Latin text without punctuation.
    Wraps a Chain built from word lists.
    """
    def __init__(self, chain, parsed_sentences, state_size):
        """Initialize with a pre-built chain and parsed sentences."""
        # Don't call super().__init__ since we're bypassing normal initialization
        self.chain = chain
        self.parsed_sentences = parsed_sentences
        self.state_size = state_size
        self.retain_original = True  # Required for to_dict() method
    
    def word_join(self, words):
        """Join words back into a sentence."""
        return ' '.join(words)
    
    def make_sentence(self, **kwargs):
        """Generate a sentence using the chain."""
        return self.word_join(self.chain.walk(**kwargs))


def build_markov_model(texts: list[str]) -> markovify.Text:
    """
    Build a Markov model from texts using word-level tokens.
    
    Splits each text into words and builds a Chain directly, which works
    perfectly for continuous Latin text without punctuation.
    """
    if not texts:
        raise ValueError("No texts provided to build model")
    
    # Filter out empty texts and split into word lists (each text is a sentence)
    sentences = []
    for text in texts:
        text = text.strip()
        if text:
            words = text.split()
            if words:  # Only add non-empty sentences
                sentences.append(words)
    
    if not sentences:
        raise ValueError("No valid sentences created from texts")
    
    total_words = sum(len(s) for s in sentences)
    print(f"Building Markov model from {total_words} words in {len(sentences)} sentences...")
    
    # Try state_size=2 first (bigram model), fall back to state_size=1 if needed
    for state_size in [2, 1]:
        try:
            # Build chain directly from word lists
            chain = markovify.Chain(sentences, state_size=state_size)
            # Wrap in our custom Text class
            model = LatinText(chain, sentences, state_size)
            print(f"Markov model built successfully with state_size={state_size}")
            return model
        except Exception as e:
            if state_size == 1:
                # If even state_size=1 fails, raise the error with more details
                raise RuntimeError(f"Failed to build Markov model even with state_size=1: {e}")
            print(f"Warning: Failed with state_size={state_size}, trying state_size={state_size-1}: {e}")
    
    # This should never be reached, but just in case
    raise RuntimeError("Failed to build Markov model")


def main():
    # Base directory containing all output subdirectories
    base_dir = Path(__file__).parent / "cp40_processing" / "output"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all step2a_merged.json files
    print(f"Searching for step2a_merged.json files in {base_dir}...")
    file_paths = find_all_step2a_files(base_dir)
    
    if not file_paths:
        print("No step2a_merged.json files found")
        return
    
    # Extract merged_text from all files (filter texts with less than state_size words)
    # We'll filter for at least 2 words to ensure state_size=2 works, but allow 1 word as fallback
    print("Extracting merged_text from files...")
    texts = extract_merged_texts(file_paths, min_words=2)
    
    if not texts:
        print("No merged_text found in any files")
        return
    
    # Build Markov model
    print("Building Markov model...")
    model = build_markov_model(texts)
    
    # Save the model
    output_path = Path(__file__).parent / "markov_model.json"
    print(f"Saving model to {output_path}...")
    
    model_json = model.to_json()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved successfully to {output_path}")
    
    # Print some statistics
    print(f"\nModel statistics:")
    print(f"  - Number of files processed: {len(file_paths)}")
    print(f"  - Number of texts extracted: {len(texts)}")
    print(f"  - Total characters: {sum(len(t) for t in texts)}")
    
    # Test generating a sentence
    try:
        test_sentence = model.make_sentence()
        if test_sentence:
            print(f"\nTest generated sentence:")
            print(f"  {test_sentence}")
    except Exception as e:
        print(f"\nNote: Could not generate test sentence: {e}")


if __name__ == "__main__":
    main()
