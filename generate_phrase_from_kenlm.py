#!/usr/bin/env python3
"""
Generate a phrase starting with a specified word using a KenLM model.
Supports both deterministic (most likely) and sampling-based generation.

Usage:
    # Sample from distribution (default)
    python generate_phrase_from_kenlm.py "et"
    
    # Most likely phrase
    python generate_phrase_from_kenlm.py "et" --method vocab
    
    # Sampling with temperature
    python generate_phrase_from_kenlm.py "et" --temperature 1.5
    
    # Top-k sampling (only consider top 50 words)
    python generate_phrase_from_kenlm.py "et" --top_k 50
    
    # Nucleus sampling (top-p)
    python generate_phrase_from_kenlm.py "et" --top_p 0.9
"""

import sys
import os
import argparse
import random
import math
from pathlib import Path

try:
    import kenlm
except ImportError:
    print("Error: kenlm package not found.")
    print("Please install it with: pip install https://github.com/kpu/kenlm/archive/master.zip")
    print("Or: pip install git+https://github.com/kpu/kenlm.git")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_kenlm_model(arpa_path: str):
    """Load a KenLM model from an ARPA file."""
    if not os.path.exists(arpa_path):
        raise FileNotFoundError(f"KenLM model file not found: {arpa_path}")
    
    print(f"Loading KenLM model from: {arpa_path}")
    model = kenlm.Model(arpa_path)
    print(f"Model loaded successfully (order: {model.order})")
    return model


def get_next_word_probabilities(model, context_words):
    """
    Get probabilities for all possible next words given a context.
    
    Args:
        model: KenLM model
        context_words: List of previous words (up to order-1 words)
    
    Returns:
        Dictionary mapping words to their log probabilities
    """
    # Build context string
    context = " ".join(context_words) if context_words else ""
    
    # KenLM doesn't provide a direct way to enumerate all possible next words
    # We'll need to use a different approach - score candidate words
    # For now, we'll use beam search with common words or vocabulary
    
    # Get the full score for the context
    if context:
        full_score = model.score(context, bos=False, eos=False)
    else:
        full_score = 0.0
    
    return full_score


def generate_phrase_greedy(model, starting_word: str, num_words: int = 10, beam_size: int = 100):
    """
    Generate the most likely phrase using greedy decoding with beam search.
    
    Args:
        model: KenLM model
        starting_word: Word to start the phrase with
        num_words: Total number of words to generate (including starting word)
        beam_size: Number of candidates to keep at each step
    
    Returns:
        List of words forming the most likely phrase
    """
    phrase = [starting_word]
    
    # For a 3-gram model, we need at most 2 words of context
    for i in range(1, num_words):
        # Get context (last 2 words for 3-gram model, but model.order-1)
        context_size = min(model.order - 1, len(phrase))
        context = phrase[-context_size:] if context_size > 0 else []
        
        # Build context string
        context_str = " ".join(context) if context else ""
        
        # We'll use a beam search approach
        # Since we can't enumerate all words, we'll try common candidates
        # or use a vocabulary if available
        
        # For now, let's use a simpler approach: try to find the best continuation
        # by scoring candidate sequences
        
        best_word = None
        best_score = float('-inf')
        
        # Try common words and words from the model's vocabulary
        # We'll use a heuristic: try words that appear in the model
        # Since we can't directly query the vocabulary, we'll use a different strategy
        
        # Strategy: Use the model's score method to evaluate continuations
        # We'll need to try candidate words. Let's use a beam search with
        # a set of candidate words we can try
        
        # For a more practical approach, let's use the fact that KenLM can score
        # full sentences. We'll use a greedy approach where we try to extend
        # the phrase one word at a time by testing candidate words.
        
        # Since we don't have direct access to the vocabulary, we'll use
        # a different method: generate using the model's internal mechanisms
        
        # Actually, let's use a simpler greedy approach:
        # Score the current phrase and try to find the best next word
        # by testing a set of candidate words
        
        # For now, let's implement a basic version that uses the model's
        # scoring to find the best continuation
        
        # We'll need to try candidate words. Let's use a set of common words
        # or we can try to extract vocabulary from the model if possible
        
        # Since KenLM Python bindings don't provide vocabulary access easily,
        # we'll use a different approach: generate by finding words that
        # maximize the conditional probability
        
        # Let's try a practical approach: use the model to score sequences
        # and find the word that gives the highest score
        
        # For a 3-gram model, P(w3 | w1, w2) = score("w1 w2 w3") - score("w1 w2")
        current_phrase_str = " ".join(phrase)
        current_score = model.score(current_phrase_str, bos=True, eos=False)
        
        # We need candidate words to try. Let's use a heuristic approach:
        # Try common words that might appear in the corpus
        
        # Since we can't enumerate all words, let's use a different strategy:
        # Use the model's perplexity or try to sample from the model
        
        # Actually, for a more practical solution, let's use beam search
        # with a limited vocabulary of candidate words
        
        # Let's implement a version that tries common Latin/English words
        # This is a limitation, but without vocabulary access, it's the best we can do
        
        # For now, let's use a simple greedy approach that tries to extend
        # the phrase by testing a set of candidate words
        
        # We'll use a set of common words as candidates
        # In a real implementation, you'd want to extract the vocabulary from the model
        
        # For demonstration, let's try common words
        candidate_words = [
            "et", "in", "ad", "de", "cum", "per", "pro", "ex", "ab", "sub",
            "super", "inter", "intra", "extra", "contra", "post", "ante",
            "sine", "apud", "propter", "ob", "prae", "trans", "ultra",
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "shall"
        ]
        
        # Try each candidate word
        for candidate in candidate_words:
            test_phrase = phrase + [candidate]
            test_str = " ".join(test_phrase)
            test_score = model.score(test_str, bos=True, eos=False)
            
            # Calculate conditional probability: P(candidate | context)
            # For 3-gram: P(w3|w1,w2) = score("w1 w2 w3") - score("w1 w2")
            conditional_score = test_score - current_score
            
            if conditional_score > best_score:
                best_score = conditional_score
                best_word = candidate
        
        # If no candidate worked well, try to use the model's internal prediction
        # or use a fallback
        
        if best_word is None:
            # Fallback: use a common word or try to extract from model
            # For now, let's just use "et" as a fallback
            best_word = "et"
        
        phrase.append(best_word)
    
    return phrase


def generate_phrase_beam_search(model, starting_word: str, num_words: int = 10, beam_size: int = 10):
    """
    Generate the most likely phrase using beam search.
    
    This is a more sophisticated approach that maintains multiple hypotheses.
    """
    # Initialize beam with starting word
    beam = [([starting_word], model.score(starting_word, bos=True, eos=False))]
    
    for i in range(1, num_words):
        new_beam = []
        
        for phrase, score in beam:
            # Get context
            context_size = min(model.order - 1, len(phrase))
            context = phrase[-context_size:] if context_size > 0 else []
            current_str = " ".join(phrase)
            current_score = model.score(current_str, bos=True, eos=False)
            
            # Try candidate words
            candidate_words = [
                "et", "in", "ad", "de", "cum", "per", "pro", "ex", "ab", "sub",
                "super", "inter", "intra", "extra", "contra", "post", "ante",
                "sine", "apud", "propter", "ob", "prae", "trans", "ultra",
                "the", "a", "an", "is", "was", "are", "were", "be", "been"
            ]
            
            for candidate in candidate_words:
                test_phrase = phrase + [candidate]
                test_str = " ".join(test_phrase)
                test_score = model.score(test_str, bos=True, eos=False)
                conditional_score = test_score - current_score
                new_score = score + conditional_score
                
                new_beam.append((test_phrase, new_score))
        
        # Keep top beam_size candidates
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]
    
    # Return the best phrase
    if beam:
        return beam[0][0]
    else:
        return [starting_word]


def extract_vocabulary_from_arpa(arpa_path: str, max_words: int = 1000):
    """
    Extract vocabulary from ARPA file by reading unigram entries.
    
    Args:
        arpa_path: Path to ARPA file
        max_words: Maximum number of words to extract
    
    Returns:
        List of words from the vocabulary
    """
    vocabulary = []
    in_unigrams = False
    
    try:
        with open(arpa_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                if line == "\\1-grams:":
                    in_unigrams = True
                    continue
                elif line.startswith("\\") and "grams:" in line:
                    in_unigrams = False
                    if len(vocabulary) >= max_words:
                        break
                    continue
                
                if in_unigrams and line:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[1]
                        # Skip special tokens
                        if word not in ["<s>", "</s>", "<unk>", "<UNK>"]:
                            vocabulary.append(word)
                            if len(vocabulary) >= max_words:
                                break
    except Exception as e:
        print(f"Warning: Could not extract vocabulary from ARPA file: {e}")
        return []
    
    return vocabulary


def generate_phrase_with_vocab(model, starting_word: str, vocabulary: list, num_words: int = 10):
    """
    Generate phrase using vocabulary from the model (greedy - most likely).
    """
    phrase = [starting_word]
    
    for i in range(1, num_words):
        context_size = min(model.order - 1, len(phrase))
        context = phrase[-context_size:] if context_size > 0 else []
        current_str = " ".join(phrase)
        current_score = model.score(current_str, bos=True, eos=False)
        
        best_word = None
        best_score = float('-inf')
        
        # Try all words in vocabulary
        for candidate in vocabulary:
            if candidate == starting_word and i == 1:
                continue  # Skip starting word at position 1
            
            test_phrase = phrase + [candidate]
            test_str = " ".join(test_phrase)
            test_score = model.score(test_str, bos=True, eos=False)
            conditional_score = test_score - current_score
            
            if conditional_score > best_score:
                best_score = conditional_score
                best_word = candidate
        
        if best_word is None:
            # Fallback
            best_word = "et" if "et" in vocabulary else vocabulary[0] if vocabulary else "et"
        
        phrase.append(best_word)
    
    return phrase


def sample_from_distribution(candidates, log_probs, temperature=1.0, top_k=None, top_p=None):
    """
    Sample a word from the probability distribution.
    
    Args:
        candidates: List of candidate words
        log_probs: List of log probabilities for each candidate
        temperature: Temperature for sampling (higher = more random, lower = more focused)
        top_k: If set, only consider top k candidates by probability
        top_p: If set, use nucleus sampling (only consider candidates with cumulative prob <= top_p)
    
    Returns:
        Sampled word
    """
    if not candidates or not log_probs:
        return None
    
    # Convert log probabilities to probabilities with temperature
    if temperature != 1.0:
        scaled_log_probs = [lp / temperature for lp in log_probs]
    else:
        scaled_log_probs = log_probs
    
    # Convert to probabilities (exp of log probs)
    # Subtract max for numerical stability
    max_log_prob = max(scaled_log_probs)
    probs = [math.exp(lp - max_log_prob) for lp in scaled_log_probs]
    
    # Apply top-k filtering
    if top_k is not None and top_k < len(candidates):
        # Get indices sorted by probability
        indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_k_indices = set(indices[:top_k])
        # Zero out probabilities for words not in top-k
        probs = [p if i in top_k_indices else 0.0 for i, p in enumerate(probs)]
    
    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        # Sort by probability
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumulative_prob = 0.0
        keep_indices = set()
        
        for idx in sorted_indices:
            keep_indices.add(idx)
            cumulative_prob += probs[idx]
            if cumulative_prob >= top_p:
                break
        
        # Zero out probabilities for words not in nucleus
        probs = [p if i in keep_indices else 0.0 for i, p in enumerate(probs)]
    
    # Normalize probabilities
    total_prob = sum(probs)
    if total_prob == 0:
        # Fallback: uniform distribution
        probs = [1.0 / len(candidates)] * len(candidates)
    else:
        probs = [p / total_prob for p in probs]
    
    # Sample from distribution
    if HAS_NUMPY:
        sampled_idx = np.random.choice(len(candidates), p=probs)
    else:
        # Use random.choices if numpy not available
        sampled_idx = random.choices(range(len(candidates)), weights=probs)[0]
    
    return candidates[sampled_idx]


def generate_phrase_sampling(model, starting_word: str, vocabulary: list, num_words: int = 10, 
                            temperature: float = 1.0, top_k: int = None, top_p: float = None):
    """
    Generate phrase by sampling from the probability distribution at each step.
    
    Args:
        model: KenLM model
        starting_word: Word to start the phrase with
        vocabulary: List of candidate words
        num_words: Total number of words to generate
        temperature: Sampling temperature (higher = more diverse, lower = more focused)
        top_k: If set, only consider top k most likely words at each step
        top_p: If set, use nucleus sampling (cumulative probability threshold)
    
    Returns:
        List of words forming the generated phrase
    """
    phrase = [starting_word]
    
    for i in range(1, num_words):
        current_str = " ".join(phrase)
        current_score = model.score(current_str, bos=True, eos=False)
        
        candidates = []
        log_probs = []
        
        # Calculate log probabilities for all vocabulary words
        for candidate in vocabulary:
            # Skip starting word at position 1 to avoid immediate repetition
            if candidate == starting_word and i == 1:
                continue
            
            test_phrase = phrase + [candidate]
            test_str = " ".join(test_phrase)
            test_score = model.score(test_str, bos=True, eos=False)
            
            # Conditional log probability: P(candidate | context)
            conditional_log_prob = test_score - current_score
            
            candidates.append(candidate)
            log_probs.append(conditional_log_prob)
        
        if not candidates:
            # Fallback
            best_word = "et" if "et" in vocabulary else vocabulary[0] if vocabulary else "et"
        else:
            # Sample from the distribution
            best_word = sample_from_distribution(candidates, log_probs, temperature, top_k, top_p)
            if best_word is None:
                # Fallback if sampling failed
                best_word = candidates[0]
        
        phrase.append(best_word)
    
    return phrase


def main():
    parser = argparse.ArgumentParser(
        description="Generate the most likely 10-word phrase starting with a specified word using KenLM"
    )
    parser.add_argument(
        "starting_word",
        type=str,
        help="Word to start the phrase with"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kenlm_model/kenlm_model_3gram.arpa",
        help="Path to KenLM ARPA model file (default: kenlm_model/kenlm_model_3gram.arpa)"
    )
    parser.add_argument(
        "--num_words",
        type=int,
        default=10,
        help="Number of words to generate (default: 10)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["greedy", "beam", "vocab", "sample"],
        default="sample",
        help="Generation method: greedy, beam, vocab (most likely), or sample (default: sample)"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Beam size for beam search (default: 10)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only for 'sample' method). Higher = more random, lower = more focused. (default: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling: only consider top k most likely words at each step (default: None, use all words)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling: cumulative probability threshold (default: None, use all words)"
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_kenlm_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generate phrase
    print(f"\nGenerating {args.num_words}-word phrase starting with '{args.starting_word}'...")
    print(f"Method: {args.method}\n")
    
    if args.method in ["vocab", "sample"]:
        # Extract vocabulary from ARPA file
        print("Extracting vocabulary from ARPA file...")
        vocabulary = extract_vocabulary_from_arpa(args.model, max_words=5000)
        print(f"Extracted {len(vocabulary)} words from vocabulary")
        
        if not vocabulary:
            print("Warning: Could not extract vocabulary, falling back to greedy method")
            phrase = generate_phrase_greedy(model, args.starting_word, args.num_words)
        else:
            if args.method == "sample":
                print(f"Sampling with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
                phrase = generate_phrase_sampling(
                    model, args.starting_word, vocabulary, args.num_words,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
                )
            else:  # vocab
                phrase = generate_phrase_with_vocab(model, args.starting_word, vocabulary, args.num_words)
    elif args.method == "beam":
        phrase = generate_phrase_beam_search(model, args.starting_word, args.num_words, args.beam_size)
    else:  # greedy
        phrase = generate_phrase_greedy(model, args.starting_word, args.num_words)
    
    # Output result
    phrase_str = " ".join(phrase)
    score = model.score(phrase_str, bos=True, eos=True)
    perplexity = 10 ** (-score / len(phrase))
    
    print(f"\nGenerated phrase ({len(phrase)} words):")
    print(f"  {phrase_str}")
    print(f"\nScore: {score:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"\nWords: {phrase}")


if __name__ == "__main__":
    main()

