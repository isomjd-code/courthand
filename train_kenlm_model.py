#!/usr/bin/env python3
"""
Extract merged_text from all step2a_merged.json files and train a KenLM language model.

This script:
1. Recursively finds all step2a_merged.json files in cp40_processing/output
2. Extracts the "merged_text" field from each file
3. Formats the text for KenLM training (one sentence per line)
4. Trains a KenLM model using the KenLM tools
5. Saves the model in both ARPA and binary formats for use with Pylaia

KenLM can be installed via:
    - pip: pip install git+https://github.com/kpu/kenlm.git
      (requires build tools: g++, cmake, boost)
    - Ubuntu/Debian: sudo apt-get install libkenlm0 libkenlm-dev kenlm
    - Or build from source: https://github.com/kpu/kenlm
    
Note: The Python bindings from pip are mainly for loading/using models.
      For training, you still need the command-line tools (lmplz, build_binary).
      If pip install doesn't provide these, install via apt-get or build from source.
"""

import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import argparse
from typing import Optional, Tuple, List
import random
import sys
import multiprocessing as mp
from functools import partial
import time

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import functions from generate_text_with_replacements
try:
    from generate_text_with_replacements import load_model, DatabaseReplacer, generate_texts_batch
    GENERATE_TEXT_AVAILABLE = True
except ImportError:
    GENERATE_TEXT_AVAILABLE = False
    print("Warning: generate_text_with_replacements not available. Synthetic text generation disabled.")

# Try to import numpy for seeding
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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


def _worker_init(model_path: Path, db_path: Path, seed_base: int):
    """Initialize worker process with its own model and replacer."""
    global _worker_model, _worker_replacer
    
    # Set unique seed per worker
    worker_seed = seed_base + mp.current_process()._identity[0] if mp.current_process()._identity else seed_base
    random.seed(worker_seed)
    if HAS_NUMPY:
        np.random.seed(worker_seed)
    
    _worker_model = load_model(model_path)
    _worker_replacer = DatabaseReplacer(db_path)
    _worker_replacer._ensure_initialized()


def _worker_generate_batch(args: Tuple[int, int, int]) -> List[str]:
    """Worker function to generate a batch of texts."""
    batch_size, tries, min_words = args
    global _worker_model, _worker_replacer
    
    results = []
    for _ in range(batch_size):
        sentence = None
        for attempt in range(tries):
            try:
                sentence = _worker_model.make_sentence()
                if sentence:
                    break
            except Exception:
                continue
        
        if not sentence:
            continue
        
        try:
            replaced_text, _ = _worker_replacer.replace_words(sentence)
            if len(replaced_text.split()) >= min_words:
                results.append(replaced_text)
        except Exception:
            continue
    
    return results


def generate_synthetic_texts(
    num_texts: int,
    model_path: Path,
    db_path: Path,
    min_words: int = 2,
    tries: int = 50,
    num_workers: int = 0
) -> List[str]:
    """
    Generate synthetic texts using Markov model and database replacements.
    
    Args:
        num_texts: Number of synthetic texts to generate
        model_path: Path to Markov model JSON file
        db_path: Path to database file
        min_words: Minimum words per text
        tries: Maximum attempts per text generation
        num_workers: Number of parallel workers (0 = auto, 1 = single-threaded)
    
    Returns:
        List of generated texts
    """
    if not GENERATE_TEXT_AVAILABLE:
        print("Warning: Cannot generate synthetic texts - generate_text_with_replacements not available")
        return []
    
    if not model_path.exists():
        print(f"Warning: Markov model not found at {model_path}. Skipping synthetic text generation.")
        return []
    
    if not db_path.exists():
        print(f"Warning: Database not found at {db_path}. Skipping synthetic text generation.")
        return []
    
    print(f"\nGenerating {num_texts:,} synthetic texts...")
    start_time = time.time()
    
    # Determine number of workers
    if num_workers == 0:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # For small counts or single worker, use optimized single-threaded version
    if num_workers == 1 or num_texts < 100:
        return _generate_synthetic_texts_single(num_texts, model_path, db_path, min_words, tries)
    
    # Multiprocessing for large batches
    print(f"  Using {num_workers} parallel workers...")
    
    # Split work into batches
    batch_size = max(10, num_texts // (num_workers * 4))  # 4 batches per worker for load balancing
    num_batches = (num_texts + batch_size - 1) // batch_size
    
    # Create batch arguments
    batches = []
    remaining = num_texts
    for _ in range(num_batches):
        this_batch = min(batch_size, remaining)
        batches.append((this_batch, tries, min_words))
        remaining -= this_batch
    
    synthetic_texts = []
    seed_base = random.randint(0, 2**31)
    
    try:
        with mp.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(model_path, db_path, seed_base)
        ) as pool:
            # Process batches with progress bar
            results = list(tqdm(
                pool.imap_unordered(_worker_generate_batch, batches),
                total=len(batches),
                desc="  Generating synthetic texts",
                unit="batch"
            ))
            
            for batch_result in results:
                synthetic_texts.extend(batch_result)
    
    except Exception as e:
        print(f"  Warning: Multiprocessing failed ({e}), falling back to single-threaded...")
        return _generate_synthetic_texts_single(num_texts, model_path, db_path, min_words, tries)
    
    elapsed = time.time() - start_time
    rate = len(synthetic_texts) / elapsed if elapsed > 0 else 0
    failed = num_texts - len(synthetic_texts)
    print(f"Generated {len(synthetic_texts):,} synthetic texts in {elapsed:.1f}s ({rate:.1f} texts/sec, {failed} failed)")
    
    return synthetic_texts


def _generate_synthetic_texts_single(
    num_texts: int,
    model_path: Path,
    db_path: Path,
    min_words: int = 2,
    tries: int = 50
) -> List[str]:
    """Single-threaded optimized text generation."""
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Warning: Failed to load Markov model: {e}. Skipping synthetic text generation.")
        return []
    
    # Initialize database replacer
    try:
        replacer = DatabaseReplacer(db_path)
    except Exception as e:
        print(f"Warning: Failed to initialize database replacer: {e}. Skipping synthetic text generation.")
        return []
    
    start_time = time.time()
    
    try:
        # Use the optimized batch generation function
        synthetic_texts = generate_texts_batch(
            model, replacer, num_texts, tries, min_words, show_progress=True
        )
        
        elapsed = time.time() - start_time
        rate = len(synthetic_texts) / elapsed if elapsed > 0 else 0
        failed = num_texts - len(synthetic_texts)
        print(f"Generated {len(synthetic_texts):,} synthetic texts in {elapsed:.1f}s ({rate:.1f} texts/sec, {failed} failed)")
        
    finally:
        replacer.close()
    
    return synthetic_texts


def prepare_training_text(texts: list[str], output_file: Path) -> None:
    """
    Prepare training text file for KenLM.
    
    KenLM expects one sentence per line. Each text is treated as a sentence.
    """
    print(f"Preparing training text file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            # Write each text as a separate line
            # KenLM expects sentences separated by newlines
            f.write(text + '\n')
    
    # Count lines and words
    line_count = len(texts)
    word_count = sum(len(text.split()) for text in texts)
    char_count = sum(len(text) for text in texts)
    
    print(f"Training text prepared:")
    print(f"  - Lines (sentences): {line_count:,}")
    print(f"  - Total words: {word_count:,}")
    print(f"  - Total characters: {char_count:,}")


def find_kenlm_binary(binary_name: str) -> Optional[str]:
    """Find KenLM binary in PATH or common locations."""
    # First try PATH
    try:
        result = subprocess.run(
            ['which', binary_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try common installation locations
    common_paths = [
        Path.home() / "kenlm" / "build" / "bin" / binary_name,
        Path("/usr/local/bin") / binary_name,
        Path("/usr/bin") / binary_name,
    ]
    
    for path in common_paths:
        if path.exists() and path.is_file():
            return str(path)
    
    return None


def check_kenlm_installed() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if KenLM tools are available.
    
    Returns:
        (is_installed, lmplz_path, build_binary_path)
    """
    lmplz_path = find_kenlm_binary('lmplz')
    build_binary_path = find_kenlm_binary('build_binary')
    
    is_installed = lmplz_path is not None
    
    return is_installed, lmplz_path, build_binary_path


def train_kenlm_model(
    training_text_file: Path,
    output_arpa: Path,
    output_binary: Path,
    order: int = 3,
    memory: str = "80%",
    lmplz_path: Optional[str] = None,
    build_binary_path: Optional[str] = None
) -> bool:
    """
    Train a KenLM language model.
    
    Args:
        training_text_file: Path to training text (one sentence per line)
        output_arpa: Path to output ARPA format file
        output_binary: Path to output binary format file
        order: N-gram order (default: 3 for trigram)
        memory: Memory limit (default: "80%")
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nTraining KenLM model (order={order})...")
    
    # Find lmplz binary
    if not lmplz_path:
        lmplz_path = find_kenlm_binary('lmplz')
        if not lmplz_path:
            print("Error: lmplz not found. Please install KenLM tools.")
            print("  Run: bash build_kenlm.sh")
            print("  Or build from source: https://github.com/kpu/kenlm")
            return False
    
    # Step 1: Build ARPA format model using lmplz
    print("Step 1: Building ARPA format model...")
    try:
        cmd = [
            lmplz_path,
            '-o', str(order),  # N-gram order
            '-S', memory,      # Memory limit
            '--discount_fallback',  # Use fallback discounts for unusual data distributions
            '--text', str(training_text_file),
            '--arpa', str(output_arpa)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Error building ARPA model:")
            print(result.stderr)
            return False
        
        print(f"✓ ARPA model saved to: {output_arpa}")
        
    except FileNotFoundError:
        print("Error: lmplz not found. Please install KenLM tools.")
        print("  Run: bash build_kenlm.sh")
        print("  Or build from source: https://github.com/kpu/kenlm")
        return False
    except subprocess.TimeoutExpired:
        print("Error: Training timed out (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"Error during ARPA model building: {e}")
        return False
    
    # Step 2: Convert ARPA to binary format using build_binary
    print("\nStep 2: Converting to binary format...")
    
    # Find build_binary
    if not build_binary_path:
        build_binary_path = find_kenlm_binary('build_binary')
    
    if not build_binary_path:
        print("Warning: build_binary not found. ARPA model is still available.")
        print("  Run: bash build_kenlm.sh")
        print("  Or build from source: https://github.com/kpu/kenlm")
        return True  # ARPA is still usable
    
    try:
        cmd = [
            build_binary_path,
            str(output_arpa),
            str(output_binary)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Error building binary model:")
            print(result.stderr)
            # ARPA model is still usable, so don't fail completely
            print("Warning: Binary conversion failed, but ARPA model is available")
            return True
        
        print(f"✓ Binary model saved to: {output_binary}")
        
    except FileNotFoundError:
        print("Warning: build_binary not found. ARPA model is still available.")
        print("  Run: bash build_kenlm.sh")
        print("  Or build from source: https://github.com/kpu/kenlm")
        return True  # ARPA is still usable
    except subprocess.TimeoutExpired:
        print("Warning: Binary conversion timed out. ARPA model is still available.")
        return True
    except Exception as e:
        print(f"Warning: Error during binary conversion: {e}")
        print("ARPA model is still available.")
        return True
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train a KenLM language model from step2a_merged.json files'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory containing step2a_merged.json files (default: cp40_processing/output)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='kenlm_model',
        help='Output directory for KenLM model files (default: kenlm_model)'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        help='N-gram order for language model (default: 3 for trigram)'
    )
    parser.add_argument(
        '--memory',
        type=str,
        default='80%',
        help='Memory limit for training (default: 80%%)'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=2,
        help='Minimum words per text to include (default: 2)'
    )
    parser.add_argument(
        '--delete-training-text',
        action='store_true',
        help='Delete the training text file after training (default: keep it)'
    )
    parser.add_argument(
        '--synthetic-count',
        type=int,
        default=0,
        help='Number of synthetic texts to generate using generate_text_with_replacements (default: 0)'
    )
    parser.add_argument(
        '--markov-model',
        type=str,
        default='markov_model.json',
        help='Path to Markov model JSON file for synthetic text generation (default: markov_model.json)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='cp40_records.db',
        help='Path to database file for synthetic text generation (default: cp40_records.db)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel workers for synthetic text generation (0=auto, 1=single-threaded)'
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent / "cp40_processing" / "output"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all step2a_merged.json files
    print(f"Searching for step2a_merged.json files in {base_dir}...")
    file_paths = find_all_step2a_files(base_dir)
    
    if not file_paths:
        print("No step2a_merged.json files found")
        return 1
    
    # Extract merged_text from all files (real text)
    print("Extracting merged_text from files...")
    real_texts = extract_merged_texts(file_paths, min_words=args.min_words)
    
    if not real_texts:
        print("No merged_text found in any files")
        return 1
    
    print(f"Extracted {len(real_texts):,} real texts")
    
    # Generate synthetic texts if requested
    synthetic_texts = []
    if args.synthetic_count > 0:
        model_path = Path(__file__).parent / args.markov_model
        db_path = Path(__file__).parent / args.db
        
        synthetic_texts = generate_synthetic_texts(
            num_texts=args.synthetic_count,
            model_path=model_path,
            db_path=db_path,
            min_words=args.min_words,
            tries=50,
            num_workers=args.workers
        )
    
    # Combine real and synthetic texts
    texts = real_texts + synthetic_texts
    
    print(f"\nTotal training texts: {len(texts):,}")
    print(f"  - Real texts: {len(real_texts):,}")
    print(f"  - Synthetic texts: {len(synthetic_texts):,}")
    
    # Check if KenLM is installed
    is_installed, lmplz_path, build_binary_path = check_kenlm_installed()
    if not is_installed:
        print("\nWarning: KenLM tools not found.")
        print("Please build KenLM from source:")
        print("  bash build_kenlm.sh")
        print("  Or manually: https://github.com/kpu/kenlm")
        print("\nContinuing anyway - will attempt to find KenLM tools...")
    else:
        print(f"\nFound KenLM tools:")
        if lmplz_path:
            print(f"  lmplz: {lmplz_path}")
        if build_binary_path:
            print(f"  build_binary: {build_binary_path}")
    
    # Prepare training text file
    training_text_file = output_dir / "training_text.txt"
    prepare_training_text(texts, training_text_file)
    
    # Define output paths
    output_arpa = output_dir / f"kenlm_model_{args.order}gram.arpa"
    output_binary = output_dir / f"kenlm_model_{args.order}gram.klm"
    
    # Train KenLM model
    success = train_kenlm_model(
        training_text_file,
        output_arpa,
        output_binary,
        order=args.order,
        memory=args.memory,
        lmplz_path=lmplz_path,
        build_binary_path=build_binary_path
    )
    
    if not success:
        print("\nError: Failed to train KenLM model")
        return 1
    
    # Clean up training text file if requested
    if args.delete_training_text and training_text_file.exists():
        print(f"\nRemoving training text file: {training_text_file}")
        training_text_file.unlink()
    else:
        if training_text_file.exists():
            size_mb = training_text_file.stat().st_size / (1024 * 1024)
            print(f"\nTraining text file kept: {training_text_file} ({size_mb:.2f} MB)")
    
    # Print summary
    print("\n" + "="*60)
    print("KenLM Model Training Complete!")
    print("="*60)
    print(f"\nModel files:")
    if output_arpa.exists():
        size_mb = output_arpa.stat().st_size / (1024 * 1024)
        print(f"  ARPA format: {output_arpa} ({size_mb:.2f} MB)")
    if output_binary.exists():
        size_mb = output_binary.stat().st_size / (1024 * 1024)
        print(f"  Binary format: {output_binary} ({size_mb:.2f} MB)")
    
    print(f"\nModel statistics:")
    print(f"  - N-gram order: {args.order}")
    print(f"  - Training sentences: {len(texts):,}")
    print(f"    - Real texts: {len(real_texts):,}")
    if synthetic_texts:
        print(f"    - Synthetic texts: {len(synthetic_texts):,}")
    print(f"  - Total words: {sum(len(t.split()) for t in texts):,}")
    
    print(f"\nUsage with Pylaia:")
    print(f"  The binary model ({output_binary.name}) can be used with Pylaia's")
    print(f"  language model decoder. Set the language model path in your")
    print(f"  Pylaia configuration or decoder settings.")
    
    return 0


if __name__ == "__main__":
    exit(main())

