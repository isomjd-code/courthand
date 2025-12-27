#!/usr/bin/env python3
"""
Extract vertical slices from training line images corresponding to each n-gram.

Uses PyLaia for CTC forced alignment to determine character positions.
"""

import os
import sys
import argparse
import json
import glob
import time
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import map_coordinates

# Import laia components
try:
    from laia.common.loader import ModelLoader
    from laia.decoders import CTCGreedyDecoder
    from laia.utils import SymbolsTable
    LAIA_AVAILABLE = True
except ImportError as e:
    LAIA_AVAILABLE = False
    print(f"Error: Failed to import laia components: {e}")
    print("Please activate the pylaia environment and ensure laia is installed.")
    sys.exit(1)


def load_ngrams(ngrams_file: str) -> List[Tuple[str, int, int]]:
    """
    Load n-grams from the generated file.
    
    Returns:
        List of (ngram, length, frequency) tuples
    """
    ngrams = []
    with open(ngrams_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                ngram = parts[0]
                length = int(parts[1])
                freq = int(parts[2])
                ngrams.append((ngram, length, freq))
    return ngrams


def load_training_data(dataset_path: str, skip_augmented: bool = True) -> List[Dict]:
    """
    Load training data (images and text).
    
    Args:
        dataset_path: Path to dataset directory
        skip_augmented: If True, skip lines from augmented images (containing "aug" in filename)
    
    Returns:
        List of dicts with 'image_path', 'text', 'id'
    """
    train_txt = os.path.join(dataset_path, 'train.txt')
    images_dir = os.path.join(dataset_path, 'images', 'train')
    
    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"Training file not found: {train_txt}")
    
    training_data = []
    skipped_count = 0
    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: train/uuid.png <space> separated characters
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue
            
            image_rel_path = parts[0]
            text = parts[1]
            
            # Skip augmented images if requested (to match generate_ngrams.py behavior)
            if skip_augmented and 'aug' in image_rel_path.lower():
                skipped_count += 1
                continue
            
            # Get image ID from path
            image_id = os.path.splitext(os.path.basename(image_rel_path))[0]
            image_path = os.path.join(images_dir, os.path.basename(image_rel_path))
            
            if os.path.exists(image_path):
                training_data.append({
                    'id': image_id,
                    'image_path': image_path,
                    'text': text  # Space-separated characters
                })
    
    if skip_augmented and skipped_count > 0:
        print(f"Skipped {skipped_count} lines from augmented images (matching generate_ngrams.py behavior)")
    
    return training_data


def clean_text_for_alignment(text: str) -> str:
    """
    Convert space-separated character format to continuous text.
    Handles <space> tokens.
    """
    chars = text.split()
    result = []
    for char in chars:
        if char == '<space>':
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


def normalize_ngram_for_search(ngram: str) -> str:
    """
    Normalize n-gram to match the format used in cleaned text.
    
    Rules:
    - Regular spaces " " are removed (they don't count)
    - "<space>" tokens are converted to space characters ' ' for matching
    - All other characters are kept as-is
    
    Args:
        ngram: N-gram string (may contain spaces or be continuous)
        
    Returns:
        Normalized n-gram string for searching in cleaned text
    """
    # Convert <space> tokens to actual space characters for matching
    # Remove regular spaces (they don't count in length)
    result = []
    i = 0
    while i < len(ngram):
        if ngram[i] == '<':
            # Check if it's a <space> token
            if ngram[i:i+7] == '<space>':
                result.append(' ')  # Convert <space> to space character
                i += 7
            else:
                # Other angle bracket token, find closing '>'
                end = ngram.find('>', i)
                if end != -1:
                    result.append(ngram[i:end+1])  # Keep the token as-is
                    i = end + 1
                else:
                    result.append(ngram[i])
                    i += 1
        elif ngram[i] == ' ':
            # Regular space - skip it (doesn't count)
            i += 1
        else:
            # Regular character
            result.append(ngram[i])
            i += 1
    
    return ''.join(result)


def get_base_image_id(image_id: str) -> str:
    """
    Extract base image ID by removing augmentation suffixes.
    
    Removes patterns like '_aug1', '_aug2', '_augmentation1', etc.
    Also handles patterns like '_aug_1', '_augmentation_1', etc.
    
    Examples:
        'image_001_aug1' -> 'image_001'
        'image_001_aug_2' -> 'image_001'
        'image_001_augmentation1' -> 'image_001'
        'image_001_augmentation_3' -> 'image_001'
        'image_001' -> 'image_001' (no change)
    
    Args:
        image_id: Image ID that may contain augmentation suffix
        
    Returns:
        Base image ID without augmentation suffix
    """
    # Remove common augmentation patterns at the end of the string
    # Pattern: _aug followed by optional underscore and one or more digits
    base_id = re.sub(r'_aug(?:_\d+|\d+)$', '', image_id)
    # Pattern: _augmentation followed by optional underscore and one or more digits
    base_id = re.sub(r'_augmentation(?:_\d+|\d+)$', '', base_id)
    
    return base_id


def ctc_forced_alignment(
    log_probs: torch.Tensor,
    text: str,
    syms: SymbolsTable
) -> Optional[List[int]]:
    """
    Perform CTC forced alignment using a simpler peak-finding approach.
    
    For each character, find the time step with the highest probability for that character,
    while respecting ordering constraints.
    
    Args:
        log_probs: (T, C) log probabilities from model
        text: Target text string
        syms: Symbols table
        
    Returns:
        List of time step indices for each character, or None if alignment fails
    """
    # Convert text to indices
    indices = []
    for char in text:
        try:
            idx = syms[char]
            # Convert to int if it's a tensor
            if isinstance(idx, torch.Tensor):
                idx = int(idx.item())
            else:
                idx = int(idx)
            indices.append(idx)
        except (KeyError, TypeError):
            # Try space
            if char == ' ':
                try:
                    idx = syms['<space>']
                    if isinstance(idx, torch.Tensor):
                        idx = int(idx.item())
                    else:
                        idx = int(idx)
                    indices.append(idx)
                except (KeyError, TypeError):
                    return None
            else:
                return None
    
    if not indices:
        return None
    
    T, C = log_probs.shape
    S = len(indices)
    
    if T < S:
        return None
    
    # Convert log_probs to probabilities for easier peak finding
    probs = torch.exp(log_probs)
    
    # Simple approach: for each character, find the peak probability time step
    # while ensuring monotonic ordering
    alignment = []
    current_t = 0
    
    for char_idx in indices:
        # Search for the best time step for this character, starting from current_t
        # Look ahead up to a reasonable distance
        search_end = min(current_t + max(1, (T - current_t) // max(1, S - len(alignment)) * 2), T)
        
        best_t = current_t
        best_prob = 0.0
        
        # Find peak probability for this character in the search window
        for t in range(current_t, search_end):
            prob = float(probs[t, char_idx].item())
            if prob > best_prob:
                best_prob = prob
                best_t = t
        
        # If we didn't find a good match, try expanding search
        if best_prob < 0.01:  # Very low probability
            # Search entire sequence
            for t in range(T):
                prob = float(probs[t, char_idx].item())
                if prob > best_prob:
                    best_prob = prob
                    best_t = t
        
        alignment.append(best_t)
        # Move forward: next character should be at or after this time step
        current_t = max(best_t + 1, current_t + 1)
        
        # Ensure we don't run out of time steps
        if current_t >= T and len(alignment) < S:
            # Pad remaining with last time step
            alignment.extend([T - 1] * (S - len(alignment)))
            break
    
    return alignment


def extract_vertical_slice(
    image: Image.Image,
    start_x: int,
    end_x: int
) -> Image.Image:
    """
    Extract a vertical slice from an image.
    
    Args:
        image: PIL Image
        start_x: Start x coordinate
        end_x: End x coordinate
        
    Returns:
        Cropped image slice
    """
    width, height = image.size
    
    # Clamp coordinates to image bounds
    start_x = max(0, min(int(start_x), width - 1))
    end_x = max(start_x + 1, min(int(end_x), width))
    
    # Ensure we have a valid slice
    if start_x >= end_x or start_x >= width or end_x <= 0:
        # Return a minimal slice if coordinates are invalid
        start_x = 0
        end_x = max(1, min(width, 10))  # At least 1 pixel wide, max 10 pixels
    
    return image.crop((start_x, 0, end_x, height))


def process_ngrams(
    ngrams: List[Tuple[str, int, int]],
    training_data: List[Dict],
    model,
    syms: SymbolsTable,
    output_dir: str,
    max_examples_per_ngram: int = 1000,
    checkpoint_file: str = None
):
    """
    Process n-grams and extract vertical slices.
    
    Args:
        ngrams: List of (ngram, length, frequency) tuples
        training_data: List of training examples
        model: PyLaia model
        syms: Symbols table
        output_dir: Output directory for slices
        max_examples_per_ngram: Maximum number of examples to extract per n-gram
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each n-gram length (up to 15)
    # Find the maximum n-gram length from the n-grams list
    max_length = max(length for _, length, _ in ngrams) if ngrams else 15
    for n in range(1, max_length + 1):
        os.makedirs(os.path.join(output_dir, f'{n}gram'), exist_ok=True)
    
    # We'll create n-gram-specific subdirectories as we process them
    
    # Group training data by text for faster lookup
    # Use a more memory-efficient approach: only store indices
    text_to_indices = defaultdict(list)
    for idx, example in enumerate(training_data):
        clean_text = clean_text_for_alignment(example['text'])
        text_to_indices[clean_text].append(idx)
    
    print(f"Processing {len(ngrams)} n-grams...")
    
    # Load checkpoint if it exists
    processed_ngrams = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_ngrams = set(checkpoint_data.get('processed_ngrams', []))
                print(f"Loaded checkpoint: {len(processed_ngrams)} n-grams already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            processed_ngrams = set()
    
    ngram_stats = defaultdict(int)
    start_time = time.time()
    last_log_time = start_time
    
    for ngram_idx, (ngram, length, freq) in enumerate(ngrams):
        # Skip if already processed
        if ngram in processed_ngrams:
            if (ngram_idx + 1) % 100 == 0:
                print(f"  Skipped {ngram_idx + 1}/{len(ngrams)} (already processed)...")
            continue
        current_time = time.time()
        if current_time - last_log_time > 10.0:  # Log every 10 seconds
            elapsed = current_time - start_time
            rate = (ngram_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(ngrams) - ngram_idx - 1) / rate if rate > 0 else 0
            print(f"  Progress: {ngram_idx + 1}/{len(ngrams)} n-grams ({ngram_idx + 1}/{len(ngrams)*100:.1f}%) - "
                  f"Rate: {rate:.1f} n-grams/sec - ETA: {remaining/60:.1f} min")
            last_log_time = current_time
        
        if (ngram_idx + 1) % 100 == 0:
            print(f"  Processed {ngram_idx + 1}/{len(ngrams)} n-grams...")
        
        # Find training examples containing this n-gram
        find_start = time.time()
        # Normalize n-gram to match cleaned text format
        normalized_ngram = normalize_ngram_for_search(ngram)
        example_indices = []
        for text, indices in text_to_indices.items():
            if normalized_ngram in text:
                example_indices.extend(indices)
        
        if not example_indices:
            continue
        
        # Limit number of examples
        example_indices = example_indices[:max_examples_per_ngram]
        examples_found = [training_data[idx] for idx in example_indices]
        before_dedup_count = len(examples_found)
        del example_indices  # Free memory
        
        # Deduplicate based on base image ID (remove augmentation variants)
        seen_base_ids = set()
        deduplicated_examples = []
        for example in examples_found:
            base_id = get_base_image_id(example['id'])
            if base_id not in seen_base_ids:
                seen_base_ids.add(base_id)
                deduplicated_examples.append(example)
        examples_found = deduplicated_examples
        after_dedup_count = len(examples_found)
        del deduplicated_examples, seen_base_ids  # Free memory
        
        if len(examples_found) > 0:
            if before_dedup_count != after_dedup_count:
                print(f"    N-gram '{ngram}' (length {length}, freq {freq}): found {before_dedup_count} instances, {after_dedup_count} after deduplication")
            else:
                print(f"    N-gram '{ngram}' (length {length}, freq {freq}): found {after_dedup_count} instances")
        
        # Create n-gram specific directory once per n-gram
        # Sanitize directory name - replace problematic characters
        ngram_safe_for_dir = ngram.replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Remove or replace other problematic characters for filesystem
        ngram_safe_for_dir = re.sub(r'[<>:"|?*]', '_', ngram_safe_for_dir)
        # Limit directory name length
        if len(ngram_safe_for_dir) > 100:
            ngram_safe_for_dir = ngram_safe_for_dir[:100]
        ngram_dir = os.path.join(output_dir, f'{length}gram', ngram_safe_for_dir)
        os.makedirs(ngram_dir, exist_ok=True)
        
        # Create temporary directory for intermediate slices (will be cleaned up after PCA)
        temp_dir = tempfile.mkdtemp(prefix='ngram_slices_')
        temp_slice_paths = []  # List of (temp_path, final_output_path, width)
        device = next(model.parameters()).device
        
        # Process each example and save slices to disk immediately
        extract_start = time.time()
        for example_idx, example in enumerate(examples_found):
            if example_idx > 0 and example_idx % 50 == 0:
                print(f"      Extracting instance {example_idx + 1}/{len(examples_found)} for '{ngram}'...")
            
            try:
                # Load and preprocess image
                img = Image.open(example['image_path']).convert('L')
                original_width, original_height = img.size
                
                # Resize to standard height (128px) for PyLaia
                target_height = 128
                scale_factor = target_height / original_height
                new_width = int(original_width * scale_factor)
                img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                
                # Convert to tensor
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
                
                # Move to same device as model
                img_tensor = img_tensor.to(device)
                
                # Get model output
                inference_start = time.time()
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor)
                inference_time = time.time() - inference_start
                if inference_time > 0.1:  # Log slow inferences
                    print(f"        Slow inference: {inference_time:.3f}s for {example['id']}")
                
                # Ensure (T, N, C) format
                if output.dim() == 3 and output.size(0) == img_tensor.size(0):
                    output = output.transpose(0, 1)
                
                log_probs = F.log_softmax(output, dim=2)
                # Remove batch dimension: should be (T, C) or (T, 1, C)
                while log_probs.dim() > 2:
                    log_probs = log_probs.squeeze(1)
                # Ensure we have (T, C) shape
                if log_probs.dim() == 1:
                    # If somehow we got (C,), expand to (1, C)
                    log_probs = log_probs.unsqueeze(0)
                
                # Calculate horizontal compression ratio
                # PyLaia models compress images horizontally (typically 4:1 or 8:1)
                # T is the number of time steps (compressed width)
                # new_width is the input image width
                T = log_probs.shape[0]
                compression_ratio = new_width / max(T, 1)  # pixels per time step
                
                # Use CTC decoder to get the actual decoded sequence
                # This gives us the model's best guess at what characters are at each time step
                from laia.decoders import CTCGreedyDecoder
                decoder = CTCGreedyDecoder(syms)
                
                # Convert log_probs to probabilities for argmax
                probs = torch.exp(log_probs)
                
                # Decode: get character indices for each time step
                decoded_indices = []
                for t in range(T):
                    char_idx = int(torch.argmax(probs[t]).item())
                    decoded_indices.append((t, char_idx))
                
                # Build decoded text by filtering out blanks and repeats
                decoded_chars = []
                decoded_times = []
                prev_char_idx = None
                for t, char_idx in decoded_indices:
                    if char_idx != 0:  # Not blank
                        if char_idx != prev_char_idx:  # Not a repeat
                            decoded_chars.append(syms[char_idx] if char_idx < len(syms) else '')
                            decoded_times.append(t)
                        prev_char_idx = char_idx
                    else:
                        prev_char_idx = None
                
                decoded_text = ''.join(decoded_chars)
                
                # Find n-gram in decoded text
                decoded_ngram_pos = decoded_text.find(ngram)
                if decoded_ngram_pos == -1:
                    # If not found in decoded text, try the ground truth text
                    clean_text = clean_text_for_alignment(example['text'])
                    ngram_pos = clean_text.find(ngram)
                    if ngram_pos == -1:
                        # Clear memory before continuing
                        del img, img_resized, img_array, img_tensor, output, log_probs, probs
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    # Use proportional mapping as fallback
                    # Map character position to approximate pixel position
                    char_ratio = ngram_pos / len(clean_text)
                    end_char_ratio = (ngram_pos + len(ngram)) / len(clean_text)
                    start_x = int(char_ratio * new_width)
                    end_x = int(end_char_ratio * new_width)
                else:
                    # Use decoded positions - more accurate
                    if decoded_ngram_pos + len(ngram) > len(decoded_times):
                        # Clear memory before continuing
                        del img, img_resized, img_array, img_tensor, output, log_probs, probs
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    ngram_times = decoded_times[decoded_ngram_pos:decoded_ngram_pos + len(ngram)]
                    if not ngram_times:
                        # Clear memory before continuing
                        del img, img_resized, img_array, img_tensor, output, log_probs, probs
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    start_time = min(ngram_times)
                    end_time = max(ngram_times) + 1
                    
                    # Convert time steps to x-coordinates
                    start_x = int(start_time * compression_ratio)
                    end_x = int(end_time * compression_ratio)
                
                # Add padding to account for alignment uncertainty and compression artifacts
                # Use larger padding (30% of compression ratio or at least 5 pixels)
                padding = max(5, int(compression_ratio * 0.3))
                start_x = max(0, start_x - padding)
                end_x = min(new_width, end_x + padding)
                
                # Ensure end_x is at least start_x + 1
                end_x = max(start_x + 1, end_x)
                
                # Clamp to image bounds
                start_x = max(0, min(start_x, new_width - 1))
                end_x = max(start_x + 1, min(end_x, new_width))
                
                # Extract slice from resized image
                try:
                    slice_img = extract_vertical_slice(img_resized, start_x, end_x)
                    
                    # Verify slice is valid
                    if slice_img.size[0] <= 0 or slice_img.size[1] <= 0:
                        # Clear memory
                        del img, img_resized, img_array, img_tensor, output, log_probs, probs, slice_img
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    # Save slice to temporary directory immediately (saves memory)
                    ngram_safe = ngram.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    # Limit filename length
                    if len(ngram_safe) > 50:
                        ngram_safe = ngram_safe[:50]
                    temp_filename = f"{example['id']}_{example_idx}.png"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    final_output_path = os.path.join(ngram_dir, temp_filename)
                    
                    # Save to disk immediately
                    slice_img.save(temp_path)
                    temp_slice_paths.append((temp_path, final_output_path, slice_img.size[0]))
                    
                    # Clear memory immediately
                    del img, img_resized, img_array, img_tensor, output, log_probs, probs, slice_img
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    # Clear memory on error
                    if 'img' in locals():
                        del img
                    if 'img_resized' in locals():
                        del img_resized
                    if 'img_tensor' in locals():
                        del img_tensor
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
            except Exception as e:
                print(f"  Error processing {example['id']} for n-gram '{ngram}': {e}")
                # Clear any remaining memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        # After processing all instances for this n-gram, filter to keep only the 5% closest to centroid
        if len(temp_slice_paths) > 1:
            print(f"    Applying PCA filtering for '{ngram}' ({len(temp_slice_paths)} instances)...")
            pca_start = time.time()
            
            # First pass: determine median width and height without loading all images
            widths = []
            height = None
            for temp_path, _, width in temp_slice_paths:
                widths.append(width)
                if height is None:
                    # Load just one image to get height
                    img = Image.open(temp_path).convert('L')
                    height = img.size[1]
                    del img
            
            median_width = int(np.median(widths))
            
            # Load and normalize images in batches to save memory
            # Process in chunks to avoid loading everything at once
            batch_size = min(100, len(temp_slice_paths))  # Process 100 at a time
            normalized_data = []  # Store (final_path, normalized_array) but only temporarily
            
            for batch_start in range(0, len(temp_slice_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(temp_slice_paths))
                batch_paths = temp_slice_paths[batch_start:batch_end]
                
                # Load batch
                batch_images = []
                for temp_path, final_path, _ in batch_paths:
                    img = Image.open(temp_path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    h, w = img_array.shape
                    
                    # Resample to median width using horizontal distance normalization
                    y_coords, x_coords = np.mgrid[0:h, 0:median_width]
                    x_orig = (x_coords / median_width) * w
                    x_orig = np.clip(x_orig, 0, w - 1)
                    y_orig = np.clip(y_coords, 0, h - 1)
                    coords = np.array([y_orig, x_orig])
                    resampled = map_coordinates(img_array, coords, order=1, mode='constant', cval=0.0)
                    normalized_data.append((final_path, resampled.astype(np.float32)))
                    
                    # Clear immediately
                    del img, img_array, resampled
                
                # Clear batch
                del batch_images
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Apply PCA to find centroid (now we have normalized data)
            if len(normalized_data) >= 2:
                # Build flattened array in batches to save memory
                # First, get the feature dimension
                sample_img = normalized_data[0][1]
                feature_dim = sample_img.size
                
                # Build X array in chunks if needed
                if len(normalized_data) * feature_dim > 100000000:  # ~100MB threshold
                    # Use incremental approach - build X in smaller chunks
                    X_parts = []
                    chunk_size = max(1, 100000000 // feature_dim)
                    for i in range(0, len(normalized_data), chunk_size):
                        chunk = normalized_data[i:i+chunk_size]
                        X_chunk = np.array([img.flatten() for _, img in chunk])
                        X_parts.append(X_chunk)
                        del chunk, X_chunk
                    X = np.vstack(X_parts)
                    del X_parts
                else:
                    # Small enough to build all at once
                    X = np.array([img.flatten() for _, img in normalized_data])
                
                n_components = min(len(normalized_data) - 1, X.shape[1], 50)
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    X_transformed = pca.fit_transform(X)
                    
                    # Find centroid
                    centroid = np.mean(X_transformed, axis=0)
                    
                    # Calculate distances to centroid
                    distances = np.linalg.norm(X_transformed - centroid, axis=1)
                    
                    # Keep top 5% closest
                    keep_count = max(1, int(len(normalized_data) * 0.05))
                    closest_indices = np.argsort(distances)[:keep_count]
                    
                    # Clear X and X_transformed to free memory
                    del X, X_transformed, distances
                    
                    # Save only the closest instances (5%)
                    for idx in closest_indices:
                        final_path, img_array = normalized_data[idx]
                        # Ensure directory exists before saving
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        img_uint8 = (img_array * 255).astype(np.uint8)
                        Image.fromarray(img_uint8).save(final_path)
                        ngram_stats[length] += 1
                    
                    # Clear normalized_data
                    del normalized_data
                    
                    pca_time = time.time() - pca_start
                    print(f"    PCA filtering complete: kept {len(closest_indices)}/{len(temp_slice_paths)} instances in {pca_time:.1f}s")
                else:
                    # If PCA fails, save all
                    for final_path, img_array in normalized_data:
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        img_uint8 = (img_array * 255).astype(np.uint8)
                        Image.fromarray(img_uint8).save(final_path)
                        ngram_stats[length] += 1
                    del normalized_data
            else:
                # If only one instance, save it
                final_path, img_array = normalized_data[0]
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                img_uint8 = (img_array * 255).astype(np.uint8)
                Image.fromarray(img_uint8).save(final_path)
                ngram_stats[length] += 1
                del normalized_data
        elif len(temp_slice_paths) == 1:
            # Only one instance, move from temp to final location
            temp_path, final_path, _ = temp_slice_paths[0]
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.move(temp_path, final_path)
            ngram_stats[length] += 1
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"    Warning: Could not clean up temp directory {temp_dir}: {e}")
        
        # Mark as processed and save checkpoint after each n-gram
        processed_ngrams.add(ngram)
        if checkpoint_file:
            try:
                checkpoint_data = {
                    'processed_ngrams': list(processed_ngrams),
                    'last_processed_idx': ngram_idx,
                    'last_update': time.time()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}")
    
    # Print statistics
    print("\n=== Extraction Statistics ===")
    for length in sorted(ngram_stats.keys()):
        print(f"  {length}-grams: {ngram_stats[length]} slices extracted")


def main():
    parser = argparse.ArgumentParser(
        description='Extract vertical slices from training images for each n-gram'
    )
    parser.add_argument(
        '--ngrams-file',
        type=str,
        default='important_ngrams.txt',
        help='Path to n-grams file (default: important_ngrams.txt)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='bootstrap_training_data/datasets/dataset_v22',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to PyLaia model checkpoint'
    )
    parser.add_argument(
        '--syms-file',
        type=str,
        help='Path to symbols file'
    )
    parser.add_argument(
        '--model-arch',
        type=str,
        help='Path to model architecture file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ngram_slices',
        help='Output directory for slices (default: ngram_slices)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=1000,
        help='Maximum examples to extract per n-gram before filtering (default: 1000, will keep 5%% closest to centroid)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file path for resume functionality (default: output_dir/.checkpoint.json)'
    )
    
    args = parser.parse_args()
    
    # Try to get model paths from settings if not provided
    if not args.model_path or not args.syms_file or not args.model_arch:
        try:
            from workflow_manager.settings import PYLAIA_MODEL, PYLAIA_SYMS, PYLAIA_ARCH
            if os.path.exists(PYLAIA_MODEL) and os.path.exists(PYLAIA_SYMS) and os.path.exists(PYLAIA_ARCH):
                args.model_path = args.model_path or PYLAIA_MODEL
                args.syms_file = args.syms_file or PYLAIA_SYMS
                args.model_arch = args.model_arch or PYLAIA_ARCH
        except ImportError:
            pass
    
    # If still not set, try to find latest bootstrap model
    if not args.model_path or not args.syms_file or not args.model_arch:
        bootstrap_models_dir = 'bootstrap_training_data/pylaia_models'
        if os.path.exists(bootstrap_models_dir):
            # Find latest model version
            model_dirs = [d for d in os.listdir(bootstrap_models_dir) 
                         if d.startswith('model_v') and os.path.isdir(os.path.join(bootstrap_models_dir, d))]
            if model_dirs:
                # Extract version numbers and find latest
                def get_version(d):
                    try:
                        return int(d.replace('model_v', ''))
                    except:
                        return 0
                latest_dir = max(model_dirs, key=get_version)
                latest_model_dir = os.path.join(bootstrap_models_dir, latest_dir)
                
                # Find checkpoint
                checkpoint = None
                checkpoint_path = os.path.join(latest_model_dir, 'epoch=*-lowest_va_cer.ckpt')
                checkpoints = glob.glob(checkpoint_path)
                if checkpoints:
                    checkpoint = checkpoints[0]
                else:
                    # Try experiment directory
                    exp_dir = os.path.join(latest_model_dir, 'experiment')
                    if os.path.exists(exp_dir):
                        checkpoints = glob.glob(os.path.join(exp_dir, 'epoch=*-lowest_va_cer.ckpt'))
                        if checkpoints:
                            checkpoint = checkpoints[0]
                
                syms_file = os.path.join(latest_model_dir, 'syms.txt')
                model_file = os.path.join(latest_model_dir, 'model')
                
                if checkpoint and os.path.exists(syms_file) and os.path.exists(model_file):
                    args.model_path = args.model_path or checkpoint
                    args.syms_file = args.syms_file or syms_file
                    args.model_arch = args.model_arch or model_file
                    print(f"Using latest bootstrap model: {latest_dir}")
    
    if not args.model_path or not args.syms_file or not args.model_arch:
        parser.error(
            "Model path, syms file, and model arch must be provided.\n"
            "Example: --model-path bootstrap_training_data/pylaia_models/model_v22/epoch=*-lowest_va_cer.ckpt "
            "--syms-file bootstrap_training_data/pylaia_models/model_v22/syms.txt "
            "--model-arch bootstrap_training_data/pylaia_models/model_v22/model"
        )
    
    print("Loading PyLaia model...")
    # Load symbols
    syms = SymbolsTable(str(args.syms_file))
    
    # Load model
    train_path = str(Path(args.model_arch).parent)
    model_filename = Path(args.model_arch).name
    loader = ModelLoader(train_path, filename=model_filename, device=args.device)
    
    try:
        checkpoint = loader.prepare_checkpoint(
            str(args.model_path), experiment_dirpath=None, monitor=None
        )
    except (TypeError, AttributeError):
        checkpoint = str(args.model_path)
    
    model = loader.load_by(checkpoint)
    model.eval()
    
    # Extract actual model from Lightning wrapper if needed
    for attr in ['model', 'net', 'crnn']:
        if hasattr(model, attr):
            actual_model = getattr(model, attr)
            actual_model.eval()
            model = actual_model
            break
    
    model = model.to(args.device)
    print("✓ Model loaded successfully\n")
    
    # Load n-grams
    print(f"Loading n-grams from {args.ngrams_file}...")
    ngrams = load_ngrams(args.ngrams_file)
    print(f"Loaded {len(ngrams)} n-grams\n")
    
    # Load training data
    print(f"Loading training data from {args.dataset_path}...")
    training_data = load_training_data(args.dataset_path)
    print(f"Loaded {len(training_data)} training examples\n")
    
    # Set checkpoint file path
    if args.checkpoint is None:
        checkpoint_file = os.path.join(args.output_dir, '.checkpoint.json')
    else:
        checkpoint_file = args.checkpoint
    
    # Process n-grams
    try:
        process_ngrams(
            ngrams,
            training_data,
            model,
            syms,
            args.output_dir,
            max_examples_per_ngram=args.max_examples,
            checkpoint_file=checkpoint_file
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint saved. You can resume by running the same command.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheckpoint saved. You can resume by running the same command.")
        sys.exit(1)
    
    print(f"\n✓ Done! Slices saved to {args.output_dir}")


if __name__ == '__main__':
    main()

