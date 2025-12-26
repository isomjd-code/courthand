"""
Generate Pylaia training dataset from corrected lines.
"""

import json
import os
import random
import shutil
import uuid
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linprog

from line_preprocessor_greyscale.processing import initial_line_extraction, process_line_image_greyscale
from workflow_manager.settings import BASE_DIR, IMAGE_DIR, logger


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance (number of edits needed)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def clean_htr_text(text: str) -> str:
    """
    Clean HTR text by removing spaces between characters and replacing <space> tokens.
    
    Args:
        text: Raw HTR text with spaces and <space> tokens
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    # Replace <space> tokens with a temporary marker
    text = text.replace("<space>", "|||SPACE|||")
    # Remove all regular spaces (which were between individual characters)
    text = text.replace(" ", "")
    # Replace the marker back with actual spaces
    text = text.replace("|||SPACE|||", " ")
    # Clean up any multiple spaces
    text = " ".join(text.split())
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into individual characters and spaces.
    
    Spaces are represented as '<space>' tokens, matching Pylaia's expected format.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens (characters and '<space>')
    """
    tokens = []
    for char in text:
        if char == " ":
            tokens.append("<space>")
        else:
            tokens.append(char)
    return tokens


def count_symbol_occurrences(text: str) -> Dict[str, int]:
    """
    Count occurrences of each symbol in the text.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary mapping each symbol to its count
    """
    counts = defaultdict(int)
    for char in text:
        counts[char] += 1
    return dict(counts)


# Define allowed characters for training data
# These characters will always appear in syms.txt in the specified order
ALLOWED_CHARACTERS = set(
    [' ', '&', "'", '-', ';'] +  # Space and punctuation
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +  # A-Z
    [chr(i) for i in range(ord('a'), ord('z') + 1)] +  # a-z
    ['¶']  # Paragraph symbol
)

# Required characters in the exact order for syms.txt
REQUIRED_SYMBOLS = [
    "<ctc>",
    "<space>",
    "<unk>",
    "&",
    "'",
    "-",
    ";",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "¶"
]


def is_text_allowed(text: str) -> bool:
    """
    Check if text contains only allowed characters.
    
    Args:
        text: Input text string
        
    Returns:
        True if text contains only allowed characters, False otherwise
    """
    for char in text:
        if char not in ALLOWED_CHARACTERS:
            return False
    return True


def get_missing_characters(lines: List[Dict[str, Any]]) -> set:
    """
    Get set of required characters that are missing from the lines.
    
    Args:
        lines: List of line dictionaries with 'corrected_text' field
        
    Returns:
        Set of characters (excluding special tokens) that are missing
    """
    found_chars = set()
    for line_data in lines:
        text = line_data.get("corrected_text", "")
        for char in text:
            if char in ALLOWED_CHARACTERS:
                found_chars.add(char)
    
    # Required characters are all except special tokens
    required_chars = ALLOWED_CHARACTERS.copy()
    missing = required_chars - found_chars
    return missing




def count_valid_training_lines(
    corrected_lines_dir: str,
    max_levenshtein_distance: Optional[float] = None,
) -> int:
    """
    Count valid training lines from corrected_lines directory.
    
    This counts lines that would pass the Levenshtein filtering criteria
    without actually generating the full dataset.
    
    Args:
        corrected_lines_dir: Directory containing corrected_lines subdirectories
        max_levenshtein_distance: Maximum Levenshtein distance/similarity threshold.
                                 If < 1.0: Treated as similarity (0.0-1.0), e.g., 0.95 = 95% similarity required.
                                 If >= 1.0: Treated as absolute distance, e.g., 50 = max 50 character differences.
                                 Lines not meeting the threshold are filtered out. If None, no filtering is applied.
    
    Returns:
        Number of valid training lines
    """
    corrected_base = os.path.join(corrected_lines_dir, "corrected_lines")
    
    if not os.path.exists(corrected_base):
        return 0
    
    valid_count = 0
    
    for subdir in os.listdir(corrected_base):
        metadata_path = os.path.join(corrected_base, subdir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                image_path = metadata.get("image_path")
                if not image_path or not os.path.exists(image_path):
                    # Try to find image
                    image_path = find_image_file(metadata.get("image_basename", ""))
                
                if not image_path:
                    continue
                
                for line_data in metadata.get("lines", []):
                    corrected_text = line_data.get("corrected_text")
                    if not corrected_text:
                        continue

                    # Exclude lines with very short ground truth (less than 4 characters)
                    if len(corrected_text) < 4:
                        continue
                    
                    # Filter lines that contain characters not in the allowed set
                    if not is_text_allowed(corrected_text):
                        continue

                    htr_text_raw = line_data.get("htr_text", "")
                    if not htr_text_raw:
                        continue
                    
                    # Clean HTR text for comparison
                    htr_text_cleaned = clean_htr_text(htr_text_raw)
                    
                    # Apply Levenshtein filtering if specified
                    if max_levenshtein_distance is not None:
                        distance = levenshtein_distance(htr_text_cleaned, corrected_text)
                        max_len = max(len(htr_text_cleaned), len(corrected_text))
                        
                        if max_levenshtein_distance < 1.0:
                            # Similarity threshold (0.0-1.0)
                            if max_len == 0:
                                similarity = 1.0
                            else:
                                similarity = 1.0 - (distance / max_len)
                            
                            if similarity < max_levenshtein_distance:
                                continue  # Filtered out
                        else:
                            # Absolute distance threshold
                            if distance > max_levenshtein_distance:
                                continue  # Filtered out
                    
                    # Check if we have required fields for image extraction
                    polygon = line_data.get("polygon")
                    if not polygon:
                        continue
                    
                    valid_count += 1
            except Exception as e:
                logger.debug(f"Error reading {metadata_path}: {e}")
                continue
    
    return valid_count


def generate_training_dataset(
    corrected_lines_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    image_height: int = 128,
    max_levenshtein_distance: Optional[float] = None,
) -> None:
    """
    Generate Pylaia training dataset from corrected lines.
    
    Args:
        corrected_lines_dir: Directory containing corrected_lines subdirectories
        output_dir: Output directory for dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for shuffling
        image_height: Target height for line images
        max_levenshtein_distance: Maximum Levenshtein distance/similarity threshold.
                                 If < 1.0: Treated as similarity (0.0-1.0), e.g., 0.95 = 95% similarity required.
                                 If >= 1.0: Treated as absolute distance, e.g., 50 = max 50 character differences.
                                 Lines not meeting the threshold are filtered out. If None, no filtering is applied.
    """
    random.seed(random_seed)
    
    # Collect all corrected lines
    all_lines = []
    filtered_count = 0
    corrected_base = os.path.join(corrected_lines_dir, "corrected_lines")
    
    if not os.path.exists(corrected_base):
        logger.error(f"Corrected lines directory not found: {corrected_base}")
        return
    
    for subdir in os.listdir(corrected_base):
        metadata_path = os.path.join(corrected_base, subdir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                image_path = metadata.get("image_path")
                if not image_path or not os.path.exists(image_path):
                    # Try to find image
                    image_path = find_image_file(metadata.get("image_basename", ""))
                
                if not image_path:
                    logger.warning(f"Image not found for {subdir}")
                    continue
                
                for line_data in metadata.get("lines", []):
                    # Use corrected_text (cleaned Gemini output) for training
                    # Skip lines without corrected_text - they should not be part of pylaia training
                    corrected_text = line_data.get("corrected_text")
                    if not corrected_text:
                        logger.warning(f"Missing corrected_text for line {line_data.get('line_id')}, skipping from pylaia training")
                        continue

                    # Exclude lines with very short ground truth (less than 4 characters)
                    if len(corrected_text) < 4:
                        continue
                    
                    # Filter lines that contain characters not in the allowed set
                    if not is_text_allowed(corrected_text):
                        filtered_count += 1
                        logger.debug(
                            f"Filtered line {line_data.get('line_id')}: contains disallowed characters "
                            f"(text: '{corrected_text[:50]}...')"
                        )
                        continue
                    
                    # Filter by Levenshtein distance if threshold is set
                    if max_levenshtein_distance is not None:
                        htr_text_raw = line_data.get("htr_text", "")
                        if htr_text_raw:
                            # Clean htr_text to handle <space> tokens (replace with spaces, remove char-separating spaces)
                            htr_text_cleaned = clean_htr_text(htr_text_raw)
                            # Compare cleaned htr_text with corrected_text as-is (no normalization)
                            distance = levenshtein_distance(htr_text_cleaned, corrected_text)
                            
                            # Calculate similarity ratio (0.0 to 1.0)
                            max_len = max(len(htr_text_cleaned), len(corrected_text))
                            if max_len > 0:
                                similarity = 1.0 - (distance / max_len)
                                # Convert similarity threshold to distance threshold
                                # If max_levenshtein_distance < 1, treat it as similarity (0.0-1.0)
                                # Otherwise, treat it as absolute distance
                                if max_levenshtein_distance < 1.0:
                                    # Using similarity threshold (e.g., 0.95)
                                    min_similarity = max_levenshtein_distance
                                    should_filter = similarity < min_similarity
                                    filter_reason = f"similarity={similarity:.4f} < {min_similarity}"
                                else:
                                    # Using absolute distance threshold (e.g., 50)
                                    should_filter = distance > max_levenshtein_distance
                                    filter_reason = f"distance={distance} > {max_levenshtein_distance}"
                                
                                if should_filter:
                                    filtered_count += 1
                                    logger.debug(
                                        f"Filtered line {line_data.get('line_id')}: "
                                        f"{filter_reason} "
                                        f"(htr: '{htr_text_cleaned[:50]}...', corrected: '{corrected_text[:50]}...')"
                                    )
                                    continue
                            else:
                                # Empty strings - skip
                                logger.debug(f"Skipping empty line {line_data.get('line_id')}")
                                continue
                    
                    # Get baseline - can be stored as string or coordinates
                    baseline_data = line_data.get("baseline") or line_data.get("baseline_coords")
                    
                    all_lines.append({
                        "image_path": image_path,
                        "line_id": line_data["line_id"],
                        "corrected_text": corrected_text,
                        "polygon": line_data.get("polygon"),
                        "bbox": line_data.get("bbox"),
                        "baseline": baseline_data,  # Can be string or coordinates - extract_line_image handles both
                    })
            except Exception as e:
                logger.error(f"Error reading {metadata_path}: {e}")
    
    logger.info(f"Found {len(all_lines)} corrected lines")
    if max_levenshtein_distance is not None:
        if max_levenshtein_distance < 1.0:
            logger.info(f"Filtered out {filtered_count} lines with similarity < {max_levenshtein_distance:.2%}")
        else:
            logger.info(f"Filtered out {filtered_count} lines with Levenshtein distance > {max_levenshtein_distance}")
    
    if len(all_lines) == 0:
        logger.warning("No corrected lines found")
        return
    
    # Shuffle and split BEFORE augmentation
    logger.info(f"Splitting {len(all_lines)} lines into train/val/test sets...")
    random.shuffle(all_lines)
    
    n = len(all_lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_lines = all_lines[:n_train]
    val_lines = all_lines[n_train:n_train + n_val]
    test_lines = all_lines[n_train + n_val:]
    
    logger.info(f"Initial split: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")
    
    # Ensure training set contains at least one instance of each required character
    # This ensures syms.txt will always have all required characters
    missing_chars = get_missing_characters(train_lines)
    if missing_chars:
        logger.warning(f"Training set is missing {len(missing_chars)} required characters: {sorted(missing_chars)}")
        logger.info("Attempting to add lines containing missing characters to training set...")
        
        # Try to find lines in val/test sets that contain missing characters
        lines_to_move = []
        remaining_missing = missing_chars.copy()
        
        # Check validation set first
        for line_data in val_lines:
            if not remaining_missing:
                break
            text = line_data.get("corrected_text", "")
            line_chars = set(text) & remaining_missing
            if line_chars:
                lines_to_move.append(line_data)
                remaining_missing -= line_chars
        
        # If still missing, check test set
        if remaining_missing:
            for line_data in test_lines:
                if not remaining_missing:
                    break
                text = line_data.get("corrected_text", "")
                line_chars = set(text) & remaining_missing
                if line_chars:
                    lines_to_move.append(line_data)
                    remaining_missing -= line_chars
        
        # Move found lines to training set
        if lines_to_move:
            train_lines.extend(lines_to_move)
            # Remove moved lines from their original sets
            val_lines = [l for l in val_lines if l not in lines_to_move]
            test_lines = [l for l in test_lines if l not in lines_to_move]
            logger.info(f"Moved {len(lines_to_move)} lines to training set to cover missing characters")
        
        # Check if we still have missing characters
        final_missing = get_missing_characters(train_lines)
        if final_missing:
            logger.warning(f"WARNING: Training set still missing {len(final_missing)} characters: {sorted(final_missing)}")
            logger.warning("These characters will not appear in the training data but will be included in syms.txt")
        else:
            logger.info("✓ All required characters now present in training set")
    
    logger.info(f"After ensuring character coverage: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")
    
    # Apply data augmentation using linear programming ONLY to training set
    logger.info("\n=== Applying Linear Programming to Training Set ===")
    logger.info("Calculating optimal line replication for training data...")
    
    # Step 1: Count symbol occurrences per line (training set only)
    line_symbol_counts = []  # List of dicts, one per line
    all_symbols = set()
    
    for line_data in train_lines:
        text = line_data.get("corrected_text", "")
        symbol_counts = count_symbol_occurrences(text)
        line_symbol_counts.append(symbol_counts)
        all_symbols.update(symbol_counts.keys())
    
    # Convert to list for consistent indexing
    all_symbols = sorted(list(all_symbols))
    n_lines = len(train_lines)
    
    # Exclude certain symbols from the 500 occurrence constraint
    # These symbols are common structural markers that don't need augmentation
    excluded_symbols = {'-', '[', ']',';'}
    constrained_symbols = [s for s in all_symbols if s not in excluded_symbols]
    
    logger.info(f"Training set: {n_lines} lines with {len(all_symbols)} unique symbols")
    logger.info(f"Applying 500 occurrence constraint to {len(constrained_symbols)} symbols")
    logger.info(f"Excluded from constraint: {sorted(excluded_symbols & set(all_symbols))}")
    
    n_symbols = len(constrained_symbols)
    
    # Step 2: Build constraint matrix for linear program
    # Variables: w_i = weighting factor for line i (continuous, >= 1)
    # Objective: minimize sum(w_i)
    # Constraints: For each constrained symbol s, sum over lines i of (w_i * count_s_in_line_i) >= 500
    
    # Objective coefficients: minimize sum of weights
    c = np.ones(n_lines)
    
    # Inequality constraints: -A_ub @ x <= -b_ub (converted to A @ x >= b)
    # For each constrained symbol, we need: sum(w_i * symbol_count[i][symbol]) >= 500
    A_ub = []
    b_ub = []
    
    for symbol in constrained_symbols:
        # Constraint row for this symbol
        row = []
        for line_idx in range(n_lines):
            # Count of this symbol in this line
            count = line_symbol_counts[line_idx].get(symbol, 0)
            # Negate for inequality format (we'll use -A @ x <= -b to represent A @ x >= b)
            row.append(-count)
        A_ub.append(row)
        b_ub.append(-500)  # Negative because we're using -A @ x <= -b form (constraint: >= 500)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: all weights >= 1 (each line must appear at least once), no upper bound
    bounds = [(1, None) for _ in range(n_lines)]
    
    # Step 3: Solve the linear program
    logger.info("Solving linear program...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        logger.error(f"Linear program failed to converge: {result.message}")
        logger.warning("Falling back to no augmentation for training set")
        # train_lines remains unchanged - no augmentation applied
    else:
        weights = result.x
        logger.info(f"Linear program solved. Optimal objective value: {result.fun:.2f}")
        
        # Step 4: Round down to get integer replication counts
        replication_counts = np.floor(weights).astype(int)
        
        # Log statistics
        total_replications = replication_counts.sum()
        lines_replicated = (replication_counts > 0).sum()
        max_replication = replication_counts.max()
        
        logger.info(f"Total replications: {total_replications}")
        logger.info(f"Lines to be replicated: {lines_replicated}/{n_lines}")
        logger.info(f"Max replication count: {max_replication}")
        
        # Detailed distribution statistics
        logger.info("\n=== Replication Count Distribution ===")
        
        # Count how many lines have each replication count
        unique_counts, count_freq = np.unique(replication_counts, return_counts=True)
        logger.info(f"Distribution of replication counts:")
        for count, freq in zip(unique_counts, count_freq):
            percentage = (freq / n_lines) * 100
            logger.info(f"  {count}x: {freq} lines ({percentage:.1f}%)")
        
        # Statistical measures
        non_zero_counts = replication_counts[replication_counts > 0]
        if len(non_zero_counts) > 0:
            logger.info(f"\nStatistics (all lines):")
            logger.info(f"  Mean: {replication_counts.mean():.2f}")
            logger.info(f"  Median: {np.median(replication_counts):.1f}")
            logger.info(f"  Std Dev: {replication_counts.std():.2f}")
            
            logger.info(f"\nStatistics (replicated lines only, count > 0):")
            logger.info(f"  Mean: {non_zero_counts.mean():.2f}")
            logger.info(f"  Median: {np.median(non_zero_counts):.1f}")
            logger.info(f"  Min: {non_zero_counts.min()}")
            logger.info(f"  Max: {non_zero_counts.max()}")
            
            # Percentiles
            if len(non_zero_counts) >= 4:
                p25, p75, p90, p95, p99 = np.percentile(non_zero_counts, [25, 75, 90, 95, 99])
                logger.info(f"  25th percentile: {p25:.1f}")
                logger.info(f"  75th percentile: {p75:.1f}")
                logger.info(f"  90th percentile: {p90:.1f}")
                logger.info(f"  95th percentile: {p95:.1f}")
                logger.info(f"  99th percentile: {p99:.1f}")
        
        # Show top replicated lines with reasons
        logger.info("\n=== Top 10 Most Replicated Lines ===")
        top_indices = np.argsort(replication_counts)[::-1][:10]
        
        for rank, idx in enumerate(top_indices, 1):
            count = replication_counts[idx]
            if count == 0:
                break
            
            line_data = all_lines[idx]
            text = line_data.get("corrected_text", "")
            line_id = line_data.get("line_id", "")
            
            # Find which symbols in this line are rare
            line_symbols = line_symbol_counts[idx]
            rare_symbols = []
            for symbol, symbol_count in line_symbols.items():
                # Calculate how many total occurrences of this symbol across all lines
                total_symbol_count = sum(
                    line_symbol_counts[i].get(symbol, 0) 
                    for i in range(n_lines)
                )
                if total_symbol_count < 1000:  # Consider symbols with < 1000 occurrences as "rare"
                    rare_symbols.append((symbol, total_symbol_count))
            
            # Sort by rarity
            rare_symbols.sort(key=lambda x: x[1])
            
            text_preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(f"{rank}. Line {line_id}: {count}x replications")
            logger.info(f"   Text: {text_preview}")
            if rare_symbols:
                rare_str = ", ".join([f"'{s}' ({c} total)" for s, c in rare_symbols[:5]])
                logger.info(f"   Rare symbols: {rare_str}")
            else:
                logger.info(f"   No rare symbols (< 1000 occurrences)")
        
        logger.info("=" * 40 + "\n")
        
        # Verify constraints are met (with rounded values)
        symbol_totals = defaultdict(int)
        for line_idx, count in enumerate(replication_counts):
            for symbol, occurrences in line_symbol_counts[line_idx].items():
                symbol_totals[symbol] += count * occurrences
        
        # Show all symbol occurrences after augmentation
        logger.info("\n=== Symbol Occurrences After Augmentation ===")
        
        # Separate constrained and excluded symbols
        constrained_symbol_counts = [(s, symbol_totals.get(s, 0)) for s in constrained_symbols]
        excluded_symbol_counts = [(s, symbol_totals.get(s, 0)) for s in sorted(excluded_symbols & set(all_symbols))]
        
        # Sort constrained symbols by count (ascending) to see which are below threshold
        constrained_symbol_counts.sort(key=lambda x: x[1])
        
        logger.info(f"\nConstrained symbols (target: >= 500 occurrences):")
        for symbol, count in constrained_symbol_counts:
            status = "✓" if count >= 500 else "✗"
            logger.info(f"  {status} '{symbol}': {count:,} occurrences")
        
        logger.info(f"\nExcluded symbols (no constraint applied):")
        for symbol, count in excluded_symbol_counts:
            logger.info(f"    '{symbol}': {count:,} occurrences")
        
        # Summary statistics
        total_constrained = len(constrained_symbols)
        meeting_threshold = sum(1 for _, count in constrained_symbol_counts if count >= 500)
        below_threshold = total_constrained - meeting_threshold
        
        logger.info(f"\nSummary:")
        logger.info(f"  Constrained symbols meeting threshold: {meeting_threshold}/{total_constrained}")
        if below_threshold > 0:
            logger.info(f"  Constrained symbols below threshold: {below_threshold}")
        logger.info("=" * 40 + "\n")
        
        # Keep original warning for symbols significantly below threshold
        symbols_below_threshold = []
        for symbol in constrained_symbols:
            total = symbol_totals[symbol]
            if total < 500:
                symbols_below_threshold.append((symbol, total))
        
        if symbols_below_threshold:
            logger.warning(f"\n⚠️  WARNING: {len(symbols_below_threshold)} constrained symbols still below 500 threshold after rounding")
            logger.warning(f"This may indicate the training set needs more diverse data for these symbols:")
            for symbol, total in sorted(symbols_below_threshold, key=lambda x: x[1])[:10]:
                logger.warning(f"  '{symbol}': {total} occurrences")
        
        # Step 5: Create augmented training dataset
        augmented_train_lines = []
        augmentation_stats = {
            "lines_replicated": 0,
            "total_replications": 0,
        }
        
        for line_idx, line_data in enumerate(train_lines):
            replication_count = replication_counts[line_idx]
            
            # Add the line replication_count times
            for rep in range(replication_count):
                if rep == 0:
                    # First instance: use original line_id
                    augmented_train_lines.append(line_data)
                else:
                    # Subsequent instances: create new unique ID
                    original_line_id = line_data["line_id"]
                    new_line_id = f"{original_line_id}_aug_{uuid.uuid4().hex[:8]}"
                    
                    duplicate_line = {
                        "image_path": line_data["image_path"],
                        "line_id": new_line_id,
                        "corrected_text": line_data["corrected_text"],
                        "polygon": line_data.get("polygon"),
                        "bbox": line_data.get("bbox"),
                        "baseline": line_data.get("baseline"),
                    }
                    augmented_train_lines.append(duplicate_line)
                    augmentation_stats["total_replications"] += 1
            
            if replication_count > 0:
                augmentation_stats["lines_replicated"] += 1
        
        logger.info(f"\nAugmentation complete: {augmentation_stats['total_replications']} additional lines created")
        logger.info(f"  - Lines replicated: {augmentation_stats['lines_replicated']}")
        logger.info(f"  - Training set size after augmentation: {len(augmented_train_lines)}")
        
        # Replace training set with augmented version
        train_lines = augmented_train_lines
    
    # Shuffle training set after augmentation
    random.shuffle(train_lines)
    
    # Val and test sets remain unchanged (single copy of each line)
    logger.info(f"\nFinal split sizes:")
    logger.info(f"  Training: {len(train_lines)} lines (with augmentation)")
    logger.info(f"  Validation: {len(val_lines)} lines (no augmentation)")
    logger.info(f"  Test: {len(test_lines)} lines (no augmentation)")
    
    splits = [
        ("train", train_lines),
        ("val", val_lines),
        ("test", test_lines),
    ]
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for split_name, split_lines in splits:
        split_images_dir = os.path.join(images_dir, split_name)
        os.makedirs(split_images_dir, exist_ok=True)
        
        # Main training file: {image_id} {tokenized_text}
        list_file = os.path.join(output_dir, f"{split_name}.txt")
        # Also keep separate files for reference
        text_file = os.path.join(output_dir, f"{split_name}_text.txt")
        ids_file = os.path.join(output_dir, f"{split_name}_ids.txt")
        
        with open(list_file, 'w', encoding='utf-8') as f_list, \
             open(text_file, 'w', encoding='utf-8') as f_text, \
             open(ids_file, 'w', encoding='utf-8') as f_ids:
            
            for line_data in split_lines:
                line_id = line_data["line_id"]
                image_path = line_data["image_path"]
                polygon = line_data.get("polygon")
                corrected_text = line_data["corrected_text"]
                
                # Check if line image already exists from previous model training
                existing_image_path = find_existing_line_image(
                    line_id,
                    output_dir,
                    split_name,
                )
                
                if existing_image_path:
                    # Copy existing image instead of creating from scratch
                    output_image_path = os.path.join(split_images_dir, f"{line_id}.png")
                    try:
                        shutil.copy2(existing_image_path, output_image_path)
                        line_image_path = os.path.abspath(output_image_path)
                        logger.debug(f"Copied existing line image for {line_id} from {existing_image_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy existing image for {line_id}: {e}, extracting from scratch")
                        line_image_path = None
                else:
                    line_image_path = None
                
                # Extract line image if not found in previous datasets
                if not line_image_path:
                    baseline_data = line_data.get("baseline")
                    line_image_path = extract_line_image(
                        image_path,
                        polygon,
                        line_id,
                        split_images_dir,
                        image_height,
                        baseline_coords=baseline_data  # Can be string or coordinates
                    )
                
                if line_image_path:
                    # Get relative path from images directory for Pylaia format
                    # Pylaia expects: {split_name}/{filename} format
                    rel_image_path = os.path.relpath(line_image_path, images_dir)
                    # Normalize path separators (use forward slash)
                    rel_image_path = rel_image_path.replace(os.sep, "/")
                    
                    # Tokenize text for Pylaia format (space-separated chars with <space> tokens)
                    tokens = tokenize_text(corrected_text)
                    tokenized_text = " ".join(tokens)
                    
                    # Write main training file: {image_id} {tokenized_text}
                    f_list.write(f"{rel_image_path} {tokenized_text}\n")
                    # Also write separate files for reference
                    f_text.write(f"{corrected_text}\n")
                    f_ids.write(f"{line_id}\n")
    
    # Generate symbols file with exact order as specified
    # Always use REQUIRED_SYMBOLS in the exact order, regardless of what's in the data
    # This ensures syms.txt is always consistent
    symbols = REQUIRED_SYMBOLS.copy()
    
    # Create symbol map with indices (matching pylaia_dataset format)
    symbol_map = {symbol: idx for idx, symbol in enumerate(symbols)}
    
    syms_file = os.path.join(output_dir, "syms.txt")
    with open(syms_file, 'w', encoding='utf-8') as f:
        # Write symbols with indices (Pylaia expects this format: "symbol index")
        # Use the exact order from REQUIRED_SYMBOLS
        for idx, symbol in enumerate(symbols):
            f.write(f"{symbol} {idx}\n")
    
    # Verify that all characters in training data are in the symbol set
    # Convert space characters to <space> tokens for checking
    training_chars = set()
    for split_name, split_lines in splits:
        for line_data in split_lines:
            text = line_data.get("corrected_text", "")
            for char in text:
                if char == " ":
                    training_chars.add("<space>")
                else:
                    training_chars.add(char)
    
    missing_from_symbols = training_chars - set(symbols)
    if missing_from_symbols:
        logger.warning(f"WARNING: Training data contains characters not in syms.txt: {sorted(missing_from_symbols)}")
        logger.warning("These characters were filtered out - this should not happen if filtering is working correctly")
    
    logger.info(f"Dataset generated in {output_dir}")


def find_image_file(basename: str) -> Optional[str]:
    """Find image file by basename."""
    search_dirs = [IMAGE_DIR, BASE_DIR]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        try:
            for file in os.listdir(search_dir):
                if os.path.splitext(file)[0] == basename:
                    return os.path.join(search_dir, file)
        except (OSError, PermissionError):
            continue
    
    return None


def find_existing_line_image(
    line_id: str,
    output_dir: str,
    split_name: str,
) -> Optional[str]:
    """
    Find existing line image from previous model training datasets.
    
    Searches previous dataset directories (dataset_v1, dataset_v2, etc.) for
    an existing line image with the same line_id. If found, returns the path
    to the existing image.
    
    Args:
        line_id: Line identifier
        output_dir: Current output directory (to determine dataset base directory)
        split_name: Split name (train/val/test) to search in
        
    Returns:
        Path to existing line image if found, None otherwise
    """
    # Extract dataset base directory from output_dir
    # output_dir is typically: bootstrap_training_data/datasets/dataset_vN
    # We want to search in: bootstrap_training_data/datasets/
    dataset_base_dir = os.path.dirname(output_dir)
    
    if not os.path.exists(dataset_base_dir):
        return None
    
    # Search all dataset directories (dataset_v1, dataset_v2, etc.)
    # Sort in reverse order to check newer datasets first
    try:
        dataset_dirs = []
        for item in os.listdir(dataset_base_dir):
            item_path = os.path.join(dataset_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("dataset_v"):
                dataset_dirs.append(item_path)
        
        # Sort by version number (descending) to check newer datasets first
        def extract_version(path):
            basename = os.path.basename(path)
            match = basename.replace("dataset_v", "")
            try:
                return int(match)
            except ValueError:
                return 0
        
        dataset_dirs.sort(key=extract_version, reverse=True)
        
        # Don't check the current output_dir
        current_output = os.path.abspath(output_dir)
        dataset_dirs = [d for d in dataset_dirs if os.path.abspath(d) != current_output]
        
        # Search for the line image in each dataset
        for dataset_dir in dataset_dirs:
            # Check in images/{split_name}/{line_id}.png
            candidate_path = os.path.join(dataset_dir, "images", split_name, f"{line_id}.png")
            if os.path.exists(candidate_path):
                return candidate_path
        
        return None
    except (OSError, PermissionError) as e:
        logger.debug(f"Error searching for existing line image {line_id}: {e}")
        return None


def extract_line_image(
    image_path: str,
    polygon: List[List[int]],
    line_id: str,
    output_dir: str,
    image_height: int = 128,
    baseline_coords: Optional[Any] = None
) -> Optional[str]:
    """
    Extract and process a line image.
    
    Args:
        image_path: Path to source image
        polygon: Polygon coordinates
        line_id: Line identifier
        output_dir: Output directory
        image_height: Target height
        baseline_coords: Baseline from Kraken - can be:
            - String format: "x1,y1 x2,y2 ..." (from parse_kraken_json_for_processing)
            - List of coordinates: [[x1, y1], [x2, y2], ...]
            - None: will derive from polygon
        
    Returns:
        Path to saved line image or None
    """
    try:
        page_image = Image.open(image_path).convert("L")
        
        if not polygon or len(polygon) < 2:
            return None
        
        # Convert polygon to list of tuples (handle both list and tuple formats)
        if polygon and len(polygon) > 0:
            if isinstance(polygon[0], (list, tuple)) and len(polygon[0]) >= 2:
                polygon_tuples = [(int(p[0]), int(p[1])) for p in polygon]
            else:
                return None
        else:
            return None
        
        # Use Kraken baseline if available, otherwise derive from polygon
        # baseline_coords can be either:
        # - A list of coordinates: [[x1, y1], [x2, y2], ...]
        # - A string: "x1,y1 x2,y2 ..." (from parse_kraken_json_for_processing)
        if baseline_coords:
            if isinstance(baseline_coords, str):
                # Already in string format
                baseline_str = baseline_coords
            elif isinstance(baseline_coords, list) and len(baseline_coords) >= 2:
                # Convert coordinates to string format
                baseline_points = [(int(p[0]), int(p[1])) for p in baseline_coords]
                baseline_str = " ".join(f"{int(p[0])},{int(p[1])}" for p in baseline_points)
            else:
                baseline_str = None
        else:
            baseline_str = None
        
        # Fallback: derive baseline from polygon if not available
        if not baseline_str:
            baseline_points = derive_baseline_from_polygon(polygon_tuples)
            if not baseline_points:
                return None
            baseline_str = " ".join(f"{int(p[0])},{int(p[1])}" for p in baseline_points)
        
        # Extract line
        initial_result = initial_line_extraction(
            page_image,
            polygon_tuples,
            baseline_str,
            padding=10,
        )
        
        if not initial_result:
            return None
        
        line_rect_img, line_polygon_coords, line_baseline_points = initial_result
        
        # Process line
        final_image = process_line_image_greyscale(
            line_rect_img,
            line_polygon_coords,
            line_baseline_points,
            final_canvas_height=image_height,
            line_id_for_debug=line_id,
        )
        
        if not final_image:
            return None
        
        # Save
        output_path = os.path.join(output_dir, f"{line_id}.png")
        final_image.save(output_path)
        return os.path.abspath(output_path)
        
    except Exception as e:
        logger.error(f"Error extracting line image {line_id}: {e}")
        return None


def derive_baseline_from_polygon(polygon: List) -> Optional[List]:
    """Derive baseline from polygon coordinates."""
    if not polygon or len(polygon) < 2:
        return None
    
    # Use bottom edge of polygon as baseline
    # Sort by y-coordinate and take bottom points
    sorted_points = sorted(polygon, key=lambda p: p[1])
    bottom_y = sorted_points[-1][1]
    
    # Get points near bottom
    bottom_points = [p for p in polygon if abs(p[1] - bottom_y) < 5]
    if len(bottom_points) < 2:
        bottom_points = sorted_points[-2:]
    
    # Sort by x-coordinate
    bottom_points.sort(key=lambda p: p[0])
    
    return bottom_points

