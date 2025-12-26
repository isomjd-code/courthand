"""Main module for generating PyLaia training datasets from master_record.json files."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from line_preprocessor.processing import initial_line_extraction, process_line_image_dt_based
from workflow_manager.settings import BASE_DIR, IMAGE_DIR, OUTPUT_DIR


def find_image_file(image_filename: str, search_dirs: List[str] = None) -> Optional[str]:
    """
    Find an image file with flexible matching.
    
    Tries multiple strategies:
    1. Exact match (case-sensitive)
    2. Case-insensitive match
    3. Variations with different spacing/hyphens
    4. Partial match (base name)
    
    Args:
        image_filename: Filename from master_record.json (e.g., "CP40-592 033a.jpg")
        search_dirs: List of directories to search. Defaults to [IMAGE_DIR, BASE_DIR]
        
    Returns:
        Path to found image file, or None if not found.
    """
    if search_dirs is None:
        search_dirs = [IMAGE_DIR, BASE_DIR]
    
    # Normalize the filename
    base_name = os.path.splitext(image_filename)[0]
    ext = os.path.splitext(image_filename)[1] or ".jpg"
    
    # Strategy 1: Exact match
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        exact_path = os.path.join(search_dir, image_filename)
        if os.path.exists(exact_path):
            return exact_path
    
    # Strategy 2: Case-insensitive match
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        try:
            for file in os.listdir(search_dir):
                if file.lower() == image_filename.lower():
                    return os.path.join(search_dir, file)
        except (OSError, PermissionError):
            continue
    
    # Strategy 3: Try variations (normalize spaces and hyphens)
    # Remove all spaces and hyphens for comparison
    normalized_base = base_name.replace(" ", "").replace("-", "").replace("_", "").lower()
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        try:
            for file in os.listdir(search_dir):
                if not file.lower().endswith(ext.lower()):
                    continue
                file_base = os.path.splitext(file)[0]
                file_normalized = file_base.replace(" ", "").replace("-", "").replace("_", "").lower()
                
                if file_normalized == normalized_base:
                    return os.path.join(search_dir, file)
        except (OSError, PermissionError):
            continue
    
    # Strategy 4: Recursive search in subdirectories
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        try:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.lower() == image_filename.lower():
                        return os.path.join(root, file)
                    
                    # Also try normalized match
                    file_base = os.path.splitext(file)[0]
                    file_normalized = file_base.replace(" ", "").replace("-", "").replace("_", "").lower()
                    if file_normalized == normalized_base and file.lower().endswith(ext.lower()):
                        return os.path.join(root, file)
        except (OSError, PermissionError):
            continue
    
    return None


def find_all_master_records(base_dir: str = None) -> List[str]:
    """
    Find all master_record.json files in the output directory.
    
    Args:
        base_dir: Base directory to search. Defaults to OUTPUT_DIR.
        
    Returns:
        List of paths to master_record.json files.
    """
    if base_dir is None:
        base_dir = OUTPUT_DIR
    
    master_records = []
    for root, dirs, files in os.walk(base_dir):
        if "master_record.json" in files:
            master_records.append(os.path.join(root, "master_record.json"))
    
    return master_records


def load_cleaned_lines_from_step1_json(master_record_dir: str, image_filename: str) -> Dict[str, str]:
    """
    Load cleaned transcriptions from step1.json file.
    
    Args:
        master_record_dir: Directory containing master_record.json
        image_filename: Image filename (e.g., "CP40-565 112a.jpg")
        
    Returns:
        Dictionary mapping line_id to cleaned transcription text.
        Empty dict if step1.json not found.
    """
    # Look for step1.json file with matching image name
    base_name = os.path.splitext(image_filename)[0]
    step1_filename = f"{image_filename}_step1.json"
    step1_path = os.path.join(master_record_dir, step1_filename)
    
    if not os.path.exists(step1_path):
        # Try alternate naming patterns
        step1_filename_alt = f"{base_name}_step1.json"
        step1_path_alt = os.path.join(master_record_dir, step1_filename_alt)
        if os.path.exists(step1_path_alt):
            step1_path = step1_path_alt
        else:
            return {}
    
    try:
        with open(step1_path, "r", encoding="utf-8") as f:
            step1_data = json.load(f)
        
        # Extract cleaned_lines (prefer this over original lines)
        cleaned_lines = step1_data.get("cleaned_lines", [])
        if not cleaned_lines:
            # Fallback to original lines if cleaned_lines doesn't exist
            cleaned_lines = step1_data.get("lines", [])
        
        # Create mapping from line_id to transcription
        line_map = {}
        for line in cleaned_lines:
            line_id = line.get("id", "")
            transcription = line.get("transcription", "")
            if line_id and transcription:  # Only include non-empty transcriptions
                line_map[line_id] = transcription
        
        return line_map
    
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return {}


def extract_line_data_from_master_record(master_record_path: str, use_cleaned_lines: bool = True) -> List[Dict[str, Any]]:
    """
    Extract line-level data from a master_record.json file.
    
    Args:
        master_record_path: Path to master_record.json file.
        use_cleaned_lines: If True, prefer cleaned transcriptions from step1.json files
        
    Returns:
        List of dictionaries, each containing:
        - image_filename: Source image filename
        - line_id: Line identifier
        - original_file_id: Original file ID from master record
        - kraken_polygon: Polygon coordinates
        - text_diplomatic: Diplomatic transcription (cleaned if available)
        - master_record_dir: Directory containing the master_record.json
        - source: "cleaned" or "original" indicating transcription source
    """
    with open(master_record_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    master_record_dir = os.path.dirname(master_record_path)
    line_data_list = []
    
    source_material = data.get("source_material", [])
    for image_data in source_material:
        image_filename = image_data.get("filename", "")
        lines = image_data.get("lines", [])
        
        # Load cleaned lines for this image if available
        cleaned_lines_map = {}
        if use_cleaned_lines:
            cleaned_lines_map = load_cleaned_lines_from_step1_json(master_record_dir, image_filename)
        
        for line in lines:
            line_id = line.get("line_id", "")
            text_diplomatic = line.get("text_diplomatic", "")
            
            # Prefer cleaned transcription if available
            transcription_source = "original"
            if line_id in cleaned_lines_map:
                text_diplomatic = cleaned_lines_map[line_id]
                transcription_source = "cleaned"
            
            if not text_diplomatic:  # Skip lines without diplomatic transcription
                continue
            
            line_data_list.append({
                "image_filename": image_filename,
                "line_id": line_id,
                "original_file_id": line.get("original_file_id", ""),
                "kraken_polygon": line.get("kraken_polygon", []),
                "text_diplomatic": text_diplomatic,
                "master_record_dir": master_record_dir,
                "source": transcription_source,
            })
    
    return line_data_list


def find_kraken_json(master_record_dir: str, image_filename: str) -> Optional[str]:
    """
    Find the Kraken JSON file for a given image.
    
    Looks for kraken.json in subdirectories matching the image name.
    
    Args:
        master_record_dir: Directory containing master_record.json
        image_filename: Image filename (e.g., "CP40-565 112a.jpg")
        
    Returns:
        Path to kraken.json file, or None if not found.
    """
    # Extract base name without extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Look for subdirectories that might contain the kraken.json
    for item in os.listdir(master_record_dir):
        item_path = os.path.join(master_record_dir, item)
        if os.path.isdir(item_path) and base_name in item:
            kraken_json_path = os.path.join(item_path, "kraken.json")
            if os.path.exists(kraken_json_path):
                return kraken_json_path
    
    return None


def get_baseline_from_kraken_json(kraken_json_path: str, line_id: str) -> Optional[List[Tuple[int, int]]]:
    """
    Extract baseline coordinates for a specific line from Kraken JSON.
    
    Args:
        kraken_json_path: Path to Kraken JSON file
        line_id: Line identifier (original_file_id from master_record)
        
    Returns:
        List of (x, y) tuples for baseline points, or None if not found.
    """
    try:
        with open(kraken_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        lines = data.get("lines", [])
        for line in lines:
            if line.get("id") == line_id:
                baseline = line.get("baseline", [])
                if baseline:
                    return [tuple(point) for point in baseline]
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        pass
    
    return None


def derive_baseline_from_polygon(polygon: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Derive a baseline approximation from polygon coordinates.
    
    Uses the bottom edge of the polygon as the baseline.
    
    Args:
        polygon: List of [x, y] coordinate pairs
        
    Returns:
        List of (x, y) tuples for baseline points.
    """
    if not polygon or len(polygon) < 2:
        return []
    
    # Convert to tuples
    points = [(p[0], p[1]) for p in polygon]
    
    # Find the maximum y (bottom) for each x range
    # Simple approach: use the bottom-most points
    max_y = max(p[1] for p in points)
    bottom_points = [p for p in points if abs(p[1] - max_y) < 10]  # Within 10 pixels of bottom
    
    if len(bottom_points) < 2:
        # Fallback: use leftmost and rightmost points
        leftmost = min(points, key=lambda p: (p[0], -p[1]))  # Leftmost, prefer bottom
        rightmost = max(points, key=lambda p: (p[0], -p[1]))  # Rightmost, prefer bottom
        return [leftmost, rightmost]
    
    # Sort by x and take a few representative points
    bottom_points.sort(key=lambda p: p[0])
    
    # Take leftmost, middle, and rightmost points for a better baseline
    if len(bottom_points) > 3:
        indices = [0, len(bottom_points) // 2, len(bottom_points) - 1]
        return [bottom_points[i] for i in indices]
    
    return bottom_points


def extract_line_image(
    image_path: str,
    polygon: List[List[int]],
    baseline: List[Tuple[int, int]],
    line_id: str,
) -> Optional[Image.Image]:
    """
    Extract and process a line image from a page image.
    
    Args:
        image_path: Path to the source page image
        polygon: Polygon coordinates for the line
        baseline: Baseline coordinates
        line_id: Line identifier for error messages
        
    Returns:
        Processed line image, or None if extraction fails.
    """
    try:
        page_image = Image.open(image_path)
    except Exception:
        return None
    
    # Convert polygon to list of tuples
    polygon_tuples = [(p[0], p[1]) for p in polygon]
    
    # Convert baseline to string format expected by line_preprocessor
    baseline_str = " ".join(f"{int(p[0])},{int(p[1])}" for p in baseline)
    
    # Extract initial line region
    initial_result = initial_line_extraction(
        page_image,
        polygon_tuples,
        baseline_str,
        padding=10,
    )
    
    if not initial_result:
        return None
    
    line_rect_img, line_polygon_coords, line_baseline_points = initial_result
    
    # Process the line image
    final_image = process_line_image_dt_based(
        line_rect_img,
        line_polygon_coords,
        line_baseline_points,
        final_canvas_height=128,  # PyLaia recommended height
        line_id_for_debug=line_id,
    )
    
    return final_image


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into individual characters and spaces.
    
    Spaces are represented as '<space>' tokens.
    
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


def generate_symbol_list(all_transcriptions: List[str]) -> Dict[str, int]:
    """
    Generate symbol list mapping tokens to indices.
    
    Args:
        all_transcriptions: List of all transcriptions (non-tokenized)
        
    Returns:
        Dictionary mapping tokens to indices, starting with <ctc> at 0.
    """
    # Collect all unique characters
    all_chars = set()
    for text in all_transcriptions:
        for char in text:
            if char == " ":
                all_chars.add("<space>")
            else:
                all_chars.add(char)
    
    # Sort symbols: <ctc> first, then <space>, then <unk>, then others
    symbols = ["<ctc>", "<space>", "<unk>"]
    other_symbols = sorted([c for c in all_chars if c not in symbols])
    symbols.extend(other_symbols)
    
    # Create mapping
    symbol_map = {symbol: idx for idx, symbol in enumerate(symbols)}
    
    return symbol_map


def split_dataset(
    line_data_list: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        line_data_list: List of line data dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    random.seed(random_seed)
    shuffled = line_data_list.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    return train_data, val_data, test_data


def generate_pylaia_dataset(
    output_dir: str,
    master_records_base_dir: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    image_height: int = 128,
    use_cleaned_lines: bool = True,
) -> None:
    """
    Generate a complete PyLaia training dataset.
    
    Args:
        output_dir: Directory where the dataset will be created
        master_records_base_dir: Base directory to search for master_record.json files.
            Defaults to OUTPUT_DIR.
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        random_seed: Random seed for dataset splitting (default: 42)
        image_height: Target height for line images in pixels (default: 128)
        use_cleaned_lines: If True, use cleaned transcriptions from step1.json files (default: True)
    """
    if master_records_base_dir is None:
        master_records_base_dir = OUTPUT_DIR
    
    print(f"Searching for master_record.json files in {master_records_base_dir}...")
    master_records = find_all_master_records(master_records_base_dir)
    print(f"Found {len(master_records)} master_record.json files")
    
    # Extract all line data
    all_line_data = []
    cleaned_count = 0
    original_count = 0
    
    for master_record_path in master_records:
        line_data = extract_line_data_from_master_record(master_record_path, use_cleaned_lines=use_cleaned_lines)
        all_line_data.extend(line_data)
        
        # Count cleaned vs original
        for line in line_data:
            if line.get("source") == "cleaned":
                cleaned_count += 1
            else:
                original_count += 1
    
    print(f"Extracted {len(all_line_data)} lines with diplomatic transcriptions")
    print(f"  - {cleaned_count} lines from cleaned_lines (step1.json)")
    print(f"  - {original_count} lines from original text_diplomatic (master_record.json)")
    
    if not all_line_data:
        print("ERROR: No line data found. Exiting.")
        return
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        all_line_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    
    print(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create output directory structure
    images_dir = os.path.join(output_dir, "images")
    train_images_dir = os.path.join(images_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    test_images_dir = os.path.join(images_dir, "test")
    
    for dir_path in [train_images_dir, val_images_dir, test_images_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process each split
    splits = [
        ("train", train_data, train_images_dir),
        ("val", val_data, val_images_dir),
        ("test", test_data, test_images_dir),
    ]
    
    all_transcriptions = []
    split_files = {
        "train": {"ids": [], "text": [], "tokenized": []},
        "val": {"ids": [], "text": [], "tokenized": []},
        "test": {"ids": [], "text": [], "tokenized": []},
    }
    
    missing_images = set()
    found_images = 0
    
    for split_name, split_data, split_images_dir in splits:
        print(f"\nProcessing {split_name} set ({len(split_data)} lines)...")
        
        for idx, line_data in enumerate(split_data):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(split_data)} lines...")
            
            # Find source image
            image_filename = line_data["image_filename"]
            master_record_dir = line_data["master_record_dir"]
            
            # Try to find image with flexible matching
            search_dirs = [IMAGE_DIR, master_record_dir, BASE_DIR]
            image_path = find_image_file(image_filename, search_dirs=search_dirs)
            
            if not image_path:
                missing_images.add(image_filename)
                if len(missing_images) <= 10:  # Only print first 10 warnings to avoid spam
                    print(f"    WARNING: Image not found: {image_filename}")
                continue
            
            found_images += 1
            
            # Get baseline from Kraken JSON if available
            kraken_json_path = find_kraken_json(master_record_dir, image_filename)
            baseline = None
            if kraken_json_path:
                baseline = get_baseline_from_kraken_json(
                    kraken_json_path,
                    line_data["original_file_id"]
                )
            
            # Fallback to derived baseline
            if not baseline:
                baseline = derive_baseline_from_polygon(line_data["kraken_polygon"])
            
            if not baseline:
                print(f"    WARNING: No baseline found for line {line_data['line_id']}")
                continue
            
            # Extract line image
            line_image = extract_line_image(
                image_path,
                line_data["kraken_polygon"],
                baseline,
                line_data["line_id"],
            )
            
            if not line_image:
                print(f"    WARNING: Failed to extract line image for {line_data['line_id']}")
                continue
            
            # Generate unique image filename
            image_id = f"{split_name}/im{idx:06d}"
            image_filename_png = f"im{idx:06d}.png"
            image_path_out = os.path.join(split_images_dir, image_filename_png)
            
            # Resize to target height if needed
            if line_image.height != image_height:
                aspect_ratio = line_image.width / line_image.height
                new_width = int(image_height * aspect_ratio)
                line_image = line_image.resize((new_width, image_height), Image.Resampling.LANCZOS)
            
            # Save image
            line_image.save(image_path_out)
            
            # Get transcription
            text = line_data["text_diplomatic"]
            all_transcriptions.append(text)
            
            # Tokenize
            tokens = tokenize_text(text)
            tokenized_str = " ".join(tokens)
            
            # Store data
            split_files[split_name]["ids"].append(image_id)
            split_files[split_name]["text"].append((image_id, text))
            split_files[split_name]["tokenized"].append((image_id, tokenized_str))
    
    # Generate symbol list
    print("\nGenerating symbol list...")
    symbol_map = generate_symbol_list(all_transcriptions)
    
    # Write output files
    print("\nWriting dataset files...")
    
    # Write symbol list
    syms_path = os.path.join(output_dir, "syms.txt")
    with open(syms_path, "w", encoding="utf-8") as f:
        for symbol, idx in sorted(symbol_map.items(), key=lambda x: x[1]):
            f.write(f"{symbol} {idx}\n")
    
    # Write split files
    for split_name in ["train", "val", "test"]:
        # Write IDs file
        ids_path = os.path.join(output_dir, f"{split_name}_ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            for image_id in split_files[split_name]["ids"]:
                f.write(f"{image_id}\n")
        
        # Write text file (non-tokenized)
        text_path = os.path.join(output_dir, f"{split_name}_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for image_id, text in split_files[split_name]["text"]:
                f.write(f"{image_id} {text}\n")
        
        # Write tokenized file
        tokenized_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(tokenized_path, "w", encoding="utf-8") as f:
            for image_id, tokenized in split_files[split_name]["tokenized"]:
                f.write(f"{image_id} {tokenized}\n")
    
    print(f"\nDataset generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total lines: {len(all_line_data)}")
    print(f"Train: {len(split_files['train']['ids'])}, Val: {len(split_files['val']['ids'])}, Test: {len(split_files['test']['ids'])}")
    print(f"Images found: {found_images}, Images missing: {len(missing_images)}")
    if missing_images:
        print(f"\nMissing images ({len(missing_images)} unique):")
        for img in sorted(missing_images):
            print(f"  - {img}")


def main() -> None:
    """CLI entry point for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PyLaia training dataset from master_record.json files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(BASE_DIR, "pylaia_dataset_output"),
        help="Output directory for the dataset (default: pylaia_dataset_output)",
    )
    parser.add_argument(
        "--master-records-dir",
        type=str,
        default=None,
        help="Base directory to search for master_record.json files (default: cp40_processing/output)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion for training set (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion for validation set (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion for test set (default: 0.1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=128,
        help="Target height for line images in pixels (default: 128)",
    )
    parser.add_argument(
        "--no-cleaned-lines",
        action="store_true",
        help="Use original transcriptions instead of cleaned_lines from step1.json (default: use cleaned lines)",
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"ERROR: Ratios must sum to 1.0 (got {total_ratio})")
        return
    
    generate_pylaia_dataset(
        output_dir=args.output_dir,
        master_records_base_dir=args.master_records_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        image_height=args.image_height,
        use_cleaned_lines=not args.no_cleaned_lines,
    )


if __name__ == "__main__":
    main()

