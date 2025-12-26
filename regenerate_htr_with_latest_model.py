#!/usr/bin/env python3
"""
Script to regenerate HTR results for all images using the latest bootstrap PyLaia model.

This script:
1. Finds all images in input_images/ that have existing HTR work
2. Uses the latest bootstrap model (highest model_vN) to re-run PyLaia decode
3. Keeps existing kraken.json (preserves line segmentation)
4. Overwrites lines/ subfolder, htr.txt, and img_list.txt
5. Updates the htr_text field in corresponding metadata.json files in corrected_lines/
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")

# Directories
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
BOOTSTRAP_DATA_DIR = os.path.join(BASE_DIR, "bootstrap_training_data")
HTR_WORK_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work")
CORRECTED_LINES_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines")
PYLAIA_MODELS_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "pylaia_models")
TEMP_SCRIPTS_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "temp_scripts")

# Virtual environment activation scripts
PYLAIA_ENV = os.path.join(HOME_DIR, "projects/pylaia-env/bin/activate")

# Line image height for bootstrap models (v1+ use 128px)
BOOTSTRAP_IMAGE_HEIGHT = 128


def find_latest_model() -> Tuple[str, str, str, int]:
    """
    Find the latest bootstrap PyLaia model.
    
    Returns:
        Tuple of (checkpoint_path, model_file_path, syms_file_path, version_number)
    """
    # Look for model directories
    model_dirs = []
    for item in os.listdir(PYLAIA_MODELS_DIR):
        if item.startswith("model_v") and os.path.isdir(os.path.join(PYLAIA_MODELS_DIR, item)):
            try:
                version = int(item.replace("model_v", ""))
                model_dirs.append((version, item))
            except ValueError:
                continue
    
    if not model_dirs:
        raise RuntimeError("No bootstrap models found in pylaia_models/")
    
    # Sort by version number and get the latest
    model_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_version, latest_dir = model_dirs[0]
    
    model_dir = os.path.join(PYLAIA_MODELS_DIR, latest_dir)
    
    # Find the best checkpoint (highest epoch with "lowest_va_cer")
    checkpoint = None
    best_epoch = -1
    
    # Check root directory first
    for filename in os.listdir(model_dir):
        if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
            try:
                epoch_str = filename.split("=")[1].split("-")[0]
                epoch = int(epoch_str)
                if epoch > best_epoch:
                    best_epoch = epoch
                    checkpoint = os.path.join(model_dir, filename)
            except (ValueError, IndexError):
                if checkpoint is None:
                    checkpoint = os.path.join(model_dir, filename)
    
    # Check experiment directory if not found
    if checkpoint is None:
        experiment_dir = os.path.join(model_dir, "experiment")
        if os.path.exists(experiment_dir):
            for filename in os.listdir(experiment_dir):
                if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                    try:
                        epoch_str = filename.split("=")[1].split("-")[0]
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            checkpoint = os.path.join(experiment_dir, filename)
                    except (ValueError, IndexError):
                        if checkpoint is None:
                            checkpoint = os.path.join(experiment_dir, filename)
    
    if checkpoint is None:
        raise RuntimeError(f"No checkpoint found for model_v{latest_version}")
    
    model_file = os.path.join(model_dir, "model")
    syms_file = os.path.join(model_dir, "syms.txt")
    
    if not os.path.exists(model_file):
        raise RuntimeError(f"Model file not found: {model_file}")
    if not os.path.exists(syms_file):
        raise RuntimeError(f"Symbols file not found: {syms_file}")
    
    print(f"Using latest model: v{latest_version}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Model file: {model_file}")
    print(f"  Symbols: {syms_file}")
    
    return checkpoint, model_file, syms_file, latest_version


def get_htr_work_dirs() -> List[str]:
    """Get all HTR work directories that have existing kraken.json."""
    work_dirs = []
    for item in os.listdir(HTR_WORK_DIR):
        item_path = os.path.join(HTR_WORK_DIR, item)
        if os.path.isdir(item_path):
            kraken_json = os.path.join(item_path, "kraken.json")
            if os.path.exists(kraken_json):
                work_dirs.append(item)
    return sorted(work_dirs)


def create_preprocessing_script(image_path: str, kraken_json: str, lines_dir: str, 
                                 list_txt: str, height: int) -> str:
    """Create a temporary preprocessing script for line extraction."""
    os.makedirs(TEMP_SCRIPTS_DIR, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    script_path = os.path.join(TEMP_SCRIPTS_DIR, f"preprocess_{basename.replace(' ', '_')}.py")
    
    script_content = f'''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '{BASE_DIR}')

# CRITICAL: Override FINAL_LINE_HEIGHT in config BEFORE any imports
import line_preprocessor_greyscale.config
line_preprocessor_greyscale.config.FINAL_LINE_HEIGHT = {height}

from line_preprocessor_greyscale.runner import _load_image, _expand_polygons, _save_line_image
from line_preprocessor.parser import parse_kraken_json_for_processing
from line_preprocessor_greyscale.processing import initial_line_extraction, process_line_image_greyscale
from PIL import Image
from tqdm import tqdm

def custom_main(image_path, json_path, output_dir, pylaia_list_path, height):
    page_image = _load_image(image_path)
    lines_to_process = parse_kraken_json_for_processing(json_path)
    if not lines_to_process:
        print("No text lines found in the Kraken JSON. Exiting.", file=sys.stderr)
        sys.exit(0)
    
    print(f"Found {{len(lines_to_process)}} lines to process. Expanding polygons...")
    expanded_lines = _expand_polygons(lines_to_process)
    
    print(f"Processing lines with height={{height}}px...")
    os.makedirs(output_dir, exist_ok=True)
    with open(pylaia_list_path, "w", encoding="utf-8") as pylaia_list_file:
        for line_data in tqdm(expanded_lines, desc="Processing lines"):
            try:
                initial_result = initial_line_extraction(
                    page_image,
                    line_data["polygon"],
                    line_data["baseline"],
                    padding=10,
                )
                if not initial_result:
                    continue
                
                line_rect_img, line_polygon_coords, line_baseline_points = initial_result
                final_image = process_line_image_greyscale(
                    line_rect_img,
                    line_polygon_coords,
                    line_baseline_points,
                    final_canvas_height=height,
                    line_id_for_debug=line_data["id"],
                )
                if final_image:
                    abs_path = _save_line_image(final_image, output_dir, line_data["id"])
                    pylaia_list_file.write(f"{{abs_path}}\\n")
            except Exception as exc:
                print(f"    - FATAL WARNING: Unhandled exception on line {{line_data['id']}}: {{exc}}", file=sys.stderr)
    
    print(f"Pylaia input file list saved to: {{pylaia_list_path}}")

if __name__ == '__main__':
    custom_main('{image_path}', '{kraken_json}', '{lines_dir}', '{list_txt}', {height})
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',  # Use bash for 'source' command support
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  Warning: {description} failed: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Warning: {description} timed out")
        return False
    except Exception as e:
        print(f"  Warning: {description} error: {e}")
        return False


def run_command_streaming(cmd: str, description: str) -> bool:
    """Run a shell command with real-time output streaming."""
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n  Warning: {description} failed with return code {process.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  Warning: {description} timed out")
        return False
    except Exception as e:
        print(f"\n  Warning: {description} error: {e}")
        return False


def create_python_batch_decode_script(work_items: List[Tuple[str, str, str]], 
                                        checkpoint: str, model_file: str, syms_file: str) -> str:
    """
    Create a Python script that loads the model once and processes all images.
    This eliminates model loading overhead between images.
    """
    os.makedirs(TEMP_SCRIPTS_DIR, exist_ok=True)
    script_path = os.path.join(TEMP_SCRIPTS_DIR, "batch_decode_python.py")
    
    # Create the work items data structure
    work_items_str = "[\n"
    for basename, list_txt, htr_res in work_items:
        work_items_str += f"    ('{basename}', '{list_txt}', '{htr_res}'),\n"
    work_items_str += "]"
    
    script_content = f'''#!/usr/bin/env python3
"""
Python-based batch decoder that loads the model once and processes all images.
This eliminates the 2-3 second model loading overhead between images.
"""
import sys
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# Add project to path
sys.path.insert(0, '{BASE_DIR}')

# Import laia components
try:
    from laia.common.loader import ModelLoader
    from laia.data import ImageDataset
    from laia.decoders import CTCGreedyDecoder
    from laia.utils import SymbolsTable, ImageToTensor
    from torch.utils.data import DataLoader
    LAIA_AVAILABLE = True
except ImportError as e:
    LAIA_AVAILABLE = False
    print(f"ERROR: Failed to import laia components: {{e}}", file=sys.stderr)
    print("", file=sys.stderr)
    print("This script requires the laia package to be installed in the PyLaia environment.", file=sys.stderr)
    print("The script will exit and the main script should fall back to CLI-based processing.", file=sys.stderr)
    sys.exit(1)

# Work items: (basename, list_txt_path, htr_res_path)
WORK_ITEMS = {work_items_str}

CHECKPOINT = '{checkpoint}'
MODEL_FILE = '{model_file}'
SYMS_FILE = '{syms_file}'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """Load PyLaia model once."""
    print("Loading model...")
    print(f"  Checkpoint: {{CHECKPOINT}}")
    print(f"  Model file: {{MODEL_FILE}}")
    print(f"  Symbols: {{SYMS_FILE}}")
    print(f"  Device: {{DEVICE}}")
    
    # Load symbols
    syms = SymbolsTable(str(SYMS_FILE))
    
    # Load model
    train_path = str(Path(MODEL_FILE).parent)
    model_filename = Path(MODEL_FILE).name
    loader = ModelLoader(train_path, filename=model_filename, device=DEVICE)
    
    try:
        checkpoint = loader.prepare_checkpoint(
            str(CHECKPOINT), experiment_dirpath=None, monitor=None
        )
    except (TypeError, AttributeError):
        checkpoint = str(CHECKPOINT)
    
    model = loader.load_by(checkpoint)
    model.eval()
    
    # Extract actual model from Lightning wrapper if needed
    for attr in ['model', 'net', 'crnn']:
        if hasattr(model, attr):
            actual_model = getattr(model, attr)
            actual_model.eval()
            model = actual_model
            break
    
    print("✓ Model loaded successfully\\n")
    return model, syms

def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess line image for PyLaia model."""
    img = Image.open(image_path).convert('L')
    width, height = img.size
    new_height = 128
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

def decode_images(model, syms, image_paths: list) -> list:
    """Decode a batch of images using PyLaia's DataLoader approach."""
    if not image_paths:
        return []
    
    try:
        # Create dataset - ImageDataset expects a list of image paths
        dataset = ImageDataset(image_paths, img_transform=ImageToTensor())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        decoder = CTCGreedyDecoder(syms)
        results = []
        path_index = 0
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # batch is a dict with 'img' and 'id' keys
                images = batch['img'].to(DEVICE)
                batch_size = images.shape[0]
                
                # Run inference
                output = model(images)
                
                # Decode each image in the batch
                for i in range(batch_size):
                    img_output = output[i:i+1]
                    decoded = decoder(img_output)
                    # Convert decoded indices to text
                    # decoded is a list of lists, get the first sequence
                    decoded_seq = decoded[0] if len(decoded) > 0 else []
                    text_chars = []
                    for idx in decoded_seq:
                        idx_int = int(idx)
                        if 0 <= idx_int < len(syms):
                            text_chars.append(syms[idx_int])
                    text = ''.join(text_chars)
                    # Format as space-separated characters (matching pylaia-htr-decode-ctc format)
                    text_formatted = ' '.join(text)
                    results.append((image_paths[path_index], text_formatted))
                    path_index += 1
        
        return results
    except Exception as e:
        # Fallback to simpler approach if DataLoader fails
        print(f"  Warning: DataLoader approach failed, using fallback: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return decode_images_fallback(model, syms, image_paths)

def decode_images_fallback(model, syms, image_paths: list) -> list:
    """Fallback decode using simple preprocessing."""
    results = []
    decoder = CTCGreedyDecoder(syms)
    
    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            try:
                tensor = preprocess_image(img_path)
                output = model(tensor)
                decoded = decoder(output)
                # Convert to text
                decoded_seq = decoded[0] if len(decoded) > 0 else []
                text_chars = []
                for idx in decoded_seq:
                    idx_int = int(idx)
                    if 0 <= idx_int < len(syms):
                        text_chars.append(syms[idx_int])
                text = ''.join(text_chars)
                text_formatted = ' '.join(text)
                results.append((img_path, text_formatted))
            except Exception as e:
                print(f"  Warning: Failed to decode {{img_path}}: {{e}}", file=sys.stderr)
    
    return results

def process_image_list(model, syms, list_txt: str, htr_res: str):
    """Process a single image list file."""
    # Read image paths
    image_paths = []
    if os.path.exists(list_txt):
        with open(list_txt, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    
    if not image_paths:
        # Create empty output file
        with open(htr_res, 'w') as f:
            pass
        return True
    
    # Decode images
    results = decode_images(model, syms, image_paths)
    
    # Write results in pylaia-htr-decode-ctc format
    with open(htr_res, 'w', encoding='utf-8') as f:
        for img_path, decoded_text in results:
            # Format: path ['conf1', 'conf2', ...] c h a r <space> t e x t
            # For now, we'll use a simplified format without confidence scores
            # (pylaia CLI format compatibility)
            text_formatted = ' '.join(decoded_text) if isinstance(decoded_text, list) else decoded_text
            f.write(f"{{img_path}} [] {{text_formatted}}\\n")
    
    return True

def main():
    """Main entry point."""
    total = len(WORK_ITEMS)
    start_time = time.time()
    
    print("=" * 70)
    print(f"Python Batch Decoder - Processing {{total}} images")
    print("=" * 70)
    print()
    
    # Load model once
    try:
        model, syms = load_model()
    except Exception as e:
        print(f"ERROR: Failed to load model: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process each image
    success_count = 0
    fail_count = 0
    
    for idx, (basename, list_txt, htr_res) in enumerate(WORK_ITEMS, 1):
        img_start = time.time()
        print(f"[{{idx}}/{{total}}] Processing: {{basename}}")
        
        try:
            if process_image_list(model, syms, list_txt, htr_res):
                img_duration = time.time() - img_start
                elapsed = time.time() - start_time
                if idx > 0:
                    remaining = (elapsed * (total - idx)) / idx
                    print(f"  ✓ Completed in {{img_duration:.1f}}s (Total: {{elapsed:.1f}}s, Est. remaining: {{remaining:.1f}}s)")
                else:
                    print(f"  ✓ Completed in {{img_duration:.1f}}s (Total: {{elapsed:.1f}}s)")
                success_count += 1
            else:
                print(f"  ✗ Failed to process {{basename}}")
                fail_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {{basename}}: {{e}}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            fail_count += 1
        
        print()
    
    total_duration = time.time() - start_time
    print("=" * 70)
    print(f"Batch decode completed in {{total_duration:.1f}} seconds")
    print(f"Success: {{success_count}}, Failed: {{fail_count}}")
    print("=" * 70)

if __name__ == '__main__':
    main()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def create_combined_decode_script(work_items: List[Tuple[str, str, str]], 
                                  checkpoint: str, model_file: str, syms_file: str,
                                  chunk_size: int = 10000) -> Tuple[str, str]:
    """
    Create a script that combines all image lists and processes them in chunks.
    Validates that all line images have height 128px before adding to queue.
    Then splits the results back to individual files.
    
    Args:
        chunk_size: Maximum number of line images per chunk (to avoid memory/CLI limits)
    
    Returns:
        Tuple of (decode_script_path, combined_list_path)
    """
    os.makedirs(TEMP_SCRIPTS_DIR, exist_ok=True)
    
    # Create combined image list
    combined_list_path = os.path.join(TEMP_SCRIPTS_DIR, "combined_img_list.txt")
    combined_results_path = os.path.join(TEMP_SCRIPTS_DIR, "combined_htr_results.txt")
    
    # Build mapping: line_image_path -> (basename, htr_res_path)
    # Filter images to only include those with height 128
    # Track which images need reprocessing
    line_to_image = {}
    all_lines = []
    skipped_count = 0
    images_to_reprocess = set()  # Track basenames that need reprocessing
    
    from PIL import Image
    
    print("  Validating line images (checking height = 128px)...")
    for basename, list_txt, htr_res in work_items:
        if os.path.exists(list_txt) and os.path.getsize(list_txt) > 0:
            image_has_invalid_lines = False
            valid_lines_count = 0
            invalid_lines_count = 0
            
            with open(list_txt, 'r') as f:
                for line_path in f:
                    line_path = line_path.strip()
                    if not line_path:
                        continue
                    
                    # Ensure absolute path for consistent matching
                    if not os.path.isabs(line_path):
                        line_path = os.path.abspath(line_path)
                    
                    # Check image height
                    try:
                        if os.path.exists(line_path):
                            with Image.open(line_path) as img:
                                width, height = img.size
                                if height == 128:
                                    # Normalize path to absolute for consistent matching
                                    abs_line_path = os.path.abspath(line_path)
                                    all_lines.append(abs_line_path)
                                    # Store normalized absolute path
                                    if abs_line_path not in line_to_image:
                                        line_to_image[abs_line_path] = (basename, htr_res)
                                    valid_lines_count += 1
                                else:
                                    invalid_lines_count += 1
                                    image_has_invalid_lines = True
                        else:
                            invalid_lines_count += 1
                            image_has_invalid_lines = True
                    except Exception as e:
                        # Skip if we can't read the image
                        invalid_lines_count += 1
                        image_has_invalid_lines = True
            
            # If this image has invalid lines, mark it for reprocessing
            if image_has_invalid_lines:
                images_to_reprocess.add((basename, list_txt, htr_res))
                skipped_count += invalid_lines_count
    
    # Reprocess images with invalid lines
    if images_to_reprocess:
        print(f"\n  Found {len(images_to_reprocess)} images with invalid line heights")
        print(f"  Reprocessing line images for these images...")
        
        for basename, list_txt, htr_res in images_to_reprocess:
            # Reprocess this image
            prep_result = process_image_prep(basename, skip_preprocessing=False)
            if prep_result:
                # Re-validate the newly processed lines
                new_basename, new_list_txt, new_htr_res = prep_result
                if os.path.exists(new_list_txt) and os.path.getsize(new_list_txt) > 0:
                    with open(new_list_txt, 'r') as f:
                        for line_path in f:
                            line_path = line_path.strip()
                            if not line_path:
                                continue
                            
                            # Check image height again
                            try:
                                if os.path.exists(line_path):
                                    with Image.open(line_path) as img:
                                        width, height = img.size
                                        if height == 128:
                                            all_lines.append(line_path)
                                            if line_path not in line_to_image:
                                                line_to_image[line_path] = (new_basename, new_htr_res)
                                        else:
                                            skipped_count += 1
                            except Exception:
                                skipped_count += 1
                print(f"    ✓ Reprocessed: {basename}")
            else:
                print(f"    ✗ Failed to reprocess: {basename}")
    
    if skipped_count > 0:
        print(f"\n  Skipped {skipped_count} line images with incorrect height (after reprocessing)")
    
    # Write combined list (remove duplicates)
    unique_lines = []
    seen = set()
    with open(combined_list_path, 'w', encoding='utf-8') as f:
        for line_path in all_lines:
            if line_path not in seen:
                f.write(f"{line_path}\n")
                unique_lines.append(line_path)
                seen.add(line_path)
    
    num_unique_lines = len(unique_lines)
    print(f"  Validated {num_unique_lines} line images with correct height (128px)")
    
    # Split into chunks if too large
    chunks = []
    if num_unique_lines > chunk_size:
        print(f"  Splitting into chunks of {chunk_size}...")
        for i in range(0, num_unique_lines, chunk_size):
            chunk_lines = unique_lines[i:i + chunk_size]
            chunk_list_path = os.path.join(TEMP_SCRIPTS_DIR, f"combined_img_list_chunk_{i//chunk_size + 1}.txt")
            chunk_results_path = os.path.join(TEMP_SCRIPTS_DIR, f"combined_htr_results_chunk_{i//chunk_size + 1}.txt")
            
            with open(chunk_list_path, 'w', encoding='utf-8') as f:
                for line_path in chunk_lines:
                    f.write(f"{line_path}\n")
            
            chunks.append((chunk_list_path, chunk_results_path, len(chunk_lines), f"chunk_{i//chunk_size + 1}"))
    else:
        chunks.append((combined_list_path, combined_results_path, num_unique_lines, "combined"))
    
    # Create script to split results
    split_script_path = os.path.join(TEMP_SCRIPTS_DIR, "split_results.py")
    
    # Serialize the mapping for the split script
    import json
    mapping_json = json.dumps(line_to_image)
    
    split_script_content = f'''#!/usr/bin/env python3
"""Split combined HTR results back to individual files."""
import sys
import os
import json
import re

COMBINED_RESULTS = '{combined_results_path}'
LINE_TO_IMAGE = {mapping_json}

def normalize_path(path):
    """Normalize path for comparison (resolve absolute paths, remove trailing slashes)."""
    return os.path.normpath(os.path.abspath(path))

def main():
    # Read combined results
    if not os.path.exists(COMBINED_RESULTS):
        print(f"ERROR: Combined results file not found: {{COMBINED_RESULTS}}", file=sys.stderr)
        sys.exit(1)
    
    # Build normalized path mapping - try multiple path formats
    normalized_mapping = {{}}
    path_variants_mapping = {{}}  # Map all path variants to the same entry
    basename_to_paths = {{}}  # Map basename -> list of full paths (for disambiguation)
    
    for orig_path, (basename, htr_res) in LINE_TO_IMAGE.items():
        # Normalize to absolute path first
        try:
            abs_path = os.path.abspath(orig_path)
            norm_path = normalize_path(abs_path)
        except:
            abs_path = orig_path
            norm_path = orig_path
        
        # Store all variants for maximum matching flexibility
        normalized_mapping[orig_path] = (basename, htr_res, orig_path)
        normalized_mapping[abs_path] = (basename, htr_res, orig_path)
        normalized_mapping[norm_path] = (basename, htr_res, orig_path)
        path_variants_mapping[orig_path] = (basename, htr_res, orig_path)
        path_variants_mapping[abs_path] = (basename, htr_res, orig_path)
        path_variants_mapping[norm_path] = (basename, htr_res, orig_path)
        
        # Store basename mapping (for fallback matching)
        path_basename = os.path.basename(orig_path)
        if path_basename not in basename_to_paths:
            basename_to_paths[path_basename] = []
        basename_to_paths[path_basename].append((norm_path, basename, htr_res))
    
    # Group results by target file
    results_by_file = {{}}
    for htr_res in set(htr_res for _, htr_res in LINE_TO_IMAGE.values()):
        results_by_file[htr_res] = []
    
    # Parse combined results and group by target file
    # Filter out stderr lines (lines that don't start with a path)
    total_lines = 0
    matched_lines = 0
    unmatched_samples = []  # Store first few unmatched for debugging
    
    with open(COMBINED_RESULTS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that look like stderr (don't start with / or contain common error patterns)
            if line.startswith('ERROR') or line.startswith('Warning') or line.startswith('Traceback'):
                continue
            if 'File "' in line or ('line ' in line and 'in ' in line):
                continue  # Python traceback lines
            if line.startswith('[') and 'INFO' in line:
                continue  # Log messages
            if 'Decoding:' in line or 'it/s' in line:
                continue  # Progress messages
            
            # Parse: path ['conf1', 'conf2', ...] 'text with spaces'
            # Path is first token, then confidence array, then quoted text
            # Example: /path/to/image.png ['0.53', '0.96'] ' c h a r <space> t e x t '
            # IMPORTANT: Paths can contain spaces (e.g., "CP 40-559 055-a"), so we can't just split on space
            
            # Find the path (first token, should start with /)
            if not line.startswith('/'):
                continue  # Skip lines that don't start with a path
            
            # Extract path - it ends with .png (or .jpg, etc.)
            # Paths can contain spaces, so we need to find the extension and extract up to it
            # Find the image extension position
            ext_pos = -1
            ext_len = 0
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                pos = line.find(ext)
                if pos > ext_pos:
                    ext_pos = pos
                    ext_len = len(ext)
            
            if ext_pos == -1:
                # No extension found - try to find bracket as fallback
                bracket_pos = line.find(" ['")
                if bracket_pos > 0:
                    line_path_raw = line[:bracket_pos].strip()
                else:
                    continue  # Skip if we can't find path end
            else:
                # Found extension - extract everything up to and including the extension
                # The path ends right after the extension (before space and bracket)
                ext_end = ext_pos + ext_len
                # Find the next space or bracket after the extension
                # Format is: path.png ['confidences'] or path.png 'text'
                next_space = line.find(' ', ext_end)
                next_bracket = line.find('[', ext_end)
                
                if next_bracket > 0 and (next_space == -1 or next_bracket < next_space):
                    # Bracket comes first (confidence scores present)
                    line_path_raw = line[:next_bracket].strip()
                elif next_space > 0:
                    # Space comes first (or only space, no bracket)
                    line_path_raw = line[:next_space].strip()
                else:
                    # No space or bracket found (shouldn't happen, but extract to extension)
                    line_path_raw = line[:ext_end].strip()
            
            # Ensure it's an absolute path
            if not os.path.isabs(line_path_raw):
                line_path_raw = os.path.abspath(line_path_raw)
            
            total_lines += 1
            
            # Try multiple matching strategies
            matched = False
            
            # Strategy 1: Direct match
            if line_path_raw in normalized_mapping:
                basename, htr_res, orig_path = normalized_mapping[line_path_raw]
                results_by_file[htr_res].append(line)
                matched_lines += 1
                matched = True
            
            # Strategy 2: Normalized match
            if not matched:
                try:
                    line_path_norm = normalize_path(line_path_raw)
                    if line_path_norm in normalized_mapping:
                        basename, htr_res, orig_path = normalized_mapping[line_path_norm]
                        results_by_file[htr_res].append(line)
                        matched_lines += 1
                        matched = True
                except:
                    pass
            
            # Strategy 3: Match by basename (filename only)
            if not matched:
                line_basename = os.path.basename(line_path_raw)
                if line_basename in path_variants_mapping:
                    basename, htr_res, orig_path = path_variants_mapping[line_basename]
                    # Verify it's the same file by checking if basenames match
                    orig_basename = os.path.basename(orig_path)
                    if line_basename == orig_basename:
                        results_by_file[htr_res].append(line)
                        matched_lines += 1
                        matched = True
            
            # Strategy 4: Try matching by basename (if unique)
            if not matched:
                line_basename = os.path.basename(line_path_raw)
                if line_basename in basename_to_paths:
                    candidates = basename_to_paths[line_basename]
                    if len(candidates) == 1:
                        # Unique match by basename
                        orig_path_check, basename_check, htr_res_check = candidates[0]
                        results_by_file[htr_res_check].append(line)
                        matched_lines += 1
                        matched = True
                    else:
                        # Multiple candidates - try to match by full path
                        for orig_path_check, basename_check, htr_res_check in candidates:
                            try:
                                if normalize_path(line_path_raw) == normalize_path(orig_path_check):
                                    results_by_file[htr_res_check].append(line)
                                    matched_lines += 1
                                    matched = True
                                    break
                            except:
                                pass
            
            if not matched:
                # Store sample unmatched for debugging
                if len(unmatched_samples) < 5:
                    unmatched_samples.append(line_path_raw[:100])
    
    # Write results to individual files
    files_written = 0
    total_results = 0
    for htr_res, lines in results_by_file.items():
        if lines:
            os.makedirs(os.path.dirname(htr_res), exist_ok=True)
            with open(htr_res, 'w', encoding='utf-8') as f:
                for result_line in lines:
                    f.write(f"{{result_line}}\\n")
            files_written += 1
            total_results += len(lines)
    
    print(f"Split {{matched_lines}} results to {{files_written}} files ({{total_lines}} total lines processed)")
    if matched_lines < total_lines:
        print(f"Warning: {{total_lines - matched_lines}} lines could not be matched to images", file=sys.stderr)
        if unmatched_samples:
            print(f"Sample unmatched paths (first 5):", file=sys.stderr)
            for sample in unmatched_samples:
                print(f"  {{sample}}", file=sys.stderr)
        # Debug: show some sample paths from mapping
        sample_mapping = list(normalized_mapping.keys())[:3]
        print(f"Sample paths in mapping (first 3):", file=sys.stderr)
        for sample in sample_mapping:
            print(f"  {{sample}}", file=sys.stderr)

if __name__ == '__main__':
    main()
'''
    
    with open(split_script_path, 'w', encoding='utf-8') as f:
        f.write(split_script_content)
    
    os.chmod(split_script_path, 0o755)
    
    # Create decode script
    decode_script_path = os.path.join(TEMP_SCRIPTS_DIR, "combined_decode.sh")
    
    # Build chunk processing commands
    chunk_commands = []
    chunk_results_list = []
    for chunk_idx, (chunk_list, chunk_results, chunk_size_val, basename) in enumerate(chunks, 1):
        chunk_results_list.append(chunk_results)
        display_name = basename if len(basename) < 50 else basename[:47] + "..."
        chunk_commands.append(f'''
echo "[Chunk {chunk_idx}/{len(chunks)}] Processing {display_name} ({chunk_size_val} lines)..."
start_chunk=$(date +%s)

# Redirect stdout to results, stderr to separate file
pylaia-htr-decode-ctc \\
    --trainer.accelerator gpu \\
    --trainer.devices 1 \\
    --common.checkpoint '{checkpoint}' \\
    --common.model_filename '{model_file}' \\
    --decode.include_img_ids true \\
    --decode.print_word_confidence_score true \\
    '{syms_file}' '{chunk_list}' > '{chunk_results}' 2>'{chunk_results}.stderr'
    
if [ $? -eq 0 ]; then
    
    end_chunk=$(date +%s)
    chunk_duration=$((end_chunk - start_chunk))
    echo "  ✓ Chunk {chunk_idx} completed in $chunk_duration seconds"
else
    echo "  ✗ Chunk {chunk_idx} failed ({display_name})" >&2
    echo "  Error output:" >&2
    if [ -f '{chunk_results}.stderr' ]; then
        tail -20 '{chunk_results}.stderr' >&2 || true
    else
        tail -20 '{chunk_results}' >&2 || true
    fi
    exit 1
fi
''')
    
    chunk_results_cat = ' '.join([f"'{cr}'" for cr in chunk_results_list])
    
    decode_script_content = f'''#!/bin/bash
# Combined decode script - processes all images in chunks
source {PYLAIA_ENV}

export PYTHONUNBUFFERED=1

echo "Processing {num_unique_lines} line images in {len(chunks)} chunk(s)..."
echo ""

start_time=$(date +%s)

{''.join(chunk_commands)}

    # Combine all chunk results (append, not overwrite)
echo ""
echo "Combining chunk results..."
# Clear combined results file first
> '{combined_results_path}'
# Append each chunk's results
cat {chunk_results_cat} >> '{combined_results_path}'

if [ $? -eq 0 ]; then
    echo "✓ All chunks processed successfully"
    echo ""
    echo "Splitting results to individual files..."
    
    # Split results
    python3 '{split_script_path}'
    
    if [ $? -eq 0 ]; then
        end_time=$(date +%s)
        total_duration=$((end_time - start_time))
        echo "✓ Results split successfully (Total time: $total_duration seconds)"
    else
        echo "✗ Error splitting results" >&2
        exit 1
    fi
else
    echo "✗ Error combining chunk results" >&2
    exit 1
fi
'''
    
    with open(decode_script_path, 'w', encoding='utf-8') as f:
        f.write(decode_script_content)
    
    os.chmod(decode_script_path, 0o755)
    
    return decode_script_path, combined_list_path


def create_batch_decode_script(work_items: List[Tuple[str, str, str]], 
                                checkpoint: str, model_file: str, syms_file: str) -> str:
    """
    Create a batch script that processes all images in a single environment activation.
    
    Args:
        work_items: List of (basename, list_txt_path, htr_res_path) tuples
        checkpoint: Path to PyLaia checkpoint
        model_file: Path to PyLaia model architecture file
        syms_file: Path to symbols file
    
    Returns:
        Path to the batch script
    """
    os.makedirs(TEMP_SCRIPTS_DIR, exist_ok=True)
    script_path = os.path.join(TEMP_SCRIPTS_DIR, "batch_decode.sh")
    
    total = len(work_items)
    script_content = f'''#!/bin/bash
# Batch PyLaia decode script - processes all images in one environment activation
source {PYLAIA_ENV}

# Ensure unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

total={total}
current=0
start_time=$(date +%s)

echo "Starting batch decode of $total images..."
echo ""

'''
    
    for idx, (basename, list_txt, htr_res) in enumerate(work_items, 1):
        script_content += f'''current={idx}
echo "[$current/$total] Processing: {basename}"
start_img=$(date +%s)

if ! pylaia-htr-decode-ctc \\
    --trainer.accelerator gpu \\
    --trainer.devices 1 \\
    --common.checkpoint '{checkpoint}' \\
    --common.model_filename '{model_file}' \\
    --decode.include_img_ids true \\
    --decode.print_word_confidence_score true \\
    '{syms_file}' '{list_txt}' > '{htr_res}' 2>&1; then
    echo "  ERROR: Failed to process {basename}" >&2
else
    end_img=$(date +%s)
    img_duration=$((end_img - start_img))
    elapsed=$((end_img - start_time))
    if [ $current -gt 0 ]; then
        remaining=$(( (elapsed * (total - current)) / current ))
        echo "  ✓ Completed in $img_duration s (Total: $elapsed s, Est. remaining: $remaining s)"
    else
        echo "  ✓ Completed in $img_duration s (Total: $elapsed s)"
    fi
fi
echo ""

'''
    
    script_content += f'''end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "Batch decode completed in $total_duration seconds"
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def parse_htr_results(htr_txt_path: str) -> Dict[str, str]:
    """
    Parse HTR results file and return a mapping of line_id -> htr_text.
    
    The htr.txt format is:
    /path/to/lines/uuid.png ['conf1', 'conf2', ...] c h a r <space> c h a r ...
    """
    results = {}
    
    if not os.path.exists(htr_txt_path) or os.path.getsize(htr_txt_path) == 0:
        return results
    
    with open(htr_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line - format: path [confidences] characters
            # Example: /path/to/uuid.png ['0.93', '0.87'] c h a r <space> t e x t
            
            # Extract line_id from path
            parts = line.split()
            if not parts:
                continue
            
            path = parts[0]
            # Extract UUID from filename
            basename = os.path.basename(path)
            line_id = os.path.splitext(basename)[0]
            
            # Find where the confidence array ends
            # Look for the pattern: [' ... ']
            bracket_start = line.find("[")
            bracket_end = line.find("]")
            
            if bracket_start != -1 and bracket_end != -1:
                # Text starts after the closing bracket
                text_part = line[bracket_end + 1:].strip()
            else:
                # No confidence scores, text is everything after path
                text_part = ' '.join(parts[1:])
            
            # The text is space-separated characters with <space> for actual spaces
            results[line_id] = text_part
    
    return results


def update_metadata_json(basename: str, htr_results: Dict[str, str]) -> bool:
    """
    Update the metadata.json file in corrected_lines with new HTR results.
    
    Args:
        basename: The image basename (e.g., "CP 40-559 055-a")
        htr_results: Dictionary mapping line_id to htr_text
    
    Returns:
        True if update was successful, False otherwise
    """
    metadata_dir = os.path.join(CORRECTED_LINES_DIR, basename)
    metadata_path = os.path.join(metadata_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"  No metadata.json found for {basename}, skipping update")
        return False
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  Error reading metadata.json for {basename}: {e}")
        return False
    
    # Update htr_text for each line
    lines = metadata.get("lines", [])
    updated_count = 0
    
    for line in lines:
        line_id = line.get("line_id")
        if line_id and line_id in htr_results:
            line["htr_text"] = htr_results[line_id]
            updated_count += 1
    
    # Write back
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  Error writing metadata.json for {basename}: {e}")
        return False
    
    if updated_count > 0:
        print(f"  Updated {updated_count}/{len(lines)} lines in metadata.json")
    
    return True


def find_image_for_basename(basename: str) -> Optional[str]:
    """Find the full image path for a given basename."""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(INPUT_IMAGES_DIR, basename + ext)
        if os.path.exists(path):
            return path
    return None


def process_image_prep(basename: str, skip_preprocessing: bool = False) -> Optional[Tuple[str, str, str]]:
    """
    Prepare a single image for processing (extract lines if needed).
    
    Returns:
        Tuple of (basename, list_txt_path, htr_res_path) if ready, None otherwise
    """
    work_dir = os.path.join(HTR_WORK_DIR, basename)
    kraken_json = os.path.join(work_dir, "kraken.json")
    lines_dir = os.path.join(work_dir, "lines")
    list_txt = os.path.join(work_dir, "img_list.txt")
    htr_res = os.path.join(work_dir, "htr.txt")
    
    # Check kraken.json exists
    if not os.path.exists(kraken_json):
        print(f"  No kraken.json found for {basename}")
        return None
    
    # Check if we can skip preprocessing
    skip_preprocess = skip_preprocessing and os.path.exists(lines_dir) and os.path.exists(list_txt)
    if skip_preprocess:
        # Verify that lines directory has images and list_txt is valid
        line_images = [f for f in os.listdir(lines_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if line_images and os.path.getsize(list_txt) > 0:
            # Verify img_list.txt matches existing line images
            try:
                with open(list_txt, 'r') as f:
                    list_paths = [line.strip() for line in f if line.strip()]
                if list_paths:
                    print(f"  Skipping preprocessing - {len(line_images)} line images already exist")
                    skip_preprocess = True
                else:
                    skip_preprocess = False
            except:
                skip_preprocess = False
        else:
            skip_preprocess = False
    
    if not skip_preprocess:
        # Find the original image
        image_path = find_image_for_basename(basename)
        if not image_path:
            # Try to get image path from kraken.json
            try:
                with open(kraken_json, 'r') as f:
                    kraken_data = json.load(f)
                    image_path = kraken_data.get("imagename")
            except:
                pass
        
        if not image_path or not os.path.exists(image_path):
            print(f"  Cannot find image for {basename}")
            return None
        
        # Remove existing lines directory and recreate
        if os.path.exists(lines_dir):
            shutil.rmtree(lines_dir)
        os.makedirs(lines_dir, exist_ok=True)
        
        # Create and run preprocessing script
        preprocess_script = create_preprocessing_script(
            image_path, kraken_json, lines_dir, list_txt, BOOTSTRAP_IMAGE_HEIGHT
        )
        
        cmd_preprocess = f"source {PYLAIA_ENV} && python3 '{preprocess_script}'"
        if not run_command(cmd_preprocess, "Preprocess Lines"):
            # Cleanup script
            if os.path.exists(preprocess_script):
                os.remove(preprocess_script)
            return None
        
        # Cleanup preprocessing script
        if os.path.exists(preprocess_script):
            os.remove(preprocess_script)
    
    # Check if we have lines to process
    if not os.path.exists(list_txt) or os.path.getsize(list_txt) == 0:
        print(f"  No lines extracted for {basename}")
        return None
    
    return (basename, list_txt, htr_res)


def process_image(basename: str, checkpoint: str, model_file: str, syms_file: str, 
                  skip_preprocessing: bool = False) -> bool:
    """
    Process a single image: regenerate line images and run HTR.
    (Legacy function - kept for backward compatibility)
    
    Args:
        basename: Image basename (e.g., "CP 40-559 055-a")
        checkpoint: Path to PyLaia checkpoint
        model_file: Path to PyLaia model architecture file
        syms_file: Path to symbols file
        skip_preprocessing: If True, skip line image extraction if lines already exist
    
    Returns:
        True if processing was successful
    """
    # Prepare the image
    prep_result = process_image_prep(basename, skip_preprocessing)
    if prep_result is None:
        return False
    
    basename, list_txt, htr_res = prep_result
    
    # Run PyLaia decode
    cmd_decode = (
        f"source {PYLAIA_ENV} && "
        f"pylaia-htr-decode-ctc "
        f"--trainer.accelerator gpu "
        f"--trainer.devices 1 "
        f"--common.checkpoint '{checkpoint}' "
        f"--common.model_filename '{model_file}' "
        f"--decode.include_img_ids true "
        f"--decode.print_word_confidence_score true "
        f"'{syms_file}' '{list_txt}' > '{htr_res}'"
    )
    
    if not run_command(cmd_decode, "PyLaia Decode"):
        return False
    
    # Parse HTR results and update metadata.json
    htr_results = parse_htr_results(htr_res)
    if htr_results:
        update_metadata_json(basename, htr_results)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate HTR results using the latest bootstrap PyLaia model"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip line image extraction if lines already exist (speeds up processing)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in a single batch (faster, activates environment once)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("HTR Regeneration Script - Using Latest Bootstrap Model")
    if args.skip_preprocessing:
        print("Mode: Skipping preprocessing if line images exist")
    if args.batch:
        print("Mode: Batch processing (single environment activation)")
    print("=" * 70)
    print()
    
    # Find the latest model
    try:
        checkpoint, model_file, syms_file, version = find_latest_model()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print()
    
    # Get all work directories
    work_dirs = get_htr_work_dirs()
    print(f"Found {len(work_dirs)} images with existing HTR work")
    print()
    
    # Initialize counters
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    if args.batch:
        # Batch mode: prepare all images, then process in one go
        print("Preparing images for batch processing...")
        work_items = []
        prep_success = 0
        prep_fail = 0
        
        for basename in tqdm(work_dirs, desc="Preparing images"):
            prep_result = process_image_prep(basename, args.skip_preprocessing)
            if prep_result:
                work_items.append(prep_result)
                prep_success += 1
            else:
                prep_fail += 1
        
        print(f"\nPrepared {prep_success} images, {prep_fail} failed")
        
        if work_items:
            print(f"\nRunning batch PyLaia decode on {len(work_items)} images...")
            # Try Python script first (keeps model in memory, eliminates 2-3s loading overhead)
            batch_script = create_python_batch_decode_script(work_items, checkpoint, model_file, syms_file)
            
            # Run the Python batch script with real-time output streaming
            print()  # Add blank line before batch output
            cmd = f"source {PYLAIA_ENV} && python3 '{batch_script}'"
            python_success = run_command_streaming(cmd, "Batch PyLaia Decode (Python API)")
            
            # Cleanup Python script
            if os.path.exists(batch_script):
                os.remove(batch_script)
            
            # If Python API failed, fall back to combined CLI processing (single call for all images)
            if not python_success:
                print("\nPython API approach failed, falling back to combined CLI processing...")
                print("(Validating image heights and processing in large batches)")
                decode_script, combined_list = create_combined_decode_script(work_items, checkpoint, model_file, syms_file)
                
                if run_command_streaming(f"bash '{decode_script}'", "Combined PyLaia Decode (CLI)"):
                    print("\nBatch decode completed successfully (using combined CLI)")
                else:
                    print("\nBatch decode had some errors (check output above)")
                
                # Cleanup scripts and temp files (but keep chunk results until after splitting)
                cleanup_paths = [
                    decode_script, 
                    combined_list,
                    os.path.join(TEMP_SCRIPTS_DIR, "split_results.py")
                ]
                # Add chunk list files (but NOT result files - they're needed for splitting)
                import glob
                cleanup_paths.extend(glob.glob(os.path.join(TEMP_SCRIPTS_DIR, "combined_img_list_chunk_*.txt")))
                # Note: Don't delete chunk result files here - they're needed for the split script
                
                for path in cleanup_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass  # Ignore cleanup errors
                
                # Keep chunk result files for debugging - don't delete them
                # They can be manually cleaned up later if needed
                # chunk_result_files = glob.glob(os.path.join(TEMP_SCRIPTS_DIR, "combined_htr_results_chunk_*.txt"))
                # chunk_stderr_files = glob.glob(os.path.join(TEMP_SCRIPTS_DIR, "combined_htr_results_chunk_*.txt.stderr"))
                # for path in chunk_result_files + chunk_stderr_files:
                #     if os.path.exists(path):
                #         try:
                #             os.remove(path)
                #         except:
                #             pass
                
                # Keep combined results file for debugging as well
                # combined_results_file = os.path.join(TEMP_SCRIPTS_DIR, "combined_htr_results.txt")
                # if os.path.exists(combined_results_file):
                #     try:
                #         os.remove(combined_results_file)
                #     except:
                #         pass
            else:
                print("\nBatch decode completed successfully (using Python API)")
            
            # Parse results and update metadata for all images
            print("\nUpdating metadata files...")
            success_count = 0
            skipped_count = 0
            fail_count = prep_fail
            
            for basename, list_txt, htr_res in tqdm(work_items, desc="Updating metadata"):
                if os.path.exists(htr_res) and os.path.getsize(htr_res) > 0:
                    htr_results = parse_htr_results(htr_res)
                    if htr_results:
                        # Try to update metadata - returns False if metadata.json doesn't exist (which is okay)
                        updated = update_metadata_json(basename, htr_results)
                        if updated:
                            success_count += 1
                        else:
                            # Check if metadata.json exists - if not, it's a skip, not a failure
                            metadata_dir = os.path.join(CORRECTED_LINES_DIR, basename)
                            metadata_path = os.path.join(metadata_dir, "metadata.json")
                            if not os.path.exists(metadata_path):
                                skipped_count += 1  # No metadata.json - expected for some images
                            else:
                                fail_count += 1  # metadata.json exists but update failed
                    else:
                        # htr.txt exists but has no parseable results
                        fail_count += 1
                else:
                    # htr.txt doesn't exist or is empty
                    fail_count += 1
        else:
            print("No images ready for processing")
            success_count = 0
            fail_count = prep_fail
            skipped_count = 0
    else:
        # Sequential mode: process each image individually
        success_count = 0
        fail_count = 0
        skipped_count = 0
        
        for basename in tqdm(work_dirs, desc="Processing images"):
            tqdm.write(f"\nProcessing: {basename}")
            
            if process_image(basename, checkpoint, model_file, syms_file, args.skip_preprocessing):
                success_count += 1
            else:
                fail_count += 1
    
    print()
    print("=" * 70)
    if skipped_count > 0:
        print(f"Completed: {success_count} successful, {skipped_count} skipped (no metadata.json), {fail_count} failed")
    else:
        print(f"Completed: {success_count} successful, {fail_count} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()

