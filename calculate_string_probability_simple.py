#!/usr/bin/env python3
"""
Calculate log probability of a string for a given line image using Pylaia HTR model.
Uses the same model loading approach as pylaia-htr-decode-ctc CLI tool.
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl

# Try to import from laia (which pylaia-htr-decode-ctc uses)
try:
    from laia.utils import SymbolsTable
    from laia.common.config import load_config
    from laia.common.model_loader import ModelLoader
    print("✓ Using laia utilities (same as CLI tool)")
except ImportError:
    print("Error: Could not import laia. Make sure you're in the pylaia environment.")
    print("Activate with: source ~/projects/pylaia-env/bin/activate")
    sys.exit(1)

from workflow_manager.settings import (
    PYLAIA_MODEL, PYLAIA_SYMS, PYLAIA_ARCH,
    OUTPUT_DIR
)


def find_master_record(image_name: str) -> str:
    """Find master_record.json for the given image name."""
    image_basename = os.path.splitext(os.path.basename(image_name))[0]
    
    # Search in OUTPUT_DIR for matching directories
    for group_id in os.listdir(OUTPUT_DIR):
        group_path = os.path.join(OUTPUT_DIR, group_id)
        if not os.path.isdir(group_path):
            continue
        
        for item in os.listdir(group_path):
            item_path = os.path.join(group_path, item)
            if os.path.isdir(item_path) and item.startswith(image_basename):
                master_record = os.path.join(item_path, "master_record.json")
                if os.path.exists(master_record):
                    return master_record
    
    raise FileNotFoundError(f"Could not find master_record.json for image: {image_name}")


def get_original_file_id(master_record_path: str, line_id: str) -> str:
    """Extract original_file_id for the given line_id from master_record.json."""
    import json
    
    with open(master_record_path, 'r') as f:
        data = json.load(f)
    
    for line in data.get('lines', []):
        if line.get('line_id') == line_id:
            return line.get('original_file_id')
    
    raise ValueError(f"Line ID '{line_id}' not found in master_record.json")


def find_line_image(image_name: str, line_id: str) -> str:
    """Find the line image file for the given image name and line ID."""
    master_record = find_master_record(image_name)
    original_file_id = get_original_file_id(master_record, line_id)
    
    # Construct the expected path
    image_basename = os.path.splitext(os.path.basename(image_name))[0]
    master_dir = os.path.dirname(master_record)
    lines_dir = os.path.join(master_dir, "lines")
    line_image = os.path.join(lines_dir, f"{original_file_id}.png")
    
    if os.path.exists(line_image):
        return line_image
    
    # Fallback: search recursively
    for root, dirs, files in os.walk(master_dir):
        for file in files:
            if file == f"{original_file_id}.png":
                return os.path.join(root, file)
    
    raise FileNotFoundError(f"Could not find line image: {original_file_id}.png")


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess image for Pylaia model (grayscale, resize to 128px height)."""
    img = Image.open(image_path).convert('L')  # Grayscale
    
    # Resize to fixed height of 128px, maintaining aspect ratio
    width, height = img.size
    new_height = 128
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to tensor: (1, 1, H, W)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def load_model_using_cli_method(checkpoint_path: str, model_arch_path: str):
    """
    Load model using the same method as pylaia-htr-decode-ctc CLI.
    The CLI calls laia.scripts.htr.decode_ctc.main, which uses laia's model loading.
    """
    # Method 1: Try importing the decode_ctc module's model loading function
    try:
        from laia.scripts.htr.decode_ctc import load_model_from_checkpoint
        model = load_model_from_checkpoint(checkpoint_path, model_filename=model_arch_path)
        print("✓ Loaded model using decode_ctc's load function")
        return model
    except (ImportError, AttributeError) as e:
        print(f"decode_ctc import failed: {e}")
    
    # Method 2: Try using laia's ModelLoader (if available)
    try:
        loader = ModelLoader(checkpoint_path, model_filename=model_arch_path)
        model = loader.load()
        print("✓ Loaded model using laia.common.model_loader")
        return model
    except (AttributeError, ImportError, Exception) as e:
        print(f"ModelLoader method failed: {e}")
    
    # Method 3: Use the same approach as decode_ctc script
    # Load model class from architecture file, then load checkpoint
    if os.path.exists(model_arch_path):
        try:
            import pickle
            with open(model_arch_path, 'rb') as f:
                model_class = pickle.load(f)
            print("✓ Loaded model class from architecture file")
            
            model = model_class.load_from_checkpoint(
                checkpoint_path,
                map_location='cpu',
                strict=False
            )
            print("✓ Loaded checkpoint into model class")
            return model
        except Exception as e:
            print(f"Could not load from arch file: {e}")
    
    # Method 4: Direct Lightning loading
    model = pl.LightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        strict=False
    )
    print("✓ Loaded model directly from checkpoint")
    return model


def extract_actual_model(model):
    """Extract the actual PyTorch model from Lightning wrapper."""
    # The model might be wrapped in a Lightning module
    # Try common attribute names
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'net'):
        return model.net
    elif hasattr(model, 'crnn'):
        return model.crnn
    elif hasattr(model, 'forward'):
        # Might already be the actual model
        return model
    else:
        raise ValueError(f"Cannot extract model from {type(model)}. Available attributes: {[a for a in dir(model) if not a.startswith('_')][:10]}")


def calculate_log_probability(model, image_tensor: torch.Tensor, text: str, syms: SymbolsTable) -> float:
    """Calculate log probability of text given image using CTC loss."""
    # Extract actual model from Lightning wrapper
    actual_model = extract_actual_model(model)
    actual_model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = actual_model(image_tensor)  # Shape: (Time, Batch, Classes) or (Batch, Time, Classes)
    
    # Ensure output is (Time, Batch, Classes) for CTC
    if output.dim() == 3:
        if output.size(0) == image_tensor.size(0):  # (Batch, Time, Classes)
            output = output.transpose(0, 1)  # -> (Time, Batch, Classes)
    
    # Apply log softmax
    log_probs = torch.nn.functional.log_softmax(output, dim=2)
    
    # Convert text to indices
    try:
        target_indices = [syms[c] for c in text]
    except KeyError as e:
        raise ValueError(f"Character '{e.args[0]}' not in symbol table. Available symbols: {list(syms.syms.keys())[:20]}...")
    
    target_tensor = torch.tensor([target_indices], dtype=torch.long)
    
    # Calculate CTC loss
    input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long)
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
    loss = ctc_loss_fn(log_probs, target_tensor, input_lengths, target_lengths)
    
    # Log probability = -loss
    log_prob = -loss.item()
    
    return log_prob


def main():
    parser = argparse.ArgumentParser(
        description="Calculate log probability of a string for a line image using Pylaia HTR"
    )
    parser.add_argument("image_name", help="Image name (e.g., CP40-565_481a.jpg)")
    parser.add_argument("line_id", help="Line ID (e.g., L02)")
    parser.add_argument("alternate_string", help="Alternate string to evaluate")
    parser.add_argument("--checkpoint", default=PYLAIA_MODEL, help="Path to model checkpoint")
    parser.add_argument("--model-arch", default=PYLAIA_ARCH, help="Path to model architecture file")
    parser.add_argument("--symbols", default=PYLAIA_SYMS, help="Path to symbols.txt")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Pylaia String Probability Calculator")
    print("=" * 70)
    print(f"Image: {args.image_name}")
    print(f"Line ID: {args.line_id}")
    print(f"String: '{args.alternate_string}'")
    print()
    
    # Load symbols
    print(f"Loading symbols from: {args.symbols}")
    syms = SymbolsTable(args.symbols)
    print(f"✓ Loaded {len(syms)} symbols")
    
    # Load model (using CLI's method)
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model_using_cli_method(args.checkpoint, args.model_arch)
    
    # Find line image
    print(f"\nFinding line image for {args.image_name} line {args.line_id}...")
    line_image_path = find_line_image(args.image_name, args.line_id)
    print(f"✓ Found: {line_image_path}")
    
    # Preprocess image
    print("\nPreprocessing image...")
    image_tensor = preprocess_image(line_image_path)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Calculate log probability
    print("\nCalculating log probability...")
    log_prob = calculate_log_probability(model, image_tensor, args.alternate_string, syms)
    probability = np.exp(log_prob)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"String: '{args.alternate_string}'")
    print(f"Log Probability: {log_prob:.6f}")
    print(f"Probability: {probability:.6e}")
    print("=" * 70)


if __name__ == "__main__":
    main()

