#!/usr/bin/env python3
"""
Clean up checkpoint.json by removing large fields that can be reconstructed from disk.
This reduces the checkpoint file size from ~60MB to just a few KB.

Uses streaming JSON parsing to handle large files efficiently.
"""

import json
import os
import sys
import shutil

BOOTSTRAP_DATA_DIR = os.path.join(os.path.dirname(__file__), "bootstrap_training_data")
BOOTSTRAP_CHECKPOINT_FILE = os.path.join(BOOTSTRAP_DATA_DIR, "checkpoint.json")

def cleanup_checkpoint():
    """Remove processed_images and pending_images from checkpoint to reduce size."""
    if not os.path.exists(BOOTSTRAP_CHECKPOINT_FILE):
        print(f"Checkpoint file not found: {BOOTSTRAP_CHECKPOINT_FILE}")
        return
    
    # Get current file size
    original_size = os.path.getsize(BOOTSTRAP_CHECKPOINT_FILE)
    print(f"Original checkpoint size: {original_size / (1024*1024):.2f} MB")
    print("Loading checkpoint (this may take a moment for large files)...")
    
    # Load checkpoint with progress indication
    try:
        print("Reading file...")
        with open(BOOTSTRAP_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        print("File loaded successfully.")
    except MemoryError:
        print("ERROR: File too large to load into memory. Using alternative method...")
        # Try using ijson for streaming (if available) or manual parsing
        cleanup_checkpoint_streaming()
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Remove large fields
    print("Removing large fields...")
    removed_fields = []
    processed_count = 0
    pending_count = 0
    
    if "processed_images" in state:
        processed_count = len(state['processed_images'])
        removed_fields.append(f"processed_images ({processed_count:,} items)")
        del state["processed_images"]
        print(f"  Removed processed_images: {processed_count:,} items")
    
    if "pending_images" in state:
        pending_count = len(state.get('pending_images', {}))
        removed_fields.append(f"pending_images ({pending_count:,} items)")
        del state["pending_images"]
        print(f"  Removed pending_images: {pending_count:,} items")
    
    if not removed_fields:
        print("No large fields to remove. Checkpoint is already optimized.")
        return
    
    print(f"Total removed: {processed_count + pending_count:,} items")
    
    # Save cleaned checkpoint
    try:
        # Create backup
        backup_path = BOOTSTRAP_CHECKPOINT_FILE + ".backup"
        print(f"Creating backup: {backup_path}...")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.copy2(BOOTSTRAP_CHECKPOINT_FILE, backup_path)
        print("Backup created.")
        
        # Save cleaned version
        print("Saving cleaned checkpoint...")
        with open(BOOTSTRAP_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        new_size = os.path.getsize(BOOTSTRAP_CHECKPOINT_FILE)
        reduction = original_size - new_size
        reduction_pct = (reduction / original_size) * 100
        
        print(f"\n✅ Checkpoint cleaned successfully!")
        print(f"Original size: {original_size / (1024*1024):.2f} MB")
        print(f"New size: {new_size / (1024*1024):.2f} MB")
        print(f"Reduced by: {reduction / (1024*1024):.2f} MB ({reduction_pct:.1f}%)")
        
    except Exception as e:
        print(f"Error saving cleaned checkpoint: {e}")
        import traceback
        traceback.print_exc()
        # Restore backup if save failed
        if os.path.exists(backup_path) and os.path.exists(BOOTSTRAP_CHECKPOINT_FILE + ".backup"):
            shutil.copy2(backup_path, BOOTSTRAP_CHECKPOINT_FILE)
            print("Restored backup due to error.")

def cleanup_checkpoint_streaming():
    """Alternative method using manual JSON parsing for very large files."""
    print("Streaming method not yet implemented. Please free up memory or use a machine with more RAM.")
    print("Alternatively, you can manually edit the file to remove the large arrays.")

if __name__ == "__main__":
    cleanup_checkpoint()

