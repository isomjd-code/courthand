#!/usr/bin/env python3
"""
Script to re-run post-correction for a specific file.
This is useful when the LLM failed to extract entities.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly from post_correction module to avoid __init__.py imports
import importlib.util
post_correction_path = project_root / "workflow_manager" / "post_correction.py"
spec = importlib.util.spec_from_file_location("post_correction", post_correction_path)
post_correction = importlib.util.module_from_spec(spec)
sys.modules["post_correction"] = post_correction
spec.loader.exec_module(post_correction)

process_image_post_correction = post_correction.process_image_post_correction
NameDatabase = post_correction.NameDatabase
BayesianConfig = post_correction.BayesianConfig

def load_step1_data(step1_path):
    """Load lines from step1.json file."""
    with open(step1_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    for line in data.get('lines', []):
        lines.append({
            'htr_text': line.get('transcription', ''),
            'bbox': line.get('bbox'),
            'line_id': line.get('id', '')
        })
    
    return lines

def main():
    if len(sys.argv) < 2:
        print("Usage: python rerun_post_correction.py <path_to_post_correction.json>")
        print("Example: python rerun_post_correction.py cp40_processing/output/CP40-563_398/CP40-563\\ 398a.jpg_post_correction.json")
        sys.exit(1)
    
    post_corr_path = Path(sys.argv[1])
    if not post_corr_path.exists():
        print(f"Error: File not found: {post_corr_path}")
        sys.exit(1)
    
    # Get directory and image name
    out_dir = post_corr_path.parent
    image_name = post_corr_path.stem.replace('_post_correction', '')
    
    # Find step1.json
    step1_path = out_dir / f"{image_name}_step1.json"
    if not step1_path.exists():
        # Try alternate naming
        step1_path = out_dir / f"{image_name}.jpg_step1.json"
        if not step1_path.exists():
            print(f"Error: Could not find step1.json for {image_name}")
            print(f"  Looked for: {out_dir / f'{image_name}_step1.json'}")
            print(f"  And: {out_dir / f'{image_name}.jpg_step1.json'}")
            sys.exit(1)
    
    print(f"Loading step1 data from: {step1_path}")
    lines = load_step1_data(step1_path)
    print(f"  Found {len(lines)} lines")
    
    # Find line images directory
    basename = image_name.replace('.jpg', '')
    htr_work_dir = out_dir / basename
    line_images_dir = htr_work_dir / 'lines' if htr_work_dir.exists() else None
    
    if line_images_dir and line_images_dir.exists():
        print(f"  Line images directory: {line_images_dir}")
    else:
        print(f"  Warning: Line images directory not found at {htr_work_dir / 'lines'}")
        line_images_dir = None
    
    # Initialize name database and config
    db_path = project_root / 'cp40_records.db'
    config = BayesianConfig()
    name_db = NameDatabase(db_path, config)
    
    print(f"\nRe-running post-correction for: {image_name}")
    print("  This will call the LLM to extract entities...")
    
    # Process post-correction
    result = process_image_post_correction(
        lines=lines,
        image_name=image_name,
        name_db=name_db,
        config=config,
        out_dir=str(line_images_dir) if line_images_dir else None
    )
    
    # Count extracted entities
    total_forenames = sum(len(line.get('forenames', [])) for line in result['lines'])
    total_surnames = sum(len(line.get('surnames', [])) for line in result['lines'])
    total_placenames = sum(len(line.get('placenames', [])) for line in result['lines'])
    
    print(f"\nExtraction results:")
    print(f"  Forenames: {total_forenames}")
    print(f"  Surnames: {total_surnames}")
    print(f"  Placenames: {total_placenames}")
    
    if total_forenames == 0 and total_surnames == 0 and total_placenames == 0:
        print("\n  ⚠️  WARNING: No entities extracted! The LLM may have failed again.")
    else:
        print("\n  ✓ Entities successfully extracted!")
    
    # Save result
    output_path = post_corr_path
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == '__main__':
    main()

