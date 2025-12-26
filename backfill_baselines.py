#!/usr/bin/env python3
"""
Backfill baseline information from Kraken JSON files into metadata.json files.

This script:
1. Finds all metadata.json files in corrected_lines directories
2. Locates corresponding Kraken JSON files
3. Extracts baseline information from Kraken JSON
4. Updates metadata.json files with baseline data
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from workflow_manager.settings import BASE_DIR, logger

# Bootstrap training paths
BOOTSTRAP_DATA_DIR = os.path.join(BASE_DIR, "bootstrap_training_data")
CORRECTED_LINES_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines")
HTR_WORK_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work")


def find_kraken_json(image_basename: str) -> Optional[str]:
    """
    Find Kraken JSON file for a given image basename.
    
    Args:
        image_basename: Basename of the image (without extension)
        
    Returns:
        Path to Kraken JSON file or None if not found
    """
    # Check in htr_work directory (where Kraken JSON is stored)
    kraken_json = os.path.join(HTR_WORK_DIR, image_basename, "kraken.json")
    if os.path.exists(kraken_json):
        return kraken_json
    
    # Try alternative locations
    alt_paths = [
        os.path.join(BOOTSTRAP_DATA_DIR, "htr_work", image_basename, "kraken.json"),
        os.path.join(BASE_DIR, "htr_work", image_basename, "kraken.json"),
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            return path
    
    return None


def extract_baseline_from_kraken_json(kraken_json_path: str, line_id: str) -> Optional[List[List[int]]]:
    """
    Extract baseline coordinates for a specific line from Kraken JSON.
    
    Args:
        kraken_json_path: Path to Kraken JSON file
        line_id: Line identifier to match
        
    Returns:
        List of baseline coordinates [[x1, y1], [x2, y2], ...] or None
    """
    try:
        with open(kraken_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lines = data.get("lines", [])
        for line in lines:
            if line.get("id") == line_id:
                baseline = line.get("baseline", [])
                if baseline and len(baseline) >= 2:
                    # Convert to list of lists format
                    return [[int(pt[0]), int(pt[1])] for pt in baseline]
    except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
        logger.debug(f"Error reading baseline from {kraken_json_path} for line {line_id}: {e}")
    
    return None


def update_metadata_with_baselines(metadata_path: str, dry_run: bool = False) -> Tuple[int, int]:
    """
    Update a metadata.json file with baseline information from Kraken JSON.
    
    Args:
        metadata_path: Path to metadata.json file
        dry_run: If True, don't actually write changes
        
    Returns:
        Tuple of (lines_updated, lines_not_found)
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {metadata_path}: {e}")
        return 0, 0
    
    image_basename = metadata.get("image_basename")
    if not image_basename:
        logger.warning(f"No image_basename in {metadata_path}")
        return 0, 0
    
    # Find Kraken JSON file
    kraken_json_path = find_kraken_json(image_basename)
    if not kraken_json_path:
        logger.debug(f"Kraken JSON not found for {image_basename}")
        return 0, 0
    
    lines_updated = 0
    lines_not_found = 0
    updated = False
    
    # Update each line with baseline information
    for line_data in metadata.get("lines", []):
        line_id = line_data.get("line_id")
        if not line_id:
            continue
        
        # Skip if baseline already exists
        if line_data.get("baseline") or line_data.get("baseline_coords"):
            continue
        
        # Extract baseline from Kraken JSON
        baseline_coords = extract_baseline_from_kraken_json(kraken_json_path, line_id)
        
        if baseline_coords:
            # Convert to string format (matching parse_kraken_json_for_processing)
            baseline_str = " ".join(f"{int(pt[0])},{int(pt[1])}" for pt in baseline_coords)
            
            line_data["baseline"] = baseline_str
            line_data["baseline_coords"] = baseline_coords
            lines_updated += 1
            updated = True
        else:
            lines_not_found += 1
    
    # Write updated metadata if changes were made
    if updated and not dry_run:
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated {metadata_path}: {lines_updated} lines with baselines, {lines_not_found} not found")
        except Exception as e:
            logger.error(f"Error writing {metadata_path}: {e}")
            return 0, 0
    
    return lines_updated, lines_not_found


def main():
    """Main entry point for baseline backfilling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Backfill baseline information from Kraken JSON into metadata files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(CORRECTED_LINES_DIR):
        logger.error(f"Corrected lines directory not found: {CORRECTED_LINES_DIR}")
        return 1
    
    logger.info(f"Scanning {CORRECTED_LINES_DIR} for metadata files...")
    
    total_updated = 0
    total_not_found = 0
    files_processed = 0
    files_updated = 0
    
    # Find all metadata.json files
    for subdir in os.listdir(CORRECTED_LINES_DIR):
        metadata_path = os.path.join(CORRECTED_LINES_DIR, subdir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        files_processed += 1
        if args.verbose:
            logger.info(f"Processing {subdir}...")
        
        lines_updated, lines_not_found = update_metadata_with_baselines(
            metadata_path,
            dry_run=args.dry_run
        )
        
        if lines_updated > 0:
            files_updated += 1
            total_updated += lines_updated
            total_not_found += lines_not_found
    
    logger.info("=" * 70)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files updated: {files_updated}")
    logger.info(f"Total lines updated with baselines: {total_updated}")
    logger.info(f"Total lines not found in Kraken JSON: {total_not_found}")
    
    if args.dry_run:
        logger.info("\n⚠️  DRY RUN - No changes were made. Run without --dry-run to apply changes.")
    else:
        logger.info(f"\n✅ Backfill complete! Updated {files_updated} files with {total_updated} baselines.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

