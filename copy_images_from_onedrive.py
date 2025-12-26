#!/usr/bin/env python3
"""
Script to copy images from OneDrive subdirectories to input_images folder.
Only copies images that don't already exist in the destination.
"""

import os
import shutil
from pathlib import Path
from typing import Set

def normalize_path(path: str) -> str:
    """
    Convert Windows/WSL network paths to proper Linux paths when running in WSL.
    
    Handles:
    - Windows WSL paths (\\wsl.localhost\\Ubuntu\\...) -> /home/...
    - Windows paths with backslashes -> forward slashes
    - Already correct Linux paths (no change)
    """
    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')
    
    # Handle WSL network path format: \\wsl.localhost\Ubuntu\home\... -> /home/...
    if path.startswith('//wsl.localhost/Ubuntu/home/'):
        path = path.replace('//wsl.localhost/Ubuntu/home/', '/home/')
    elif path.startswith('/wsl.localhost/Ubuntu/home/'):
        path = path.replace('/wsl.localhost/Ubuntu/home/', '/home/')
    
    # Handle Windows drive paths if needed (e.g., C:/Users/...)
    # For now, just return the normalized path
    return os.path.normpath(path)

# Source directory (OneDrive) - will be normalized
# Using raw string (r"...") to avoid escape sequence issues
SOURCE_DIR_RAW = r"\\wsl.localhost\Ubuntu\home\qj\projects\OneDrive_2_11-21-2025"

# Destination directory - will be normalized
# Using raw string (r"...") to avoid escape sequence issues
DEST_DIR_RAW = r"\\wsl.localhost\Ubuntu\home\qj\projects\latin_bho\input_images"

# Normalize paths
SOURCE_DIR = normalize_path(SOURCE_DIR_RAW)
DEST_DIR = normalize_path(DEST_DIR_RAW)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}


def get_existing_files(dest_dir: str) -> Set[str]:
    """
    Get a set of all existing filenames in the destination directory.
    
    Args:
        dest_dir: Destination directory path
        
    Returns:
        Set of lowercase filenames (for case-insensitive comparison)
    """
    existing = set()
    if os.path.exists(dest_dir):
        for filename in os.listdir(dest_dir):
            existing.add(filename.lower())
    return existing


def find_images(source_dir: str) -> list[tuple[str, str]]:
    """
    Recursively find all image files in source directory and subdirectories.
    
    Args:
        source_dir: Source directory to search
        
    Returns:
        List of tuples: (source_path, filename)
    """
    images = []
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return images
    
    print(f"Scanning for images in: {source_dir}")
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = Path(file)
            ext = file_path.suffix.lower()
            
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                images.append((full_path, file))
    
    return images


def copy_images(source_dir: str, dest_dir: str, dry_run: bool = False) -> None:
    """
    Copy images from source to destination, skipping existing files.
    
    Args:
        source_dir: Source directory to search
        dest_dir: Destination directory
        dry_run: If True, only print what would be copied without actually copying
    """
    # Get list of existing files in destination
    existing_files = get_existing_files(dest_dir)
    print(f"Found {len(existing_files)} existing files in destination")
    
    # Find all images in source
    images = find_images(source_dir)
    print(f"Found {len(images)} image files in source directory")
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)
    
    # Copy images
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for source_path, filename in images:
        dest_path = os.path.join(dest_dir, filename)
        
        # Check if file already exists (case-insensitive)
        if filename.lower() in existing_files:
            print(f"  SKIP: {filename} (already exists)")
            skipped_count += 1
            continue
        
        # Check if file exists with exact case
        if os.path.exists(dest_path):
            print(f"  SKIP: {filename} (already exists)")
            skipped_count += 1
            continue
        
        try:
            if dry_run:
                print(f"  WOULD COPY: {filename}")
                print(f"    From: {source_path}")
                print(f"    To: {dest_path}")
            else:
                print(f"  COPYING: {filename}")
                shutil.copy2(source_path, dest_path)
                print(f"    ✓ Copied successfully")
                copied_count += 1
        except Exception as e:
            print(f"  ERROR copying {filename}: {e}")
            error_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images found: {len(images)}")
    if dry_run:
        print(f"Would copy: {len(images) - skipped_count - error_count}")
    else:
        print(f"Copied: {copied_count}")
    print(f"Skipped (already exists): {skipped_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print("="*60)


def main():
    """Main function."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Copy images from OneDrive subdirectories to input_images folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python copy_images_from_onedrive.py
  
  # Specify custom source and destination
  python copy_images_from_onedrive.py --source /home/qj/projects/OneDrive_2_11-21-2025 --dest /home/qj/projects/latin_bho/input_images
  
  # Dry run (preview without copying)
  python copy_images_from_onedrive.py --dry-run
        """
    )
    parser.add_argument('--source', '-s', type=str, default=None,
                        help='Source directory path (default: from script)')
    parser.add_argument('--dest', '-d', type=str, default=None,
                        help='Destination directory path (default: from script)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview what would be copied without actually copying')
    
    args = parser.parse_args()
    
    # Use provided paths or defaults
    source_dir = args.source if args.source else SOURCE_DIR
    dest_dir = args.dest if args.dest else DEST_DIR
    
    # Normalize paths if they weren't provided as arguments
    if not args.source:
        source_dir = normalize_path(source_dir)
    if not args.dest:
        dest_dir = normalize_path(dest_dir)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied\n")
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print()
    
    # Verify paths exist
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory does not exist: {source_dir}")
        print(f"\nTroubleshooting:")
        print(f"1. Check if the path is correct")
        print(f"2. If running from WSL, use Linux paths like: /home/qj/projects/OneDrive_2_11-21-2025")
        print(f"3. Try specifying the path as an argument:")
        print(f"   python copy_images_from_onedrive.py --source /home/qj/projects/OneDrive_2_11-21-2025")
        
        # Suggest alternative paths
        alt_paths = [
            f"/home/qj/projects/OneDrive_2_11-21-2025",
            f"~/projects/OneDrive_2_11-21-2025",
        ]
        print(f"\n4. Try these alternative paths:")
        for alt in alt_paths:
            expanded = os.path.expanduser(alt)
            exists = os.path.exists(expanded)
            status = "✓ EXISTS" if exists else "✗ not found"
            print(f"   {alt} -> {expanded} [{status}]")
        return
    
    if not os.path.exists(os.path.dirname(dest_dir)):
        print(f"WARNING: Destination parent directory does not exist: {os.path.dirname(dest_dir)}")
        print(f"Will attempt to create destination directory: {dest_dir}")
    print()
    
    # Confirm before proceeding (unless dry-run)
    if not args.dry_run:
        response = input("Proceed with copying? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    copy_images(source_dir, dest_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

