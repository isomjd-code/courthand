#!/usr/bin/env python3
"""Script to delete all .tex and .pdf files from subdirectories of cp40_processing/output."""

import os
from pathlib import Path
from typing import List


def find_tex_pdf_files(root_dir: Path) -> List[Path]:
    """Find all .tex and .pdf files in subdirectories of root_dir."""
    files_to_delete = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.tex', '.pdf')):
                file_path = Path(root) / file
                files_to_delete.append(file_path)
    
    return files_to_delete


def delete_files(files: List[Path]) -> tuple[int, int]:
    """Delete the specified files and return (successful, failed) counts."""
    successful = 0
    failed = 0
    
    for file_path in files:
        try:
            file_path.unlink()
            print(f"Deleted: {file_path}")
            successful += 1
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
            failed += 1
    
    return successful, failed


def main():
    """Main function to delete .tex and .pdf files."""
    # Target directory - use Linux path when running from WSL
    # Try Linux path first (for WSL), fall back to Windows path if needed
    linux_path = Path("/home/qj/projects/latin_bho/cp40_processing/output")
    windows_path = Path(r"\\wsl.localhost\Ubuntu\home\qj\projects\latin_bho\cp40_processing\output")
    script_dir = Path(__file__).resolve().parent
    relative_path = script_dir / "cp40_processing" / "output"
    
    # Use the path that exists
    if linux_path.exists():
        target_dir = linux_path
    elif windows_path.exists():
        target_dir = windows_path
    elif relative_path.exists():
        target_dir = relative_path
    else:
        target_dir = linux_path  # Default to Linux path for error message
    
    # Check if directory exists
    if not target_dir.exists():
        print(f"Error: Directory does not exist: {target_dir}")
        print(f"Tried paths:")
        print(f"  - {linux_path}")
        print(f"  - {windows_path}")
        print(f"  - {relative_path}")
        return
    
    if not target_dir.is_dir():
        print(f"Error: Path is not a directory: {target_dir}")
        return
    
    print(f"Searching for .tex and .pdf files in: {target_dir}")
    print("-" * 80)
    
    # Find all .tex and .pdf files
    files_to_delete = find_tex_pdf_files(target_dir)
    
    if not files_to_delete:
        print("No .tex or .pdf files found.")
        return
    
    print(f"\nFound {len(files_to_delete)} file(s) to delete:")
    for file_path in files_to_delete:
        print(f"  - {file_path}")
    
    # Confirm deletion
    print("\n" + "-" * 80)
    response = input(f"\nDelete these {len(files_to_delete)} file(s)? (yes/no): ").strip().lower()
    
    if response not in ('yes', 'y'):
        print("Deletion cancelled.")
        return
    
    # Delete files
    print("\nDeleting files...")
    print("-" * 80)
    successful, failed = delete_files(files_to_delete)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {successful} file(s) deleted successfully, {failed} file(s) failed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

