"""Image grouping utilities."""

from __future__ import annotations

import os
import re
from typing import Dict, List

from .settings import OUTPUT_DIR, logger


class ImageGrouper:
    def __init__(self, directory: str, output_directory: str | None = None) -> None:
        self.directory = directory
        self.output_directory = output_directory
        self.groups: Dict[str, List[str]] = {}

    def scan(self) -> Dict[str, List[str]]:
        """
        Scan directory and group images by case identifier.

        Groups images based on filename patterns:
        - Extracts case identifier from filename (e.g., "CP40-565_481" from "CP40-565_481a.jpg")
        - Handles special "d" or "D" suffix for dorse images (e.g., "268d" or "268-d" -> "268d")
        - Prioritizes incomplete cases if output_directory is provided

        Returns:
            Dictionary mapping group IDs to lists of full image file paths.
            Group IDs are derived from filenames (e.g., "CP40-565_481").
            Returns empty dict if directory not found or no images present.
        """
        logger.info(f"Scanning directory: {self.directory}")
        pattern = re.compile(r"^(.*\d)", re.IGNORECASE)
        try:
            files = sorted(
                [
                    f
                    for f in os.listdir(self.directory)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
            )
        except FileNotFoundError:
            logger.error(f"Directory not found: {self.directory}")
            return {}

        if not files:
            logger.warning(f"No images (.jpg, .png) found in {self.directory}")
            return {}

        for filename in files:
            filename_no_ext = os.path.splitext(filename)[0]
            match = pattern.match(filename_no_ext)
            gid = match.group(1).strip() if match else filename_no_ext
            
            # Check for dorse suffix: "d" or "D" directly after the last digit
            # Handles both "268d" and "268-d" patterns
            if "d-" in filename.lower():
                gid = gid + "d"  # Use lowercase d for consistency
            elif match and len(filename_no_ext) > len(gid):
                # Check if there's a "d" or "D" immediately after the matched portion
                remaining = filename_no_ext[len(gid):].strip()
                if remaining and remaining[0].lower() == "d":
                    gid = gid + "d"  # Use lowercase d for consistency
            
            self.groups.setdefault(gid, []).append(os.path.join(self.directory, filename))

        if self.output_directory and os.path.exists(self.output_directory):
            logger.info(f"Scanning {self.output_directory} for incomplete cases to prioritize...")
            priority_groups: Dict[str, List[str]] = {}
            standard_groups: Dict[str, List[str]] = {}

            try:
                existing_output_folders = set(os.listdir(self.output_directory))
            except OSError:
                existing_output_folders = set()

            for gid, paths in self.groups.items():
                folder_name = gid.replace(" ", "_")
                is_priority = False

                if folder_name in existing_output_folders:
                    case_dir = os.path.join(self.output_directory, folder_name)
                    final_json = os.path.join(case_dir, "final_index.json")
                    if not os.path.exists(final_json):
                        is_priority = True

                if is_priority:
                    priority_groups[gid] = paths
                else:
                    standard_groups[gid] = paths

            if priority_groups:
                logger.info("--- PRIORITY QUEUE ---")
                logger.info(f"Found {len(priority_groups)} cases pending in output. Processing these first.")
                self.groups = {**priority_groups, **standard_groups}
            else:
                logger.info("No incomplete output cases found. Proceeding with standard order.")

        logger.info(f"Found {len(self.groups)} total groups to process.")
        return self.groups

