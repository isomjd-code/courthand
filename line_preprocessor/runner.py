"""CLI runner for line preprocessing."""

from __future__ import annotations

import os
import sys
from typing import List

from PIL import Image
from tqdm import tqdm

from .config import FINAL_LINE_HEIGHT, POLYGON_EXPANSION_PERCENT
from .geometry import expand_polygon
from .parser import parse_kraken_json_for_processing
from .processing import initial_line_extraction, process_line_image_dt_based


def _load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path.

    Args:
        image_path: Path to the image file to load.

    Returns:
        A PIL Image object.

    Raises:
        SystemExit: If the image file cannot be found or opened.
    """
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        print(f"FATAL: Source image not found at {image_path}", file=sys.stderr)
        sys.exit(1)


def _expand_polygons(lines: List[dict]) -> List[dict]:
    """
    Expand polygons for all lines using baseline-based expansion.

    Applies polygon expansion to each line in the list to ensure adequate
    context around text regions.

    Args:
        lines: List of line dictionaries, each containing "polygon" and "baseline" keys.

    Returns:
        A new list of line dictionaries with expanded polygons.
    """
    expanded = []
    for line in lines:
        expanded_polygon = expand_polygon(
            line["polygon"],
            line["baseline"],
            POLYGON_EXPANSION_PERCENT,
            min_padding=10,
        )
        line["polygon"] = expanded_polygon
        expanded.append(line)
    return expanded


def _save_line_image(image: Image.Image, output_dir: str, line_id: str) -> str:
    """
    Save a processed line image to disk.

    Args:
        image: The processed line image to save.
        output_dir: Directory where the image should be saved.
        line_id: Identifier used as the filename (without extension).

    Returns:
        The absolute path to the saved image file.
    """
    filename = f"{line_id}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return os.path.abspath(path)


def main(image_path: str, json_path: str, output_dir: str, pylaia_list_path: str) -> None:
    """
    Main entry point for line preprocessing pipeline.

    Processes a page image by extracting individual text lines based on Kraken
    segmentation data, normalizing each line, and saving them for PyLaia HTR.
    Creates a list file that PyLaia can use for batch processing.

    The pipeline:
    1. Loads the page image and Kraken JSON segmentation data
    2. Parses line boundaries and baselines from the JSON
    3. Expands polygons to include context around text
    4. For each line:
       - Extracts the line region from the page
       - Normalizes (rotates, rectifies, deslants, scales, binarizes)
       - Saves the processed line image
       - Adds the path to the PyLaia list file

    Args:
        image_path: Path to the source page image file.
        json_path: Path to the Kraken JSON file containing segmentation data.
        output_dir: Directory where processed line images will be saved.
        pylaia_list_path: Path where the PyLaia input list file will be written.
            This file contains one absolute path per line, one per line.

    Exits:
        SystemExit(1): If the source image cannot be loaded.
        SystemExit(0): If no text lines are found in the JSON (graceful exit).
    """
    print(f"Reading source image: {image_path}")
    page_image = _load_image(image_path)

    print(f"Reading Kraken JSON: {json_path}")
    lines_to_process = parse_kraken_json_for_processing(json_path)
    if not lines_to_process:
        print("No text lines found in the Kraken JSON. Exiting.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(lines_to_process)} lines to process. Expanding polygons...")
    expanded_lines = _expand_polygons(lines_to_process)

    print("Processing and saving line images...")
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
                final_image = process_line_image_dt_based(
                    line_rect_img,
                    line_polygon_coords,
                    line_baseline_points,
                    final_canvas_height=FINAL_LINE_HEIGHT,
                    line_id_for_debug=line_data["id"],
                )
                if final_image:
                    abs_path = _save_line_image(final_image, output_dir, line_data["id"])
                    pylaia_list_file.write(f"{abs_path}\n")
            except Exception as exc:
                print(f"    - FATAL WARNING: Unhandled exception on line {line_data['id']}: {exc}", file=sys.stderr)

    print(f"Pylaia input file list saved to: {pylaia_list_path}")

