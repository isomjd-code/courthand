"""Parsing helpers for Kraken JSON output."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple


def parse_kraken_json_for_processing(json_path: str) -> List[Dict[str, Any]]:
    """
    Parse Kraken segmentation JSON output and extract line metadata.

    Reads a JSON file produced by Kraken's segmentation tool and extracts
    information about each detected text line, including polygon boundaries
    and baseline coordinates.

    Args:
        json_path: Path to the Kraken JSON file containing segmentation data.

    Returns:
        A list of dictionaries, each containing:
        - "id": Line identifier from the JSON
        - "polygon": List of (x, y) tuples defining the line boundary
        - "baseline": Space-separated string of "x,y" coordinate pairs
        - "baseline_coords": List of (x, y) tuples for baseline points
        Returns an empty list if the file cannot be read or parsed, or if
        no valid lines are found.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"ERROR: Could not parse Kraken JSON: {exc}", file=sys.stderr)
        return []

    lines_data: List[Dict[str, Any]] = []
    for line in data.get("lines", []):
        line_id = line.get("id")
        polygon = [tuple(point) for point in line.get("boundary", [])]
        baseline = [tuple(point) for point in line.get("baseline", [])]
        baseline_str = " ".join(f"{int(point[0])},{int(point[1])}" for point in baseline)
        if polygon and baseline:
            lines_data.append(
                {
                    "id": line_id,
                    "polygon": polygon,
                    "baseline": baseline_str,
                    "baseline_coords": baseline,
                }
            )
    return lines_data

