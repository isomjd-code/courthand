"""
Line preprocessing package for handwritten text recognition.

This package provides utilities for extracting and normalizing text lines from
page images. It processes Kraken segmentation output to create standardized
line images suitable for HTR (Handwritten Text Recognition) systems like PyLaia.

Main components:
- processing: Core image processing pipeline (rotation, rectification, deslanting, binarization)
- geometry: Polygon and coordinate transformation utilities
- parser: Kraken JSON parsing and data extraction
- runner: Main CLI entry point for batch processing
- config: Configuration constants for preprocessing parameters
"""

from .runner import main
from .processing import process_line_image_dt_based
from .parser import parse_kraken_json_for_processing

__all__ = ["main", "process_line_image_dt_based", "parse_kraken_json_for_processing"]

