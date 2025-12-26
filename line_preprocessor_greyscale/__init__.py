"""
Greyscale line preprocessing package for handwritten text recognition.

This package provides utilities for extracting and normalizing text lines from
page images, producing normalized greyscale output instead of binarized images.
It processes Kraken segmentation output to create standardized line images 
suitable for HTR (Handwritten Text Recognition) systems.

Main components:
- processing: Core image processing pipeline (rotation, rectification, deslanting, greyscale normalization)
- runner: Main CLI entry point for batch processing
- config: Configuration constants for preprocessing parameters

Key difference from the standard line_preprocessor:
- Outputs normalized greyscale images instead of binary (black/white) images
- Uses percentile-based contrast normalization
- Applies optional denoising and edge enhancement
"""

from .runner import main
from .processing import process_line_image_greyscale

__all__ = ["main", "process_line_image_greyscale"]
