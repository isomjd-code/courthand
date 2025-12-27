"""Configuration constants for greyscale line preprocessing."""

from __future__ import annotations

# Minimum dimensions for valid line images
MIN_LINE_HEIGHT = 8
"""Minimum height in pixels for a valid line image."""

MIN_LINE_WIDTH = 8
"""Minimum width in pixels for a valid line image."""

FINAL_LINE_HEIGHT = 128
"""Target height in pixels for normalized line images."""

FINAL_LINE_WIDTH_PADDING = 10
"""Horizontal padding in pixels added to each side of normalized lines."""

MAX_LINE_WIDTH = 6000
"""Maximum width in pixels allowed for normalized line images."""

DESLANT_ANGLE_RANGE = (-15, 16)
"""Allowed range in degrees for deslanting correction (min, max)."""

PRE_PROCESSING_BLUR_KERNEL = (3, 3)
"""Gaussian blur kernel size (width, height) applied before processing. (0, 0) disables blur. (3, 3) enables mild blur to reduce noise."""

POLYGON_EXPANSION_PERCENT = 12.0
"""Percentage by which to expand polygons outward from baseline."""

FINAL_BASELINE_Y_OFFSET = 58
"""Target Y position in pixels for baseline in normalized images."""

WHITESPACE_PADDING_VERTICAL = 2
"""Vertical padding in pixels at top and bottom of normalized images."""

# Greyscale normalization settings
CONTRAST_PERCENTILE_LOW = 2
"""Lower percentile for contrast normalization (values below become 0)."""

CONTRAST_PERCENTILE_HIGH = 90
"""Upper percentile for contrast normalization (values above become 255). Lower = whiter background."""

ADAPTIVE_DENOISE = True
"""Whether to apply adaptive denoising to the image."""

DENOISE_STRENGTH = 5
"""Strength of non-local means denoising (higher = more smoothing, lower preserves detail)."""

NORMALIZE_TO_WHITE_BACKGROUND = False
"""Whether to invert images to ensure white background (text as dark on light).
Set to False if source images are known to have correct polarity (dark text on light background).
Only enable this if working with potentially inverted source images (e.g., photographic negatives)."""

USE_CLAHE = True
"""Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement."""

CLAHE_CLIP_LIMIT = 2.5
"""Clip limit for CLAHE (higher = more contrast, but can amplify noise)."""

CLAHE_TILE_SIZE = 8
"""Tile grid size for CLAHE (smaller = more local adaptation)."""

FLATTEN_BACKGROUND = True
"""Apply morphological background flattening to remove uneven illumination."""

BACKGROUND_KERNEL_SIZE = 51
"""Kernel size for background estimation (should be larger than stroke width)."""

UNSHARP_MASK_AMOUNT = 0.8
"""Amount of unsharp mask to apply for edge enhancement (0 = none)."""

UNSHARP_MASK_RADIUS = 1.0
"""Radius of unsharp mask in pixels."""

FINAL_GAMMA = 0.85
"""Gamma correction for final output (< 1 darkens midtones/text, > 1 lightens)."""

BBOX_LEFT_EXTENSION = 150
"""Number of pixels to extend bounding boxes and baselines to the left for line image extraction.
This makes PyLaia operate on a longer line image with additional context on the left side."""
