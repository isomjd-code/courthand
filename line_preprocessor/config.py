"""Configuration constants for line preprocessing."""

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

SAUVOLA_WINDOW = 31
"""Window size for Sauvola binarization (must be odd)."""

SAUVOLA_K = 0.15
"""Sensitivity parameter for Sauvola binarization (typically 0.15-0.2)."""

DESLANT_ANGLE_RANGE = (-15, 16)
"""Allowed range in degrees for deslanting correction (min, max)."""

PRE_BINARIZATION_BLUR_KERNEL = (3, 3)
"""Gaussian blur kernel size (width, height) applied before binarization. (0, 0) disables blur. (3, 3) enables mild blur to reduce noise."""

POLYGON_EXPANSION_PERCENT = 12.0
"""Percentage by which to expand polygons outward from baseline."""

FINAL_BASELINE_Y_OFFSET = 58
"""Target Y position in pixels for baseline in normalized images."""

WHITESPACE_PADDING_VERTICAL = 2
"""Vertical padding in pixels at top and bottom of normalized images."""

DESPECKLE_MIN_SIZE = 15
"""Minimum area (in pixels) for a connected component to be kept. Removes dots/dust."""

MORPH_DILATION_KERNEL = (2, 2)
"""Kernel size for thickening text. (2, 2) is subtle, (3, 3) is strong."""

MORPH_DILATION_ITERATIONS = 1
"""How many times to apply the thickening effect."""

