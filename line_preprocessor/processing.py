"""Core image processing pipeline for line extraction."""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_sauvola

from .config import (
    DESLANT_ANGLE_RANGE,
    DESPECKLE_MIN_SIZE,
    FINAL_BASELINE_Y_OFFSET,
    FINAL_LINE_HEIGHT,
    FINAL_LINE_WIDTH_PADDING,
    MAX_LINE_WIDTH,
    MIN_LINE_HEIGHT,
    MIN_LINE_WIDTH,
    MORPH_DILATION_ITERATIONS,
    MORPH_DILATION_KERNEL,
    PRE_BINARIZATION_BLUR_KERNEL,
    SAUVOLA_K,
    SAUVOLA_WINDOW,
    WHITESPACE_PADDING_VERTICAL,
)
from .geometry import (
    apply_background_mask,
    expand_polygon,
    get_angle_from_baseline,
    transform_polygon_deslant,
    transform_polygon_rotate,
    transform_polygon_scale,
)


def initial_line_extraction(
    page_image: Image.Image,
    polygon: List[Tuple[int, int]],
    baseline_str: str,
    padding: int = 5,
) -> Optional[Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Extract a line region from a page image based on polygon coordinates.

    Crops the image to the bounding box of the polygon with padding, and adjusts
    polygon and baseline coordinates to the cropped coordinate system.
    Extends the polygon and baseline to the left by BBOX_LEFT_EXTENSION pixels
    to provide additional context for PyLaia processing.

    Args:
        page_image: The full page image to extract from.
        polygon: List of (x, y) coordinates defining the line boundary polygon.
        baseline_str: Space-separated string of "x,y" coordinate pairs for the baseline.
        padding: Additional pixels to add around the bounding box. Defaults to 5.

    Returns:
        A tuple of (cropped_image, adjusted_polygon, adjusted_baseline) if successful,
        or None if extraction fails. The adjusted coordinates are relative to the
        cropped image's coordinate system.
    """
    try:
        from .config import BBOX_LEFT_EXTENSION
        
        working_image = page_image.convert("L") if page_image.mode != "L" else page_image
        
        # Extend polygon and baseline to the left by BBOX_LEFT_EXTENSION pixels
        # This provides additional context for PyLaia processing
        extended_polygon = [(max(0, x - BBOX_LEFT_EXTENSION), y) for x, y in polygon]
        
        # Parse and extend baseline
        baseline_points = [tuple(map(int, pair.split(","))) for pair in baseline_str.split()]
        extended_baseline = [(max(0, x - BBOX_LEFT_EXTENSION), y) for x, y in baseline_points]
        
        # Compute bounding box from extended polygon
        xs, ys = [point[0] for point in extended_polygon], [point[1] for point in extended_polygon]
        bbox = (
            max(0, min(xs) - padding),
            max(0, min(ys) - padding),
            min(working_image.width, max(xs) + padding),
            min(working_image.height, max(ys) + padding),
        )
        
        # Crop image to extended bounding box
        cropped = working_image.crop(bbox)
        
        # Adjust coordinates to the cropped coordinate system
        adjusted_polygon = [(x - bbox[0], y - bbox[1]) for x, y in extended_polygon]
        adjusted_baseline = [(x - bbox[0], y - bbox[1]) for x, y in extended_baseline]
        
        return cropped, adjusted_polygon, adjusted_baseline
    except Exception as exc:
        print(f"ERROR: Failed initial line extraction: {exc}", file=sys.stderr)
        return None


def sauvola_binarize(img: Image.Image, window_size: int, k: float) -> Image.Image:
    """
    Binarize an image using Sauvola's adaptive thresholding algorithm.

    Sauvola's method computes a local threshold for each pixel based on the mean
    and standard deviation of a window around it, making it effective for
    documents with varying illumination.

    Args:
        img: Input grayscale image to binarize.
        window_size: Size of the local window for threshold calculation (must be odd).
        k: Parameter controlling the sensitivity of the threshold (typically 0.15-0.2).

    Returns:
        A binary PIL Image with pixel values of 0 (black) or 255 (white).
        The window size is automatically adjusted if the image is smaller than
        the requested window size.
    """
    img_np = np.array(img.convert("L"))
    if img_np.shape[0] < window_size or img_np.shape[1] < window_size:
        new_window = min(img_np.shape[0], img_np.shape[1])
        if new_window % 2 == 0:
            new_window -= 1
        window_size = max(3, new_window)
    threshold = threshold_sauvola(img_np, window_size=window_size, k=k)
    binary = np.where(img_np > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def clean_and_thicken_binary(img: Image.Image) -> Image.Image:
    """
    Apply morphological operations to clean noise and thicken strokes.
    
    1. Removes small connected components (dust/speckles).
    2. Erodes (thickens) the black text to connect broken strokes.
    
    Args:
        img: Input binary PIL Image (0=Black Text, 255=White BG)
        
    Returns:
        Cleaned and thickened binary PIL Image
    """
    # Convert PIL to OpenCV (0=Black Text, 255=White BG)
    img_np = np.array(img)
    
    # 1. DESPECKLING (Remove Noise)
    # Invert so text is white (255) for analysis
    binary_inv = cv2.bitwise_not(img_np)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    
    # Create mask of valid components (stats[i, 4] is area)
    # Label 0 is the background, so we start from 1
    mask = np.zeros_like(img_np)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= DESPECKLE_MIN_SIZE:
            mask[labels == i] = 255
            
    # Invert back to Black text
    cleaned = cv2.bitwise_not(mask)
    
    # 2. THICKENING (Fix Broken Strokes)
    # We use 'erode' because in this color space, we want to erode the WHITE background
    # effectively making the BLACK text thicker.
    kernel = np.ones(MORPH_DILATION_KERNEL, np.uint8)
    thickened = cv2.erode(cleaned, kernel, iterations=MORPH_DILATION_ITERATIONS)
    
    return Image.fromarray(thickened)


def deslant_image_contour_based(img: Image.Image) -> Tuple[Image.Image, float, float]:
    """
    Remove slant from handwritten text using contour-based analysis.

    Analyzes character contours to detect the dominant slant angle and applies
    a shear transformation to correct it. This is useful for normalizing
    handwritten text that has been written at an angle.

    Args:
        img: Input grayscale image containing text.

    Returns:
        A tuple of (deslanted_image, shear_factor, x_shift):
        - deslanted_image: The corrected image with slant removed.
        - shear_factor: The shear factor applied (tan of slant angle).
        - x_shift: Horizontal shift applied to compensate for negative shear.
        If no valid contours are found, returns the original image with (0.0, 0.0).
    """
    cv_img = np.array(img.convert("L"))
    inverted = cv2.bitwise_not(cv_img)
    try:
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, 0.0, 0.0

    angles = []
    for contour in contours:
        if 5 < cv2.contourArea(contour) < (img.height * img.width) * 0.8 and len(contour) >= 5:
            try:
                _, _, angle = cv2.fitEllipse(contour)
                if 45 < angle < 135:
                    angles.append(angle)
            except cv2.error:
                continue

    if not angles:
        return img, 0.0, 0.0

    angle_hist, bin_edges = np.histogram(angles, bins=90, range=(45, 135))
    dominant_angle = bin_edges[np.argmax(angle_hist)] + (bin_edges[1] - bin_edges[0]) / 2
    slant_angle_deg = dominant_angle - 90.0
    max_abs = max(abs(DESLANT_ANGLE_RANGE[0]), abs(DESLANT_ANGLE_RANGE[1]))
    if abs(slant_angle_deg) > max_abs:
        slant_angle_deg = np.sign(slant_angle_deg) * max_abs
    shear_factor = np.tan(np.deg2rad(slant_angle_deg))
    x_shift = shear_factor * img.height if shear_factor < 0 else 0
    matrix = np.float32([[1, shear_factor, -x_shift], [0, 1, 0]])
    new_width = img.width + int(abs(shear_factor * img.height))
    deslanted = cv2.warpAffine(
        cv_img,
        matrix,
        (new_width, img.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return Image.fromarray(deslanted), shear_factor, x_shift


def rectify_image_and_polygon_from_baseline(
    image: Image.Image,
    baseline_points: List[Tuple[int, int]],
    polygon: List[Tuple[int, int]],
) -> Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Rectify an image by straightening the baseline to a horizontal line.

    Uses the baseline points to create a remapping that shifts each column
    vertically so that the baseline becomes horizontal. This corrects for
    curved or wavy baselines in handwritten text.

    Args:
        image: Input grayscale image to rectify.
        baseline_points: List of (x, y) coordinates defining the baseline curve.
        polygon: List of (x, y) coordinates defining the text region polygon.

    Returns:
        A tuple of (rectified_image, rectified_polygon, rectified_baseline):
        - rectified_image: Image with baseline straightened to horizontal.
        - rectified_polygon: Polygon coordinates adjusted for the rectification.
        - rectified_baseline: Baseline coordinates adjusted (should be horizontal).
        If fewer than 2 baseline points are provided, returns original image unchanged.
    """
    cv_img = np.array(image.convert("L"))
    height, width = cv_img.shape
    if len(baseline_points) < 2:
        return image, polygon, []

    baseline_points.sort(key=lambda point: point[0])
    map_x, map_y = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    target_y = np.mean([point[1] for point in baseline_points])
    baseline_xs = [point[0] for point in baseline_points]
    baseline_ys = [point[1] for point in baseline_points]
    extended_xs, extended_ys = [], []
    if baseline_xs[0] > 0:
        extended_xs.append(0)
        extended_ys.append(baseline_ys[0])
    extended_xs.extend(baseline_xs)
    extended_ys.extend(baseline_ys)
    if extended_xs[-1] < width - 1:
        extended_xs.append(width - 1)
        extended_ys.append(baseline_ys[-1])
    interp_baseline_y = np.interp(np.arange(width), extended_xs, extended_ys)
    remap_shift = interp_baseline_y - target_y

    for y in range(height):
        for x in range(width):
            map_x[y, x] = x
            map_y[y, x] = y + remap_shift[x]

    rectified = cv2.remap(cv_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    all_points = polygon + baseline_points
    x_coords = np.clip(np.array([point[0] for point in all_points]), 0, width - 1)
    interp_y = np.interp(x_coords, extended_xs, extended_ys)
    shifts = target_y - interp_y
    rectified_polygon = [(point[0], int(point[1] + shifts[i])) for i, point in enumerate(polygon)]
    rectified_baseline = [
        (point[0], int(point[1] + shifts[len(polygon) + i])) for i, point in enumerate(baseline_points)
    ]
    return Image.fromarray(rectified), rectified_polygon, rectified_baseline


def process_line_image_dt_based(
    line_image: Image.Image,
    polygon_coords: List[Tuple[int, int]],
    baseline_points: List[Tuple[int, int]],
    final_canvas_height: int = FINAL_LINE_HEIGHT,
    line_id_for_debug: str = "",
) -> Optional[Image.Image]:
    """
    Process a line image through a complete normalization pipeline.

    Applies a series of transformations to normalize handwritten text lines:
    1. Rotation correction based on baseline angle
    2. Baseline rectification (straightening curved baselines)
    3. Deslanting (removing italic/slanted writing)
    4. Scaling to target height
    5. Binarization using Sauvola's method
    6. Final canvas creation with padding

    This is the main processing function that orchestrates all normalization steps
    to produce a standardized line image suitable for HTR recognition.

    Args:
        line_image: Input grayscale image of a text line.
        polygon_coords: List of (x, y) coordinates defining the text region boundary.
        baseline_points: List of (x, y) coordinates defining the text baseline.
        final_canvas_height: Target height for the output image in pixels.
            Defaults to FINAL_LINE_HEIGHT (96).
        line_id_for_debug: Optional identifier for error messages. Defaults to "".

    Returns:
        A normalized binary PIL Image with standardized dimensions, or None if
        processing fails at any stage (e.g., invalid polygon, image too small).
    """
    def warn(step: str, reason: str) -> None:
        print(f"    - WARNING (Line: {line_id_for_debug}): {step}: {reason}", file=sys.stderr)

    grayscale = line_image.convert("L")
    if PRE_BINARIZATION_BLUR_KERNEL and PRE_BINARIZATION_BLUR_KERNEL != (0, 0):
        grayscale = Image.fromarray(cv2.GaussianBlur(np.array(grayscale), PRE_BINARIZATION_BLUR_KERNEL, 0))

    current_image = grayscale.copy()
    current_polygon = list(polygon_coords)
    current_baseline_points = list(baseline_points)

    temp_mask = Image.new("L", grayscale.size, 0)
    ImageDraw.Draw(temp_mask).polygon(current_polygon, fill=255)
    background_pixels = np.array(grayscale)[np.array(temp_mask) == 0]
    median_background_color = int(np.median(background_pixels)) if background_pixels.size > 0 else 255

    angle = get_angle_from_baseline(current_baseline_points)
    if abs(angle) > 0.1:
        original_size = current_image.size
        current_image = current_image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=median_background_color,
        )
        current_polygon = transform_polygon_rotate(current_polygon, -angle, original_size, expand=True)
        current_baseline_points = transform_polygon_rotate(current_baseline_points, -angle, original_size, expand=True)

    current_image, current_polygon, current_baseline_points = rectify_image_and_polygon_from_baseline(
        current_image, current_baseline_points, current_polygon
    )
    deslanted_image, shear_factor, x_shift = deslant_image_contour_based(current_image)
    if abs(shear_factor) > 0.01:
        current_image = deslanted_image
        current_polygon = transform_polygon_deslant(current_polygon, shear_factor, x_shift)
        current_baseline_points = transform_polygon_deslant(current_baseline_points, shear_factor, x_shift)

    current_image = apply_background_mask(current_image, current_polygon, median_background_color)
    if not current_polygon:
        warn("Polygon Check", "Empty polygon.")
        return None

    min_x = max(0, min(point[0] for point in current_polygon))
    max_x = min(current_image.width, max(point[0] for point in current_polygon))
    min_y = max(0, min(point[1] for point in current_polygon))
    max_y = min(current_image.height, max(point[1] for point in current_polygon))
    bbox = (min_x, min_y, max_x, max_y)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        warn("Crop", f"Invalid bbox: {bbox}")
        return None

    try:
        tight_image = current_image.crop(bbox)
        tight_polygon = [(point[0] - bbox[0], point[1] - bbox[1]) for point in current_polygon]
        tight_baseline = [(point[0] - bbox[0], point[1] - bbox[1]) for point in current_baseline_points]
    except (ValueError, TypeError) as exc:
        warn("Crop", f"Error: {exc}")
        return None

    if tight_image.height < MIN_LINE_HEIGHT or tight_image.width < MIN_LINE_WIDTH:
        warn("Size Check", f"Image too small: {tight_image.size}.")
        return None

    polygon_heights = [point[1] for point in tight_polygon]
    if not polygon_heights:
        warn("Scaling", "Empty polygon post-crop.")
        return None
    original_polygon_height = max(polygon_heights) - min(polygon_heights)
    if original_polygon_height <= 0:
        warn("Scaling", "Zero height polygon.")
        return None

    target_content_height = final_canvas_height - (2 * WHITESPACE_PADDING_VERTICAL)
    if target_content_height <= 0:
        warn("Scaling", "Invalid target height.")
        return None
    scale_factor = target_content_height / original_polygon_height
    scaled_width = int(tight_image.width * scale_factor)
    scaled_height = int(tight_image.height * scale_factor)
    if (
        scaled_width <= 0
        or scaled_height <= 0
        or scaled_width > MAX_LINE_WIDTH
        or scaled_width < MIN_LINE_WIDTH
    ):
        warn("Scaling", f"Invalid scaled width: {scaled_width}")
        return None

    # Use BICUBIC instead of LANCZOS for smoother resizing that preserves stroke connectivity
    # Lanczos creates sharp edges which can look like "ringing" noise when binarized.
    # Bicubic is smoother and preserves stroke connectivity better for handwriting.
    scaled_image = tight_image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
    scaled_polygon = transform_polygon_scale(tight_polygon, scale_factor)
    scaled_baseline_points = transform_polygon_scale(tight_baseline, scale_factor)

    vertical_shift = 0.0
    if scaled_baseline_points:
        polygon_min_y = min(point[1] for point in scaled_polygon)
        baseline_relative = scaled_baseline_points[0][1] - polygon_min_y
        vertical_shift = FINAL_BASELINE_Y_OFFSET - (WHITESPACE_PADDING_VERTICAL + baseline_relative)
    else:
        warn("Vertical Alignment", "No scaled baseline.")

    int_shift = int(round(vertical_shift))
    shifted_canvas = Image.new("L", scaled_image.size, median_background_color)
    shifted_canvas.paste(scaled_image, (0, int_shift))
    normalized_image = shifted_canvas
    transformed_polygon = [(point[0], point[1] + int_shift) for point in scaled_polygon]
    normalized_image = apply_background_mask(normalized_image, transformed_polygon, median_background_color)
    binarized = sauvola_binarize(normalized_image, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K)

    # Apply cleanup and thickening to remove noise and fix broken strokes
    binarized = clean_and_thicken_binary(binarized)

    mask = Image.new("L", binarized.size, 0)
    if len(transformed_polygon) >= 3:
        ImageDraw.Draw(mask).polygon(transformed_polygon, fill=255)
        masked_np = np.where(np.array(mask) == 255, np.array(binarized), 255)
        masked_image = Image.fromarray(masked_np)
    else:
        warn("Masking", "Not enough points for mask.")
        masked_image = binarized

    final_width = masked_image.width + 2 * FINAL_LINE_WIDTH_PADDING
    final_canvas = Image.new("L", (final_width, final_canvas_height), 255)
    final_canvas.paste(masked_image, (FINAL_LINE_WIDTH_PADDING, WHITESPACE_PADDING_VERTICAL))
    return final_canvas

