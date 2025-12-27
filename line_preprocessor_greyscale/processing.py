"""Core image processing pipeline for greyscale line extraction."""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import (
    ADAPTIVE_DENOISE,
    BACKGROUND_KERNEL_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    CONTRAST_PERCENTILE_HIGH,
    CONTRAST_PERCENTILE_LOW,
    DENOISE_STRENGTH,
    DESLANT_ANGLE_RANGE,
    FINAL_BASELINE_Y_OFFSET,
    FINAL_GAMMA,
    FINAL_LINE_HEIGHT,
    FINAL_LINE_WIDTH_PADDING,
    FLATTEN_BACKGROUND,
    MAX_LINE_WIDTH,
    MIN_LINE_HEIGHT,
    MIN_LINE_WIDTH,
    NORMALIZE_TO_WHITE_BACKGROUND,
    PRE_PROCESSING_BLUR_KERNEL,
    UNSHARP_MASK_AMOUNT,
    UNSHARP_MASK_RADIUS,
    USE_CLAHE,
    WHITESPACE_PADDING_VERTICAL,
)

# Import geometry helpers from the original module (they're shared)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from line_preprocessor.geometry import (
    apply_background_mask,
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


def normalize_greyscale(
    img: Image.Image,
    percentile_low: int = CONTRAST_PERCENTILE_LOW,
    percentile_high: int = CONTRAST_PERCENTILE_HIGH,
) -> Image.Image:
    """
    Normalize greyscale image using percentile-based contrast stretching.

    Stretches the histogram so that values at the lower percentile become 0
    and values at the upper percentile become 255. This normalizes for
    varying document illumination and ink density.

    Args:
        img: Input grayscale image to normalize.
        percentile_low: Lower percentile (values below mapped to 0).
        percentile_high: Upper percentile (values above mapped to 255).

    Returns:
        A normalized grayscale PIL Image with improved contrast.
    """
    img_np = np.array(img.convert("L")).astype(np.float32)
    
    # Calculate percentiles
    p_low = np.percentile(img_np, percentile_low)
    p_high = np.percentile(img_np, percentile_high)
    
    # Avoid division by zero
    if p_high - p_low < 1:
        p_high = p_low + 1
    
    # Stretch contrast
    normalized = (img_np - p_low) / (p_high - p_low) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return Image.fromarray(normalized)


def denoise_image(img: Image.Image, strength: int = DENOISE_STRENGTH) -> Image.Image:
    """
    Apply non-local means denoising to reduce noise while preserving edges.

    Args:
        img: Input grayscale image to denoise.
        strength: Strength of the denoising filter (higher = smoother).

    Returns:
        A denoised grayscale PIL Image.
    """
    img_np = np.array(img.convert("L"))
    
    # fastNlMeansDenoising parameters: src, dst, h (filter strength), 
    # templateWindowSize, searchWindowSize
    denoised = cv2.fastNlMeansDenoising(img_np, None, strength, 7, 21)
    
    return Image.fromarray(denoised)


def apply_unsharp_mask(
    img: Image.Image,
    amount: float = UNSHARP_MASK_AMOUNT,
    radius: float = UNSHARP_MASK_RADIUS,
) -> Image.Image:
    """
    Apply unsharp masking to enhance edges and text clarity.

    Args:
        img: Input grayscale image.
        amount: Strength of the sharpening effect (0-2 typical).
        radius: Radius of the blur used for the mask.

    Returns:
        A sharpened grayscale PIL Image.
    """
    if amount <= 0:
        return img
    
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=3))


def detect_needs_inversion(img: Image.Image, mask: Optional[Image.Image] = None) -> bool:
    """
    Conservatively detect if an image needs inversion to have dark text on white background.
    
    This function is designed to have very few false positives - it should only return True
    when the image is DEFINITELY inverted (light text on dark background), not just when
    there's dense text or dark parchment.
    
    The key insight is that BACKGROUND pixels (outside text regions or at image edges)
    should always be bright in a normal document. Dense text coverage doesn't change
    the background color - it just means there's less background visible.

    Args:
        img: Input grayscale image.
        mask: Optional mask where 255 = content area (polygon interior).

    Returns:
        True if the image should be inverted, False otherwise.
    """
    img_np = np.array(img.convert("L"))
    h, w = img_np.shape
    
    # CRITICAL: Focus on BACKGROUND pixels, not content pixels
    # Background pixels are the true indicator of inversion, not text density
    
    if mask is not None:
        mask_np = np.array(mask)
        # Get pixels OUTSIDE the polygon - these are pure background
        background_pixels = img_np[mask_np == 0]
    else:
        background_pixels = np.array([])
    
    # Also sample corners - these are almost always background
    corner_size = max(3, min(h // 10, w // 10, 15))
    corner_pixels = []
    corner_pixels.extend(img_np[:corner_size, :corner_size].flatten())
    corner_pixels.extend(img_np[:corner_size, -corner_size:].flatten())
    corner_pixels.extend(img_np[-corner_size:, :corner_size].flatten())
    corner_pixels.extend(img_np[-corner_size:, -corner_size:].flatten())
    corner_pixels = np.array(corner_pixels)
    
    # Combine background sources
    if len(background_pixels) > 100:
        # Prefer pixels outside polygon if available
        bg_sample = background_pixels
    elif len(corner_pixels) > 0:
        bg_sample = corner_pixels
    else:
        # Fallback: use edges of image
        edge_pixels = np.concatenate([
            img_np[0, :],      # Top edge
            img_np[-1, :],     # Bottom edge  
            img_np[:, 0],      # Left edge
            img_np[:, -1]      # Right edge
        ])
        bg_sample = edge_pixels
    
    if len(bg_sample) == 0:
        return False
    
    # For a truly inverted image, the BACKGROUND must be dark
    # This is the primary and most reliable check
    bg_median = np.median(bg_sample)
    bg_p75 = np.percentile(bg_sample, 75)  # 75th percentile of background
    bg_p90 = np.percentile(bg_sample, 90)  # 90th percentile of background
    
    # Very conservative threshold: background must be definitively dark
    # Normal parchment/paper is typically 180-255
    # Inverted background would be 0-80
    # We require STRONG evidence of inversion
    
    # Primary check: Is the background dark?
    background_is_dark = bg_median < 100 and bg_p75 < 120
    
    if not background_is_dark:
        # Background is not dark - definitely not inverted
        return False
    
    # Secondary check: If background is dark, verify that bright pixels exist
    # (these would be the "text" in an inverted image)
    # In a truly inverted image, there should be some bright pixels (the text)
    if mask is not None:
        mask_np = np.array(mask)
        content_pixels = img_np[mask_np == 255]
    else:
        content_pixels = img_np.flatten()
    
    if len(content_pixels) == 0:
        return False
    
    # Check if there are bright pixels that would be the "text" in inverted image
    p95 = np.percentile(content_pixels, 95)
    bright_pixel_ratio = np.sum(content_pixels > 180) / len(content_pixels)
    
    # For truly inverted image: dark background + some bright text pixels
    # Require both conditions to be very clear
    has_bright_text = p95 > 160 or bright_pixel_ratio > 0.05
    
    # Final decision: only invert if background is clearly dark AND there's bright content
    return background_is_dark and has_bright_text


def invert_image(img: Image.Image) -> Image.Image:
    """Invert a grayscale image."""
    img_np = np.array(img.convert("L"))
    return Image.fromarray(255 - img_np)


def flatten_background(
    img: Image.Image, 
    kernel_size: int = BACKGROUND_KERNEL_SIZE
) -> Image.Image:
    """
    Flatten uneven background illumination using morphological operations.
    
    Estimates the background using a large morphological closing operation,
    then divides the image by this background estimate to normalize illumination.
    This removes parchment texture and uneven lighting while preserving text.

    Args:
        img: Input grayscale image.
        kernel_size: Size of the morphological kernel (should be larger than stroke width).

    Returns:
        A grayscale PIL Image with flattened background.
    """
    img_np = np.array(img.convert("L")).astype(np.float32)
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Estimate background using morphological closing (fills in text)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to smooth the background estimate
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    
    # Avoid division by zero
    background = np.maximum(background, 1)
    
    # Normalize: divide image by background and scale
    # This makes the background uniform while preserving relative text intensity
    normalized = (img_np / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return Image.fromarray(normalized)


def apply_clahe(
    img: Image.Image,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_size: int = CLAHE_TILE_SIZE
) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE provides local contrast enhancement, which is better than global
    histogram equalization for documents with varying illumination. The clip
    limit prevents over-amplification of noise.

    Args:
        img: Input grayscale image.
        clip_limit: Threshold for contrast limiting (higher = more contrast).
        tile_size: Size of grid for histogram equalization.

    Returns:
        A grayscale PIL Image with enhanced local contrast.
    """
    img_np = np.array(img.convert("L"))
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img_np)
    
    return Image.fromarray(enhanced)


def apply_gamma_correction(img: Image.Image, gamma: float = FINAL_GAMMA) -> Image.Image:
    """
    Apply gamma correction to adjust midtone brightness.
    
    Gamma < 1.0 darkens midtones (makes text bolder)
    Gamma > 1.0 lightens midtones
    Gamma = 1.0 no change

    Args:
        img: Input grayscale image.
        gamma: Gamma value for correction.

    Returns:
        A gamma-corrected grayscale PIL Image.
    """
    if abs(gamma - 1.0) < 0.01:
        return img
    
    img_np = np.array(img.convert("L")).astype(np.float32) / 255.0
    corrected = np.power(img_np, gamma) * 255.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return Image.fromarray(corrected)


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
    
    # For greyscale, we need to create a binary version for contour detection
    # Use adaptive thresholding for better results
    binary = cv2.adaptiveThreshold(
        cv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    try:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    
    # Calculate the median background color for fill
    median_bg = int(np.median(cv_img))
    
    x_shift = shear_factor * img.height if shear_factor < 0 else 0
    matrix = np.float32([[1, shear_factor, -x_shift], [0, 1, 0]])
    new_width = img.width + int(abs(shear_factor * img.height))
    deslanted = cv2.warpAffine(
        cv_img,
        matrix,
        (new_width, img.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=median_bg,
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

    # Calculate median background color for border fill
    median_bg = int(np.median(cv_img))

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

    rectified = cv2.remap(cv_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=median_bg)
    all_points = polygon + baseline_points
    x_coords = np.clip(np.array([point[0] for point in all_points]), 0, width - 1)
    interp_y = np.interp(x_coords, extended_xs, extended_ys)
    shifts = target_y - interp_y
    rectified_polygon = [(point[0], int(point[1] + shifts[i])) for i, point in enumerate(polygon)]
    rectified_baseline = [
        (point[0], int(point[1] + shifts[len(polygon) + i])) for i, point in enumerate(baseline_points)
    ]
    return Image.fromarray(rectified), rectified_polygon, rectified_baseline


def process_line_image_greyscale(
    line_image: Image.Image,
    polygon_coords: List[Tuple[int, int]],
    baseline_points: List[Tuple[int, int]],
    final_canvas_height: int = FINAL_LINE_HEIGHT,
    line_id_for_debug: str = "",
) -> Optional[Image.Image]:
    """
    Process a line image through a complete greyscale normalization pipeline.

    Applies a series of transformations to normalize handwritten text lines:
    1. Rotation correction based on baseline angle
    2. Baseline rectification (straightening curved baselines)
    3. Deslanting (removing italic/slanted writing)
    4. Scaling to target height
    5. Greyscale normalization (contrast stretching, denoising)
    6. Final canvas creation with padding

    This is the main processing function that orchestrates all normalization steps
    to produce a standardized greyscale line image suitable for HTR recognition.

    Args:
        line_image: Input grayscale image of a text line.
        polygon_coords: List of (x, y) coordinates defining the text region boundary.
        baseline_points: List of (x, y) coordinates defining the text baseline.
        final_canvas_height: Target height for the output image in pixels.
            Defaults to FINAL_LINE_HEIGHT (128).
        line_id_for_debug: Optional identifier for error messages. Defaults to "".

    Returns:
        A normalized greyscale PIL Image with standardized dimensions, or None if
        processing fails at any stage (e.g., invalid polygon, image too small).
    """
    def warn(step: str, reason: str) -> None:
        print(f"    - WARNING (Line: {line_id_for_debug}): {step}: {reason}", file=sys.stderr)

    grayscale = line_image.convert("L")
    if PRE_PROCESSING_BLUR_KERNEL and PRE_PROCESSING_BLUR_KERNEL != (0, 0):
        grayscale = Image.fromarray(cv2.GaussianBlur(np.array(grayscale), PRE_PROCESSING_BLUR_KERNEL, 0))

    current_image = grayscale.copy()
    current_polygon = list(polygon_coords)
    current_baseline_points = list(baseline_points)

    # Optional early inversion detection - only if NORMALIZE_TO_WHITE_BACKGROUND is enabled
    # and the image appears to be truly inverted (light text on dark background)
    if NORMALIZE_TO_WHITE_BACKGROUND:
        temp_mask = Image.new("L", grayscale.size, 0)
        ImageDraw.Draw(temp_mask).polygon(current_polygon, fill=255)
        if detect_needs_inversion(current_image, temp_mask):
            current_image = invert_image(current_image)

    temp_mask = Image.new("L", current_image.size, 0)
    ImageDraw.Draw(temp_mask).polygon(current_polygon, fill=255)
    background_pixels = np.array(current_image)[np.array(temp_mask) == 0]
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

    # Use BICUBIC for smooth resizing that preserves stroke quality
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

    # Create content mask for analysis and final masking
    content_mask = Image.new("L", normalized_image.size, 0)
    if len(transformed_polygon) >= 3:
        ImageDraw.Draw(content_mask).polygon(transformed_polygon, fill=255)
    
    # Note: Inversion detection (if enabled) is done early in the pipeline before
    # geometric transformations. A single check is sufficient and avoids false positives
    # that could occur from analyzing transformed/normalized pixel distributions.
    
    # Step 1: Flatten background to remove uneven illumination and parchment texture
    if FLATTEN_BACKGROUND:
        normalized_image = flatten_background(normalized_image)
    
    # Step 2: Apply CLAHE for local contrast enhancement
    if USE_CLAHE:
        normalized_image = apply_clahe(normalized_image)
    
    # Step 3: Apply percentile-based contrast normalization
    normalized_image = normalize_greyscale(normalized_image)
    
    # Step 4: Denoise to reduce remaining noise
    if ADAPTIVE_DENOISE:
        normalized_image = denoise_image(normalized_image)
    
    # Step 5: Sharpen edges
    if UNSHARP_MASK_AMOUNT > 0:
        normalized_image = apply_unsharp_mask(normalized_image)
    
    # Step 6: Apply gamma correction to adjust text darkness
    if FINAL_GAMMA != 1.0:
        normalized_image = apply_gamma_correction(normalized_image)

    # Apply polygon mask to set background to white (255)
    if len(transformed_polygon) >= 3:
        masked_np = np.where(np.array(content_mask) == 255, np.array(normalized_image), 255)
        masked_image = Image.fromarray(masked_np.astype(np.uint8))
    else:
        warn("Masking", "Not enough points for mask.")
        masked_image = normalized_image

    final_width = masked_image.width + 2 * FINAL_LINE_WIDTH_PADDING
    final_canvas = Image.new("L", (final_width, final_canvas_height), 255)
    final_canvas.paste(masked_image, (FINAL_LINE_WIDTH_PADDING, WHITESPACE_PADDING_VERTICAL))
    return final_canvas
