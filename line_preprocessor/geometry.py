"""Geometry helpers for polygon and baseline manipulation."""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_polygon_on_image(
    image: Image.Image,
    polygon: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    width: int = 2,
) -> Image.Image:
    """
    Draw a polygon outline on an image.

    Draws the boundary of a polygon on the image using the specified color
    and line width. Useful for visualization and debugging.

    Args:
        image: Input image to draw on.
        polygon: List of (x, y) coordinates defining the polygon vertices.
        color: RGB tuple (r, g, b) for the line color.
        width: Line width in pixels. Defaults to 2.

    Returns:
        A new RGB image with the polygon drawn on it. Returns a copy of the
        original image if the polygon has fewer than 2 points.
    """
    if not polygon or len(polygon) < 2:
        return image.copy()
    img_copy = image.convert("RGB") if image.mode != "RGB" else image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.line(polygon + [polygon[0]], fill=color, width=width)
    return img_copy


def apply_background_mask(image: Image.Image, polygon: List[Tuple[int, int]], background_color: int) -> Image.Image:
    """
    Mask out areas outside a polygon, replacing them with a background color.

    Creates a mask from the polygon and sets all pixels outside the polygon
    to the specified background color. This is useful for isolating text
    regions and removing background noise.

    Args:
        image: Input grayscale image to mask.
        polygon: List of (x, y) coordinates defining the region to keep.
        background_color: Grayscale value (0-255) to use for masked areas.

    Returns:
        A new image with the polygon region preserved and everything else
        set to background_color. Returns the original image if the polygon
        has fewer than 3 points.
    """
    if not polygon or len(polygon) < 3:
        return image
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    img_np = np.array(image)
    mask_np = np.array(mask)
    masked_np = np.where(mask_np == 255, img_np, background_color)
    return Image.fromarray(masked_np.astype(np.uint8))


def get_angle_from_baseline(baseline_points: List[Tuple[int, int]]) -> float:
    """
    Calculate the rotation angle of a baseline.

    Computes the angle between the first and last points of a baseline,
    which represents the overall rotation of the text line.

    Args:
        baseline_points: List of (x, y) coordinates defining the baseline.

    Returns:
        The angle in degrees. Positive values indicate counter-clockwise
        rotation. Returns 0.0 if fewer than 2 points are provided.
    """
    if len(baseline_points) < 2:
        return 0.0
    x1, y1 = baseline_points[0]
    x2, y2 = baseline_points[-1]
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle_rad)


def transform_polygon_rotate(
    polygon: List[Tuple[int, int]], angle: float, original_size: Tuple[int, int], expand: bool
) -> List[Tuple[int, int]]:
    """
    Rotate a polygon around the center of an image.

    Applies a rotation transformation to all points in the polygon. If expand
    is True, the coordinate system is adjusted to account for image expansion
    during rotation.

    Args:
        polygon: List of (x, y) coordinates to rotate.
        angle: Rotation angle in degrees (positive = counter-clockwise).
        original_size: (width, height) tuple of the original image dimensions.
        expand: If True, adjusts coordinates for expanded image bounds after rotation.

    Returns:
        A new list of (x, y) coordinates representing the rotated polygon.
        Returns an empty list if the input polygon is empty.
    """
    if not polygon:
        return []
    ox, oy = original_size[0] / 2, original_size[1] / 2
    if expand:
        rad_angle = np.deg2rad(angle)
        cos_a, sin_a = np.abs(np.cos(rad_angle)), np.abs(np.sin(rad_angle))
        new_w = int(original_size[0] * cos_a + original_size[1] * sin_a)
        new_h = int(original_size[0] * sin_a + original_size[1] * cos_a)
        shift_x, shift_y = (new_w / 2) - ox, (new_h / 2) - oy
    else:
        shift_x = shift_y = 0

    rotated = []
    rad = np.deg2rad(angle)
    cos_angle, sin_angle = np.cos(rad), np.sin(rad)
    for x, y in polygon:
        p_x, p_y = x - ox, y - oy
        r_x = p_x * cos_angle - p_y * sin_angle
        r_y = p_x * sin_angle + p_y * cos_angle
        rotated.append((int(r_x + ox + shift_x), int(r_y + oy + shift_y)))
    return rotated


def transform_polygon_deslant(polygon: List[Tuple[int, int]], shear_factor: float, x_shift: float) -> List[Tuple[int, int]]:
    """
    Apply a deslanting (shear) transformation to polygon coordinates.

    Transforms polygon points using the same shear matrix applied during
    image deslanting, ensuring coordinates remain aligned with the transformed image.

    Args:
        polygon: List of (x, y) coordinates to transform.
        shear_factor: Shear factor (tan of slant angle) used in deslanting.
        x_shift: Horizontal shift applied during deslanting to keep image in bounds.

    Returns:
        A new list of (x, y) coordinates representing the deslanted polygon.
        Returns an empty list if the input polygon is empty.
    """
    if not polygon:
        return []
    matrix = np.float32([[1, shear_factor, -x_shift], [0, 1, 0]])
    points = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(points, matrix)
    return [(int(point[0][0]), int(point[0][1])) for point in transformed]


def transform_polygon_scale(polygon: List[Tuple[int, int]], scale_factor: float) -> List[Tuple[int, int]]:
    """
    Scale a polygon by a uniform factor.

    Multiplies all x and y coordinates by the scale factor, useful for
    adjusting coordinates after image resizing.

    Args:
        polygon: List of (x, y) coordinates to scale.
        scale_factor: Multiplier for all coordinates (e.g., 2.0 doubles size).

    Returns:
        A new list of (x, y) coordinates representing the scaled polygon.
        Returns an empty list if the input polygon is empty.
    """
    if not polygon:
        return []
    return [(int(point[0] * scale_factor), int(point[1] * scale_factor)) for point in polygon]


def expand_polygon(
    polygon: List[Tuple[int, int]], baseline_str: str, percentage: float, min_padding: int
) -> List[Tuple[int, int]]:
    """
    Expand a polygon outward from the baseline by a percentage.

    Moves each polygon point away from the baseline by a percentage of its
    distance from the baseline, with a minimum padding guarantee. This is
    useful for ensuring text regions include sufficient context around characters.

    Args:
        polygon: List of (x, y) coordinates defining the polygon to expand.
        baseline_str: Space-separated string of "x,y" coordinate pairs for the baseline.
        percentage: Percentage of distance from baseline to expand (e.g., 12.0 = 12%).
        min_padding: Minimum expansion in pixels, regardless of percentage.

    Returns:
        A new list of (x, y) coordinates representing the expanded polygon.
        Returns the original polygon if parsing fails or insufficient data is provided.
    """
    if not polygon or len(polygon) < 3 or not baseline_str:
        return polygon
    try:
        baseline_points = [tuple(map(int, pair.split(","))) for pair in baseline_str.split()]
    except (ValueError, IndexError):
        return polygon
    if len(baseline_points) < 2:
        return polygon

    baseline_xs = [point[0] for point in baseline_points]
    baseline_ys = [point[1] for point in baseline_points]
    sorted_indices = np.argsort(baseline_xs)
    sorted_xs = np.array(baseline_xs)[sorted_indices]
    sorted_ys = np.array(baseline_ys)[sorted_indices]

    def interp_baseline(x: int) -> float:
        return np.interp(x, sorted_xs, sorted_ys)

    expanded = []
    for x, y in polygon:
        baseline_y = interp_baseline(x)
        vec_y = y - baseline_y
        pct_expansion = abs(vec_y) * (percentage / 100.0)
        final_expansion = max(pct_expansion, min_padding)
        shift = np.sign(vec_y) * final_expansion
        expanded.append((x, int(round(y + shift))))
    return expanded

