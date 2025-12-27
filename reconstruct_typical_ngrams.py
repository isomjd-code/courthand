#!/usr/bin/env python3
"""
Reconstruct typical n-gram images using PCA.

For each n-gram:
1. Use PCA to find vertical vectors that best reconstruct the n-gram images
2. Normalize each instance by horizontal distance (0-1)
3. Find centroid loading in PCA space
4. Reconstruct typical image from centroid
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import map_coordinates


def load_ngram_images(ngram_dir: str, ngram: str) -> List[np.ndarray]:
    """
    Load all images for a specific n-gram.
    
    Args:
        ngram_dir: Directory containing n-gram images
        ngram: The n-gram string
        
    Returns:
        List of image arrays (normalized to common dimensions)
    """
    images = []
    
    # Find all images for this n-gram
    ngram_safe = ngram.replace(' ', '_').replace('/', '_').replace('\\', '_')
    if len(ngram_safe) > 50:
        # Try to find files that start with the n-gram
        pattern = ngram_safe[:50]
    else:
        pattern = ngram_safe
    
    for filename in os.listdir(ngram_dir):
        if filename.startswith(pattern) and filename.endswith('.png'):
            img_path = os.path.join(ngram_dir, filename)
            try:
                img = Image.open(img_path).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"  Warning: Failed to load {filename}: {e}")
                continue
    
    return images


def normalize_horizontal_distance(images: List[np.ndarray], target_width: int = None) -> List[np.ndarray]:
    """
    Normalize images to a common width using horizontal distance normalization.
    
    Each image is resampled so that horizontal position maps to normalized distance (0-1).
    This ensures that pixel at position x in original maps to position x_norm in normalized,
    where x_norm = x / width, creating a consistent horizontal coordinate system.
    
    Args:
        images: List of image arrays
        target_width: Target width for normalization (default: median width)
        
    Returns:
        List of normalized image arrays with consistent horizontal distance mapping
    """
    if not images:
        return []
    
    # Determine target width (use median if not specified)
    if target_width is None:
        widths = [img.shape[1] for img in images]
        target_width = int(np.median(widths))
    
    normalized = []
    target_height = images[0].shape[0]  # Use first image's height (should be consistent)
    
    for img in images:
        height, width = img.shape
        
        # Resample image so that normalized horizontal distance (0-1) maps consistently
        # For each normalized position x_norm in [0, 1], sample from original at x = x_norm * width
        from scipy.ndimage import map_coordinates
        
        # Create normalized coordinate grid
        # Each column j in output corresponds to normalized distance j / target_width
        y_coords, x_coords = np.mgrid[0:height, 0:target_width]
        
        # Map normalized x coordinates back to original image coordinates
        # x_norm = j / target_width, so x_orig = x_norm * width = (j / target_width) * width
        x_orig = (x_coords / target_width) * width
        
        # Ensure coordinates are within bounds
        x_orig = np.clip(x_orig, 0, width - 1)
        y_orig = np.clip(y_coords, 0, height - 1)
        
        # Sample from original image using bilinear interpolation
        coords = np.array([y_orig, x_orig])
        resampled = map_coordinates(img, coords, order=1, mode='constant', cval=0.0)
        
        # Ensure exact dimensions
        if resampled.shape != (target_height, target_width):
            # Crop or pad to exact size
            h, w = resampled.shape
            if h != target_height or w != target_width:
                result = np.zeros((target_height, target_width), dtype=np.float32)
                h_min = min(h, target_height)
                w_min = min(w, target_width)
                result[:h_min, :w_min] = resampled[:h_min, :w_min]
                resampled = result
        
        normalized.append(resampled.astype(np.float32))
    
    return normalized


def apply_pca_to_ngram(images: List[np.ndarray], n_components: int = None) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Apply PCA to n-gram images.
    
    Each image is flattened and treated as a sample. PCA finds principal components
    that capture the most variance across instances.
    
    Args:
        images: List of normalized image arrays (all same size)
        n_components: Number of PCA components (default: min(n_samples, n_features) or 50)
        
    Returns:
        Tuple of (pca_model, transformed_data, reconstructed_images)
    """
    if not images:
        return None, None, None
    
    # Flatten images: each row is an image, each column is a pixel
    height, width = images[0].shape
    n_samples = len(images)
    n_features = height * width
    
    # Reshape images to vectors
    X = np.array([img.flatten() for img in images])
    
    # Determine number of components
    if n_components is None:
        n_components = min(n_samples - 1, n_features, 50)  # Limit to reasonable number
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    
    # Reconstruct images
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstructed_images = [X_reconstructed[i].reshape(height, width) for i in range(n_samples)]
    
    return pca, X_transformed, reconstructed_images


def find_centroid_loading(transformed_data: np.ndarray) -> np.ndarray:
    """
    Find the centroid (mean) in PCA space.
    
    Args:
        transformed_data: PCA-transformed data (n_samples, n_components)
        
    Returns:
        Centroid vector in PCA space
    """
    return np.mean(transformed_data, axis=0)


def reconstruct_typical_image(pca: PCA, centroid_loading: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Reconstruct typical image from centroid loading in PCA space.
    
    Args:
        pca: Fitted PCA model
        centroid_loading: Centroid vector in PCA space
        height: Image height
        width: Image width
        
    Returns:
        Reconstructed image array
    """
    # Transform centroid back to pixel space
    reconstructed = pca.inverse_transform(centroid_loading.reshape(1, -1))
    
    # Reshape to image
    img = reconstructed[0].reshape(height, width)
    
    # Clip to valid range [0, 1]
    img = np.clip(img, 0, 1)
    
    return img


def process_ngram_class(
    ngram: str,
    ngram_dir: str,
    output_dir: str,
    n_components: int = None,
    target_width: int = None
) -> bool:
    """
    Process a single n-gram class to find typical representation.
    
    Args:
        ngram: The n-gram string
        ngram_dir: Directory containing n-gram images
        output_dir: Output directory for typical images
        n_components: Number of PCA components
        target_width: Target width for normalization (if None, uses median width of this n-gram)
        
    Returns:
        True if successful, False otherwise
    """
    # Load images
    images = load_ngram_images(ngram_dir, ngram)
    
    if len(images) < 2:
        print(f"  Skipping '{ngram}': need at least 2 instances (found {len(images)})")
        return False
    
    print(f"  Processing '{ngram}': {len(images)} instances")
    
    # Calculate median width for this n-gram class if not specified
    if target_width is None:
        widths = [img.shape[1] for img in images]
        target_width = int(np.median(widths))
        print(f"    Using median width: {target_width} pixels")
    
    # Normalize to common dimensions using median width
    normalized_images = normalize_horizontal_distance(images, target_width)
    
    if not normalized_images:
        print(f"  Skipping '{ngram}': normalization failed")
        return False
    
    height, width = normalized_images[0].shape
    
    # Apply PCA
    pca, transformed_data, reconstructed_images = apply_pca_to_ngram(normalized_images, n_components)
    
    if pca is None:
        print(f"  Skipping '{ngram}': PCA failed")
        return False
    
    # Find centroid
    centroid_loading = find_centroid_loading(transformed_data)
    
    # Find the actual instance closest to the centroid in PCA space
    distances = np.linalg.norm(transformed_data - centroid_loading, axis=1)
    closest_idx = np.argmin(distances)
    
    # Use the actual closest instance as the typical image
    typical_image = normalized_images[closest_idx]
    
    # Save typical image
    os.makedirs(output_dir, exist_ok=True)
    ngram_safe = ngram.replace(' ', '_').replace('/', '_').replace('\\', '_')
    if len(ngram_safe) > 100:
        ngram_safe = ngram_safe[:100]
    
    output_path = os.path.join(output_dir, f"{ngram_safe}_typical.png")
    
    # Convert to uint8 and save
    img_uint8 = (typical_image * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_path)
    
    # Save PCA info
    info_path = os.path.join(output_dir, f"{ngram_safe}_pca_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"N-gram: {ngram}\n")
        f.write(f"Number of instances: {len(images)}\n")
        f.write(f"Image dimensions: {height}x{width}\n")
        f.write(f"PCA components: {pca.n_components_}\n")
        f.write(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}\n")
        f.write(f"Centroid loading shape: {centroid_loading.shape}\n")
        f.write(f"Closest instance index: {closest_idx}\n")
        f.write(f"Distance to centroid: {distances[closest_idx]:.6f}\n")
    
    print(f"    Saved typical image: {output_path}")
    print(f"    Using instance {closest_idx} (distance to centroid: {distances[closest_idx]:.6f})")
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct typical n-gram images using PCA'
    )
    parser.add_argument(
        '--ngrams-file',
        type=str,
        default='important_ngrams.txt',
        help='Path to n-grams file'
    )
    parser.add_argument(
        '--slices-dir',
        type=str,
        default='ngram_slices',
        help='Directory containing n-gram slice images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='typical_ngrams',
        help='Output directory for typical images'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=None,
        help='Number of PCA components (default: auto)'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=None,
        help='Target width for normalization (default: median width per n-gram class)'
    )
    parser.add_argument(
        '--min-instances',
        type=int,
        default=2,
        help='Minimum number of instances required (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Load n-grams
    print(f"Loading n-grams from {args.ngrams_file}...")
    ngrams = []
    with open(args.ngrams_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 1:
                ngram = parts[0]
                ngrams.append(ngram)
    
    print(f"Found {len(ngrams)} n-grams\n")
    
    # Process each n-gram length separately
    os.makedirs(args.output_dir, exist_ok=True)
    
    stats = defaultdict(int)
    
    for ngram in ngrams:
        length = len(ngram)
        ngram_length_dir = os.path.join(args.slices_dir, f'{length}gram')
        
        if not os.path.exists(ngram_length_dir):
            continue
        
        success = process_ngram_class(
            ngram,
            ngram_length_dir,
            args.output_dir,
            args.n_components,
            args.target_width
        )
        
        if success:
            stats[length] += 1
    
    # Print summary
    print("\n=== Summary ===")
    for length in sorted(stats.keys()):
        print(f"  {length}-grams: {stats[length]} typical images generated")
    print(f"\nTotal: {sum(stats.values())} typical images saved to {args.output_dir}")


if __name__ == '__main__':
    main()

