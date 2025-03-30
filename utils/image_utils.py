"""
Image Utilities for NFC Detector

This module provides image processing utilities for the card classification pipeline, including:
- Edge enhancement and detection to highlight cutting differences
- Image preprocessing workflows for model training
- Blending and overlay techniques for visual analysis
- Batch processing of images with controlled randomization
- Support for multiple image formats (JPG, PNG)

These utilities support the image classification pipeline by improving feature visibility
and preparing images for model training with enhanced edge characteristics.
"""

import cv2
import shutil
import numpy as np
from pathlib import Path

def enhance_edges(img_path, dest_path, kernel_size=3, sigma=1.0):
    """
    Enhance edges in an image to highlight cutting differences
    
    Args:
        img_path: Path to source image
        dest_path: Path to save enhanced image
        kernel_size: Size of Gaussian kernel for edge detection
        sigma: Standard deviation for Gaussian kernel
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Couldn't read image {img_path}")
        return False
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    # Detect edges using Cannyre
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Add edge overlay to original image
    edge_overlay = img.copy()
    edge_overlay[dilated_edges > 0] = [0, 255, 0]  # Highlight edges in green
    
    # Blend original with edge overlay
    result = cv2.addWeighted(img, 0.7, edge_overlay, 0.3, 0)
    
    # Save the result
    cv2.imwrite(str(dest_path), result)
    return True

def preprocess_with_edge_enhancement(source_dir, target_dir, enhance_probability=0.5):
    """
    Preprocess images with edge enhancement for some percentage of images
    
    Args:
        source_dir: Directory containing original images
        target_dir: Directory to save processed images
        enhance_probability: Probability of applying edge enhancement
    """
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Process all image files
    total = 0
    enhanced = 0
    
    for img_file in source_dir.glob("*.jpg"):
        total += 1
        if np.random.random() < enhance_probability:
            if enhance_edges(img_file, target_dir / img_file.name):
                enhanced += 1
        else:
            shutil.copy(img_file, target_dir / img_file.name)
    
    for img_file in source_dir.glob("*.png"):
        total += 1
        if np.random.random() < enhance_probability:
            if enhance_edges(img_file, target_dir / img_file.name):
                enhanced += 1
        else:
            shutil.copy(img_file, target_dir / img_file.name)
    
    print(f"Processed {total} images, enhanced {enhanced} edges")
