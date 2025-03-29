"""
Dataset Utilities for NFC Detector

This module provides functions for dataset preparation and management:
- Copying and organizing images into class directories
- Balanced sampling to prevent class imbalance
- Support for multi-source folder structures

These utilities help prepare training data for FastAI models by organizing
source images into the proper directory structure for ImageDataLoaders.
"""

from pathlib import Path
import shutil
import random

def copy_images_to_class(source_folders, target_dir, class_name):
    """
    Copy images from source folders to a class directory in the target directory
    
    Args:
        source_folders: List of source folder paths
        target_dir: Target directory path
        class_name: Name of the class (subdirectory to create)
    
    Returns:
        int: Number of images copied
    """
    # Create class directory if it doesn't exist
    class_dir = target_dir / class_name
    class_dir.mkdir(exist_ok=True, parents=True)
    
    # Track number of images copied
    copied_count = 0
    
    # Copy images from each source folder
    for folder in source_folders:
        if not folder.exists():
            continue
            
        # Find all image files in the source folder
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        # Copy each image to the class directory
        for img in image_files:
            shutil.copy(img, class_dir / img.name)
            copied_count += 1
    
    return copied_count

def balanced_copy_images(source_folders, target_dir, class_name, max_per_folder=None):
    """
    Copy images from source folders to a class directory, with optional balancing
    
    Args:
        source_folders: List of source folder paths
        target_dir: Target directory path
        class_name: Name of the class (subdirectory to create)
        max_per_folder: Maximum number of images to take from each folder (for balancing)
    
    Returns:
        int: Number of images copied
    """
    # Create class directory if it doesn't exist
    class_dir = target_dir / class_name
    class_dir.mkdir(exist_ok=True, parents=True)
    
    # Track number of images copied
    total_copied = 0
    
    # Process each source folder
    for folder in source_folders:
        if not folder.exists():
            continue
            
        # Find all image files in the source folder
        folder_images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        folder_count = len(folder_images)
        
        # Sample if there are too many and a limit is set
        if max_per_folder is not None and folder_count > max_per_folder:
            # Create temporary sample directory
            temp_sample_dir = Path(target_dir).parent / f"temp_sample_{folder.name}"
            temp_sample_dir.mkdir(exist_ok=True, parents=True)
            
            # Sample random images
            sampled_images = random.sample(folder_images, max_per_folder)
            for img in sampled_images:
                shutil.copy(img, temp_sample_dir / img.name)
                
            # Copy sampled images to class dir
            copied = copy_images_to_class([temp_sample_dir], target_dir, class_name)
            
            # Clean up temp directory
            shutil.rmtree(temp_sample_dir)
        else:
            # Copy all images directly
            copied = copy_images_to_class([folder], target_dir, class_name)
        
        total_copied += copied
        print(f"  - Added {copied} images from {folder.name}")
    
    return total_copied

def count_images_in_folders(folder_paths):
    """
    Count the total number of images across multiple folders
    
    Args:
        folder_paths: List of Path objects pointing to folders containing images
        
    Returns:
        int: Total count of image files found across all folders
    """
    total = 0
    for folder in folder_paths:
        if folder.exists():
            # Count files with common image extensions
            img_count = len(list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + 
                          list(folder.glob('*.png')) + list(folder.glob('*.bmp')))
            total += img_count
    return total
