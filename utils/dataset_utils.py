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

def balanced_copy_images(source_paths, target_dir, class_name, max_per_folder):
    """
    Copy a balanced subset of images from multiple source folders to a target folder.
    Handles empty folders and ensures proper error handling.
    
    Args:
        source_paths: List of Path objects pointing to source directories
        target_dir: Target directory where images will be copied
        class_name: Name of the class (will create a subdirectory with this name)
        max_per_folder: Maximum number of images to copy from each source folder
        
    Returns:
        Total number of images copied
    """
    import random
    from pathlib import Path
    import shutil
    
    # Create target class directory if it doesn't exist
    target_class_dir = target_dir / class_name
    target_class_dir.mkdir(exist_ok=True, parents=True)
    
    total_copied = 0
    valid_folders = 0
    
    for source_path in source_paths:
        if not source_path.exists():
            print(f"  Warning: Source path {source_path} does not exist. Skipping.")
            continue
            
        # Find all jpg images (case insensitive)
        image_files = list(source_path.glob('*.[jJ][pP][gG]'))
        
        if not image_files:
            print(f"  Note: No images found in {source_path}. Skipping.")
            continue
            
        valid_folders += 1
        
        # Determine how many to copy (not exceeding available images)
        num_to_copy = min(len(image_files), max_per_folder)
        
        # Randomly sample if we need fewer than all available
        if num_to_copy < len(image_files):
            selected_images = random.sample(image_files, num_to_copy)
        else:
            selected_images = image_files
            
        # Copy the selected images
        for img_path in selected_images:
            dest_path = target_class_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            
        total_copied += num_to_copy
        print(f"  Copied {num_to_copy} images from {source_path.name}")
    
    if valid_folders == 0:
        print(f"  Warning: No valid folders found for class '{class_name}'")
        
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

def prepare_balanced_dataset(class_paths_dict, temp_dir, max_images_per_class=800, visualize=False, viz_path=None, model_name=None):
    """
    Prepare a balanced dataset from multiple classes with dynamic balancing.
    
    Args:
        class_paths_dict: Dictionary with class names as keys and lists of Path objects as values
        temp_dir: Target directory where balanced dataset will be created
        max_images_per_class: Maximum cap of images per class (default: 800)
        visualize: Whether to visualize class balance (default: False)
        viz_path: Path where visualization will be saved (required if visualize=True)
        model_name: Name of the model for the visualization title (optional)
        
    Returns:
        dict: Dictionary with class names as keys and number of copied images as values
    """
    # Validate inputs
    if visualize and viz_path is None:
        raise ValueError("viz_path must be provided when visualize=True")
    
    # Count available images for each class
    class_available = {}
    for class_name, paths in class_paths_dict.items():
        class_available[class_name] = count_images_in_folders(paths)
    
    # Print available images count
    print("\nAvailable images:")
    for class_name, count in class_available.items():
        print(f"  {class_name}: {count}")
        
    # Check for empty classes
    empty_classes = [class_name for class_name, count in class_available.items() if count == 0]
    if empty_classes:
        raise ValueError(f"The following classes have zero images: {', '.join(empty_classes)}. Please check your data directories.")
    
    # Determine balanced sampling strategy (minority class or capped)
    min_class_count = min(class_available.values())
    images_per_class = min(min_class_count, max_images_per_class)
    
    # Process each class
    class_counts = {}
    for class_name, paths in class_paths_dict.items():
        # For multi-folder classes, distribute evenly
        valid_folders = sum(1 for p in paths if p.exists() and any(p.glob('*.[jJ][pP][gG]')))
        if valid_folders == 0:
            raise ValueError(f"No valid folders with images found for '{class_name}' class")
            
        images_per_folder = images_per_class // valid_folders
        
        print(f"\nProcessing {class_name} images:")
        print(f"  Using max {images_per_folder} images per folder (controlled by max_images_per_class)")
        
        # Copy the images
        copied = balanced_copy_images(paths, temp_dir, class_name, images_per_folder)
        class_counts[class_name] = copied
    
    # Summary of class distribution
    print("\nClass distribution for dataset:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Visualize the class balance if requested
    if visualize:
        visualize_class_balance(class_counts, viz_path, model_name)
    
    return class_counts

def visualize_class_balance(class_counts, save_path, model_name=None):
    """
    Visualize the class balance as a bar chart and save to file.
    
    Args:
        class_counts: Dictionary with class names as keys and counts as values
        save_path: Path where the visualization will be saved
        model_name: Optional model name for the title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f"Class Distribution{' for ' + model_name if model_name else ''}")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Class balance visualization saved to {save_path}")
