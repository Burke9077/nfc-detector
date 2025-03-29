"""
Test 01: Card Corner Orientation Classification

This model determines if a card corner image is in the correct orientation or not.
It classifies corner images into two categories:
- normal: Corner images shown in proper orientation
- wrong-orientation: Corner images that are upside down, sideways, or otherwise incorrectly oriented

This is a preliminary quality control check to ensure that subsequent corner models
can assume proper card orientation for their analysis.
"""

# Model metadata
MODEL_NAME = "orientation_corners"
MODEL_NUMBER = "01"
MODEL_DESCRIPTION = "Card corner orientation check - Detects normal vs wrong orientation for corners"
MODEL_CATEGORY = "QC & Prep"

from pathlib import Path
import shutil
import os
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import balanced_copy_images, count_images_in_folders
from image_test_utils import train_and_save_model

def test_orientation_corners(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 01: Card Corner Orientation Classification
    Classifies corner images into: normal or wrong-orientation
    """
    print("\n=== Running Test 01: Card Corner Orientation Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "orientation_corners")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define folder mapping to target classes
    # Only corner images go to 'normal' class
    normal_folders = [
        "factory-cut-corners-backs", 
        "factory-cut-corners-fronts", 
        "nfc-corners-backs", 
        "nfc-corners-fronts",
    ]
    
    # Only corner wrong orientation images go to 'wrong-orientation' class
    wrong_orientation_folders = [
        "corners-wrong-orientation",
    ]
    
    # Convert folder names to full paths
    normal_paths = [data_path / folder for folder in normal_folders]
    wrong_orientation_paths = [data_path / folder for folder in wrong_orientation_folders]
    
    # Count total available images in each class
    normal_available = count_images_in_folders(normal_paths)
    wrong_orient_available = count_images_in_folders(wrong_orientation_paths)
    
    print("\nAvailable images:")
    print(f"  Normal orientation images: {normal_available}")
    print(f"  Wrong orientation images: {wrong_orient_available}")
    
    # Determine balanced sampling strategy
    # Option 1: Balance to the minority class
    images_per_class = min(normal_available, wrong_orient_available)
    
    # Option 2: Set a maximum cap from environment variable or default
    max_images_per_class = int(os.environ.get('MAX_IMAGES_PER_CLASS', 800))
    images_per_class = min(images_per_class, max_images_per_class)
    
    # Calculate per-folder limits for normal class (distributing evenly)
    normal_folders_exist = sum(1 for p in normal_paths if p.exists())
    max_per_normal_folder = images_per_class // normal_folders_exist if normal_folders_exist > 0 else 0
    
    # Copy images based on dynamic balancing
    print("\nProcessing normal orientation corner images:")
    print(f"  Using max {max_per_normal_folder} images per folder (controlled by MAX_IMAGES_PER_CLASS)")
    normal_count = balanced_copy_images(normal_paths, temp_dir, "normal", max_per_normal_folder)
    
    # For wrong orientation, set the target to match the number of normal images
    print("\nProcessing wrong-orientation corner images:")
    max_wrong_orient = normal_count  # Match the actual number of normal images copied
    wrong_orient_count = balanced_copy_images(wrong_orientation_paths, temp_dir, "wrong-orientation", max_wrong_orient)
    
    # Summary of class distribution
    print("\nClass distribution for corner orientation model:")
    print(f"  Normal orientation images: {normal_count}")
    print(f"  Wrong orientation images: {wrong_orient_count}")
    
    # Train and save model with updated naming convention (01_)
    model_path = models_path / "01_orientation_corners_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=15,  # Fewer epochs for simpler binary task
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed
        use_tta=True,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn
