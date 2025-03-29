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
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import balanced_copy_images
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
    
    # Copy images from all normal folders (limiting to balance classes)
    print("\nProcessing normal orientation corner images:")
    max_per_folder = 200  # Lower limit per folder to balance classes
    
    # Convert folder names to full paths
    normal_paths = [data_path / folder for folder in normal_folders]
    normal_count = balanced_copy_images(normal_paths, temp_dir, "normal", max_per_folder)
    
    # Copy all wrong orientation images
    print("\nProcessing wrong-orientation corner images:")
    wrong_orientation_paths = [data_path / folder for folder in wrong_orientation_folders]
    wrong_orient_count = balanced_copy_images(wrong_orientation_paths, temp_dir, "wrong-orientation")
    
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
