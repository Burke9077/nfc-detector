"""
Test 01: Card Orientation Classification

This model determines if a card image is in the correct orientation or not.
It classifies images into two categories:
- normal: Cards shown in proper orientation
- wrong-orientation: Cards that are upside down, sideways, or otherwise incorrectly oriented

This is a preliminary quality control check to ensure that subsequent models
can assume proper card orientation for their analysis.
"""

# Model metadata
MODEL_NAME = "orientation"
MODEL_NUMBER = "01"
MODEL_DESCRIPTION = "Card orientation check - Detects normal vs wrong orientation"
MODEL_CATEGORY = "QC & Prep"

from pathlib import Path
import shutil
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import balanced_copy_images
from image_test_utils import train_and_save_model

def test_orientation(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 01: Card Orientation Classification
    Classifies images into: normal or wrong-orientation
    """
    print("\n=== Running Test 01: Card Orientation Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "orientation")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define folder mapping to target classes
    # All standard card images (corners and sides) go to 'normal' class
    normal_folders = [
        "factory-cut-corners-backs", 
        "factory-cut-corners-fronts", 
        "nfc-corners-backs", 
        "nfc-corners-fronts",
        "factory-cut-sides-backs-die-cut", 
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-backs-rough-cut",
        "factory-cut-sides-fronts-rough-cut", 
        "nfc-sides-backs",
        "nfc-sides-fronts"
    ]
    
    # All wrong orientation images (corners and sides) go to 'wrong-orientation' class
    wrong_orientation_folders = [
        "corners-wrong-orientation",
        "sides-wrong-orientation"
    ]
    
    # Copy images from all normal folders (limiting to balance classes)
    print("\nProcessing normal orientation images:")
    max_per_folder = 200  # Lower limit per folder to balance classes
    
    # Convert folder names to full paths
    normal_paths = [data_path / folder for folder in normal_folders]
    normal_count = balanced_copy_images(normal_paths, temp_dir, "normal", max_per_folder)
    
    # Copy all wrong orientation images
    print("\nProcessing wrong-orientation images:")
    wrong_orientation_paths = [data_path / folder for folder in wrong_orientation_folders]
    wrong_orient_count = balanced_copy_images(wrong_orientation_paths, temp_dir, "wrong-orientation")
    
    # Summary of class distribution
    print("\nClass distribution for orientation model:")
    print(f"  Normal orientation images: {normal_count}")
    print(f"  Wrong orientation images: {wrong_orient_count}")
    
    # Train and save model with updated naming convention (01_)
    model_path = models_path / "01_orientation_model.pkl"
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