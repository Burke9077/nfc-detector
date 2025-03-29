"""
Test 04: Card Side Focus Classification

This model determines if a card side image is in focus or blurry.
It classifies side images into two categories:
- clear: Side images that are properly focused and suitable for further analysis
- blurry: Side images that are out of focus, too blurry for reliable processing

This quality control check ensures that subsequent side models receive 
clear images for accurate classification, as blurry images may lead
to incorrect results in differentiating factory-cut from NFC cards.
"""

# Model metadata
MODEL_NAME = "focus_sides"
MODEL_NUMBER = "04"
MODEL_DESCRIPTION = "Card side focus check - Detects clear vs blurry side images"

from pathlib import Path
import shutil
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_focus_sides(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 04: Card Side Focus Classification
    Classifies side images into: clear or blurry
    """
    print("\n=== Running Test 04: Card Side Focus Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "focus_sides")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define folder mapping to target classes
    # All standard side card images go to 'clear' class
    clear_folders = [
        "factory-cut-sides-backs-die-cut", 
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-backs-rough-cut",
        "factory-cut-sides-fronts-rough-cut", 
        "nfc-sides-backs",
        "nfc-sides-fronts"
    ]
    
    # Only side blurry images go to 'blurry' class
    blurry_folders = [
        "sides-blurry"
    ]
    
    # Copy images from clear folders (limiting to balance classes)
    print("\nProcessing clear side images:")
    max_per_folder = 200  # Lower limit per folder to balance classes
    clear_count = 0
    for folder in clear_folders:
        source = data_path / folder
        if source.exists():
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            
            # Sample if there are too many
            if folder_count > max_per_folder:
                import random
                sampled_images = random.sample(folder_images, max_per_folder)
                temp_sample_dir = work_path / f"temp_sample_{folder}"
                temp_sample_dir.mkdir(exist_ok=True, parents=True)
                for img in sampled_images:
                    shutil.copy(img, temp_sample_dir / img.name)
                copy_images_to_class([temp_sample_dir], temp_dir, "clear")
                copied_count = len(sampled_images)
            else:
                copy_images_to_class([source], temp_dir, "clear")
                copied_count = folder_count
            
            clear_count += copied_count
            print(f"  - Added {copied_count} images from {folder}")
    
    # Copy all blurry side images
    print("\nProcessing blurry side images:")
    blurry_count = 0
    for folder in blurry_folders:
        source = data_path / folder
        if source.exists():
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            copy_images_to_class([source], temp_dir, "blurry")
            blurry_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for side focus model:")
    print(f"  Clear images: {clear_count}")
    print(f"  Blurry images: {blurry_count}")
    
    # Train and save model
    model_path = models_path / "04_focus_sides_model.pkl"
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
