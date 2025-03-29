"""
Test 11: Side Front/Back Classification

This model determines if a side image shows the front or back of a card.
It classifies side images into two categories:
- front: Images showing the front side of cards (regardless of cut type)
- back: Images showing the back side of cards (regardless of cut type)

This model helps in identifying the orientation and side of card edges,
which is important for subsequent factory vs NFC detection models.
"""

# Model metadata
MODEL_NAME = "side_front_back"
MODEL_NUMBER = "11"
MODEL_DESCRIPTION = "Side front/back classifier - Distinguishes front vs back for sides"
MODEL_CATEGORY = "Front/Back Detection"

from pathlib import Path
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_side_front_back(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 11: Side Front/Back Classification
    Classifies side images as either front or back regardless of cut type
    """
    print("\n=== Running Test 11: Side Front/Back Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_front_back")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define source folders for fronts and backs
    front_folders = [
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-fronts-rough-cut",  # Include if it exists
        "nfc-sides-fronts"
    ]
    
    back_folders = [
        "factory-cut-sides-backs-die-cut",
        "factory-cut-sides-backs-rough-cut",  # Include if it exists
        "nfc-sides-backs"
    ]
    
    # Copy front side images to 'front' class
    print("\nProcessing front side images:")
    front_count = 0
    for folder in front_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "front")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            front_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Copy back side images to 'back' class
    print("\nProcessing back side images:")
    back_count = 0
    for folder in back_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "back")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            back_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for side front/back model:")
    print(f"  Front images: {front_count}")
    print(f"  Back images: {back_count}")
    
    # Train and save model
    model_path = models_path / "11_side_front_back_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=20,
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed for front/back detection
        use_tta=True,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn
