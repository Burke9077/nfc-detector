"""
Test 32: Factory vs NFC Side Front Classification

This model determines if a front side image shows a factory-cut or NFC-cut card.
It classifies front side images into two categories:
- factory: Images showing factory-cut sides on the front of cards (both die-cut and rough-cut)
- nfc: Images showing NFC-cut sides on the front of cards

This model helps identify non-factory cuts (NFCs) by examining the
front side characteristics of the card.
"""

from pathlib import Path
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_side_front_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 32: Factory vs NFC Side Front Classification
    Compares factory-cut and NFC sides on the front of the card
    """
    print("\n=== Running Test 32: Factory vs NFC (Side Front) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_front_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory side fronts to 'factory' class (combining die-cut and rough-cut)
    factory_sources = [
        data_path / "factory-cut-sides-fronts-die-cut",
        data_path / "factory-cut-sides-fronts-rough-cut"  # Include if it exists
    ]
    # Filter out non-existent paths
    factory_sources = [p for p in factory_sources if p.exists()]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC side fronts to 'nfc' class
    nfc_sources = [data_path / "nfc-sides-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "32_side_front_factory_vs_nfc_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn
