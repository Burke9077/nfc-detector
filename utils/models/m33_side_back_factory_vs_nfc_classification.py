"""
Test 33: Side Back Factory vs NFC Classification

This model determines if a back side image shows a factory-cut card or an NFC card.
It classifies back side images into two categories:
- factory-cut: Back sides that are factory manufactured (die-cut or rough-cut)
- nfc: Back sides that show signs of NFC modification

This model helps identify whether cards have been tampered with based on
their back side appearance.
"""

# Model metadata
MODEL_NAME = "side_back_factory_vs_nfc"
MODEL_NUMBER = "33"
MODEL_DESCRIPTION = "Side back factory vs NFC - Detects if a back side is factory-cut or NFC"
MODEL_CATEGORY = "Cut Classification"

from pathlib import Path
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_side_back_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 33: Factory vs NFC Side Back Classification
    Compares factory-cut and NFC sides on the back of the card
    """
    print("\n=== Running Test 33: Factory vs NFC (Side Back) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_back_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory side backs to 'factory' class (combining die-cut and rough-cut)
    factory_sources = [
        data_path / "factory-cut-sides-backs-die-cut",
        data_path / "factory-cut-sides-backs-rough-cut"  # Include if it exists
    ]
    # Filter out non-existent paths
    factory_sources = [p for p in factory_sources if p.exists()]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC side backs to 'nfc' class
    nfc_sources = [data_path / "nfc-sides-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "33_side_back_factory_vs_nfc_model.pkl"
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
