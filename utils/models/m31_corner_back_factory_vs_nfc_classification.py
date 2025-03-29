"""
Test 31: Factory vs NFC Corner Back Classification

This model determines if a back corner image shows a factory-cut or NFC-cut card.
It classifies back corner images into two categories:
- factory: Images showing factory-cut corners on the back of cards
- nfc: Images showing NFC-cut corners on the back of cards

This model helps identify non-factory cuts (NFCs) by examining the
back corner characteristics of the card.
"""

from pathlib import Path
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_corner_back_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 31: Factory vs NFC Corner Back Classification
    Compares factory-cut and NFC corners on the back of the card
    """
    print("\n=== Running Test 31: Factory vs NFC (Corner Back) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "corner_back_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory corner backs to 'factory' class
    factory_sources = [data_path / "factory-cut-corners-backs"]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corner backs to 'nfc' class
    nfc_sources = [data_path / "nfc-corners-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "31_corner_back_factory_vs_nfc_model.pkl"
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
