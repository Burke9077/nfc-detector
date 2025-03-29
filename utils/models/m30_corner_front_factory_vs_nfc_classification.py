"""
Test 30: Corner Front Factory vs NFC Classification

This model determines if a front corner image shows a factory-cut card or an NFC card.
It classifies front corner images into two categories:
- factory-cut: Front corners that are factory manufactured
- nfc: Front corners that show signs of NFC modification

This model helps identify whether cards have been tampered with based on
their front corner appearance.
"""

# Model metadata
MODEL_NAME = "corner_front_factory_vs_nfc"
MODEL_NUMBER = "30"
MODEL_DESCRIPTION = "Corner front factory vs NFC - Detects if a front corner is factory-cut or NFC"

from pathlib import Path
from fastai.vision.all import *
from utils.directory_utils import (find_latest_checkpoint, setup_temp_dir)
from utils.dataset_utils import copy_images_to_class
from image_test_utils import train_and_save_model

def test_corner_front_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 30: Factory vs NFC Corner Front Classification
    Compares factory-cut and NFC corners on the front of the card
    """
    print("\n=== Running Test 30: Factory vs NFC (Corner Front) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "corner_front_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory corner fronts to 'factory' class
    factory_sources = [data_path / "factory-cut-corners-fronts"]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corner fronts to 'nfc' class
    nfc_sources = [data_path / "nfc-corners-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "30_corner_front_factory_vs_nfc_model.pkl"
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
