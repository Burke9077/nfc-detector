"""
Test 10: Corner Front/Back Classification

This model determines if a corner image shows the front or back of a card.
It classifies corner images into two categories:
- front: Images showing the front corner of cards
- back: Images showing the back corner of cards

This model helps in identifying the orientation and side of card corners,
which is important for subsequent factory vs NFC detection models.
"""

# Model metadata
MODEL_NAME = "corner_front_back"
MODEL_NUMBER = "10"
MODEL_DESCRIPTION = "Corner front/back classifier - Distinguishes front vs back for corners"

from pathlib import Path
import os
from utils.test_utils import run_classification_test

def test_corner_front_back(data_path, work_path, models_path, resume=False, recalculate_lr=False, force_overwrite=False):
    """
    Test 10: Corner Front/Back Classification
    Classifies corner images as either front or back
    """
    # Define class folders
    class_folders = {
        "front": [
            "factory-cut-corners-fronts",
            "nfc-corners-fronts" 
        ],
        "back": [
            "factory-cut-corners-backs",
            "nfc-corners-backs"
        ]
    }
    
    # Define training parameters
    train_params = {
        "epochs": 20,
        "img_size": (720, 1280),
        "enhance_edges_prob": 0.0,  # No edge enhancement needed for front/back detection
        "use_tta": True,
        "max_rotate": 1.0,  # Minimal rotation as requested
    }
    
    # Run the test using the standardized workflow
    return run_classification_test(
        test_name="Corner Front/Back Classification",
        model_name=MODEL_NAME,
        model_number=MODEL_NUMBER,
        data_path=data_path,
        work_path=work_path,
        models_path=models_path,
        class_folders_dict=class_folders,
        train_params=train_params,
        resume=resume,
        recalculate_lr=recalculate_lr, force_overwrite=force_overwrite
    )
