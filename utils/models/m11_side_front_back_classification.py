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

from pathlib import Path
import os
from utils.test_utils import run_classification_test

def test_side_front_back(data_path, work_path, models_path, resume=False, recalculate_lr=False, force_overwrite=False):
    """
    Test 11: Side Front/Back Classification
    Classifies side images as either front or back regardless of cut type
    """
    # Define class folders
    class_folders = {
        "front": [
            "factory-cut-sides-fronts-die-cut",
            "factory-cut-sides-fronts-rough-cut",  # Include if it exists
            "nfc-sides-fronts"
        ],
        "back": [
            "factory-cut-sides-backs-die-cut",
            "factory-cut-sides-backs-rough-cut",  # Include if it exists
            "nfc-sides-backs"
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
        test_name="Side Front/Back Classification",
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
