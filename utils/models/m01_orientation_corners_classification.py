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
import os
from utils.test_utils import run_classification_test

def test_orientation_corners(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 01: Card Corner Orientation Classification
    Classifies corner images into: normal or wrong-orientation
    """
    # Define class folders
    class_folders = {
        "normal": [
            "factory-cut-corners-backs", 
            "factory-cut-corners-fronts", 
            "nfc-corners-backs", 
            "nfc-corners-fronts",
        ],
        "wrong-orientation": [
            "corners-wrong-orientation",
        ]
    }
    
    # Define training parameters
    train_params = {
        "epochs": 15,  # Fewer epochs for simpler binary task
        "img_size": (720, 1280),
        "enhance_edges_prob": 0.0,  # No edge enhancement needed
        "use_tta": True,
        "max_rotate": 1.0,  # Minimal rotation as requested
    }
    
    # Run the test using the standardized workflow
    return run_classification_test(
        test_name="Card Corner Orientation Classification",
        model_name=MODEL_NAME,
        model_number=MODEL_NUMBER,
        data_path=data_path,
        work_path=work_path,
        models_path=models_path,
        class_folders_dict=class_folders,
        train_params=train_params,
        resume=resume,
        recalculate_lr=recalculate_lr
    )
