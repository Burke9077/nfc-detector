"""
Test 02: Card Side Orientation Classification

This model determines if a card side image is in the correct orientation or not.
It classifies side images into two categories:
- normal: Side images shown in proper orientation
- wrong-orientation: Side images that are upside down, sideways, or otherwise incorrectly oriented

This is a preliminary quality control check to ensure that subsequent side models
can assume proper card orientation for their analysis.
"""

# Model metadata
MODEL_NAME = "orientation_sides"
MODEL_NUMBER = "02"
MODEL_DESCRIPTION = "Card side orientation check - Detects normal vs wrong orientation for sides"

from pathlib import Path
from utils.test_utils import run_classification_test

def test_orientation_sides(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 02: Card Side Orientation Classification
    Classifies side images into: normal or wrong-orientation
    """
    # Define class folders
    class_folders = {
        "normal": [
            "factory-cut-sides-backs-die-cut", 
            "factory-cut-sides-fronts-die-cut",
            "factory-cut-sides-backs-rough-cut",
            "factory-cut-sides-fronts-rough-cut", 
            "nfc-sides-backs",
            "nfc-sides-fronts"
        ],
        "wrong-orientation": [
            "sides-wrong-orientation",
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
        test_name="Card Side Orientation Classification",
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
