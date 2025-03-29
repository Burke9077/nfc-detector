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
from utils.test_utils import run_classification_test

def test_focus_sides(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 04: Card Side Focus Classification
    Classifies side images into: clear or blurry
    """
    # Define class folders
    class_folders = {
        "clear": [
            "factory-cut-sides-backs-die-cut", 
            "factory-cut-sides-fronts-die-cut",
            "factory-cut-sides-backs-rough-cut",
            "factory-cut-sides-fronts-rough-cut", 
            "nfc-sides-backs",
            "nfc-sides-fronts"
        ],
        "blurry": [
            "sides-blurry"
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
        test_name="Card Side Focus Classification",
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
