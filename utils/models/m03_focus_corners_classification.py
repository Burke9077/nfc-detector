"""
Test 03: Card Corner Focus Classification

This model determines if a card corner image is in focus or blurry.
It classifies corner images into two categories:
- clear: Corner images that are properly focused and suitable for further analysis
- blurry: Corner images that are out of focus, too blurry for reliable processing

This quality control check ensures that subsequent corner models receive 
clear images for accurate classification, as blurry images may lead
to incorrect results in differentiating factory-cut from NFC cards.
"""

# Model metadata
MODEL_NAME = "focus_corners"
MODEL_NUMBER = "03"
MODEL_DESCRIPTION = "Card corner focus check - Detects clear vs blurry corner images"

from pathlib import Path
from utils.test_utils import run_classification_test

def test_focus_corners(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 03: Card Corner Focus Classification
    Classifies corner images into: clear or blurry
    """
    # Define class folders
    class_folders = {
        "clear": [
            "factory-cut-corners-backs", 
            "factory-cut-corners-fronts", 
            "nfc-corners-backs", 
            "nfc-corners-fronts"
        ],
        "blurry": [
            "corners-blurry"
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
        test_name="Card Corner Focus Classification",
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
