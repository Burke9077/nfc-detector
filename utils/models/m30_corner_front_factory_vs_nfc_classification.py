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
from utils.test_utils import run_classification_test

def test_corner_front_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False, force_overwrite=False):
    """
    Test 30: Factory vs NFC Corner Front Classification
    Compares factory-cut and NFC corners on the front of the card
    """
    # Define class folders
    class_folders = {
        "factory": ["factory-cut-corners-fronts"],
        "nfc": ["nfc-corners-fronts"]
    }
    
    # Define training parameters
    train_params = {
        "epochs": 64,
        "img_size": (720, 1280),
        "enhance_edges_prob": 0.3,
        "use_tta": True,
        "max_rotate": 1.0,  # Minimal rotation as requested
    }
    
    # Run the test using the standardized workflow
    return run_classification_test(
        test_name="Factory vs NFC (Corner Front)",
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
