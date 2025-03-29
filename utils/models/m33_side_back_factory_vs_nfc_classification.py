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

from pathlib import Path
from utils.test_utils import run_classification_test

def test_side_back_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 33: Factory vs NFC Side Back Classification
    Compares factory-cut and NFC sides on the back of the card
    """
    # Define class folders
    class_folders = {
        "factory": [
            "factory-cut-sides-backs-die-cut", 
            "factory-cut-sides-backs-rough-cut"
        ],
        "nfc": ["nfc-sides-backs"]
    }
    
    # Define training parameters
    train_params = {
        "epochs": 25,
        "img_size": (720, 1280),
        "enhance_edges_prob": 0.3,
        "use_tta": True,
        "max_rotate": 1.0,  # Minimal rotation as requested
    }
    
    # Run the test using the standardized workflow
    return run_classification_test(
        test_name="Factory vs NFC (Side Back)",
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
