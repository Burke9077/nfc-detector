from pathlib import Path
import os
from image_test_utils import *
from fastai.vision.all import *

def test_combined_corners(base_path):
    """
    Test 1: Factory corners (backs + fronts) vs NFC corners (backs + fronts)
    """
    print("\n=== Running Test 1: Factory vs NFC (Combined Front/Back) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory corners (both front and back) to 'factory' class
    factory_sources = [
        base_path / "factory-cut-corner-backs",
        base_path / "factory-cut-corner-fronts"
    ]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corners (both front and back) to 'nfc' class
    nfc_sources = [
        base_path / "nfc-corners-backs",
        base_path / "nfc-corners-fronts"
    ]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model
    model_path = base_path / "models" / "combined_corners_model.pkl"
    learn = train_and_save_model(temp_dir, model_path)
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn

def test_fronts_only(base_path):
    """
    Test 2: Factory fronts vs NFC fronts
    """
    print("\n=== Running Test 2: Factory vs NFC (Fronts Only) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory fronts to 'factory_front' class
    factory_sources = [base_path / "factory-cut-corner-fronts"]
    copy_images_to_class(factory_sources, temp_dir, "factory_front")
    
    # Copy NFC fronts to 'nfc_front' class
    nfc_sources = [base_path / "nfc-corners-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_front")
    
    # Train and save model
    model_path = base_path / "models" / "fronts_only_model.pkl"
    learn = train_and_save_model(temp_dir, model_path)
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn
    
def test_backs_only(base_path):
    """
    Test 3: Factory backs vs NFC backs
    """
    print("\n=== Running Test 3: Factory vs NFC (Backs Only) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory backs to 'factory_back' class
    factory_sources = [base_path / "factory-cut-corner-backs"]
    copy_images_to_class(factory_sources, temp_dir, "factory_back")
    
    # Copy NFC backs to 'nfc_back' class
    nfc_sources = [base_path / "nfc-corners-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_back")
    
    # Train and save model
    model_path = base_path / "models" / "backs_only_model.pkl"
    learn = train_and_save_model(temp_dir, model_path)
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn

def main():
    # Set base path to your image dataset (using relative path)
    base_path = Path("data")  # Changed from absolute to relative path
    
    # Create models directory if it doesn't exist
    models_dir = base_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Run all three tests
    test_combined_corners(base_path)
    test_fronts_only(base_path)
    test_backs_only(base_path)
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
