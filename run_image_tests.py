from pathlib import Path
import os
from image_test_utils import *
from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

def test_combined_corners(base_path):
    """
    Test 3: Factory corners (backs + fronts) vs NFC corners (backs + fronts)
    """
    print("\n=== Running Test 1: Factory vs NFC (Combined Front/Back) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory corners (both front and back) to 'factory' class
    factory_sources = [
        base_path / "factory-cut-corners-backs",
        base_path / "factory-cut-corners-fronts"
    ]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corners (both front and back) to 'nfc' class
    nfc_sources = [
        base_path / "nfc-corners-backs",
        base_path / "nfc-corners-fronts"
    ]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = base_path / "models" / "combined_corners_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path, 
        epochs=25,  # Increased epochs (early stopping will prevent overfitting)
        img_size=(1280, 720),  # Even higher resolution to catch subtle differences
        enhance_edges_prob=0.3,  # Apply edge enhancement to 30% of images
        use_tta=True,  # Use Test Time Augmentation
        progressive_resizing=False  # Skip progressive resizing to preserve subtle edge details
    )
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn

def test_fronts_only(base_path):
    """
    Test 1: Factory fronts vs NFC fronts
    """
    print("\n=== Running Test 2: Factory vs NFC (Fronts Only) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory fronts to 'factory_front' class
    factory_sources = [base_path / "factory-cut-corners-fronts"]
    copy_images_to_class(factory_sources, temp_dir, "factory_front")
    
    # Copy NFC fronts to 'nfc_front' class
    nfc_sources = [base_path / "nfc-corners-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_front")
    
    # Train and save model with enhanced settings
    model_path = base_path / "models" / "fronts_only_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path, 
        epochs=25,
        img_size=(1280, 720),  # Higher resolution
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False  # Skip progressive resizing
    )
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn
    
def test_backs_only(base_path):
    """
    Test 2: Factory backs vs NFC backs
    """
    print("\n=== Running Test 3: Factory vs NFC (Backs Only) ===")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(base_path)
    
    # Copy factory backs to 'factory_back' class
    factory_sources = [base_path / "factory-cut-corners-backs"]
    copy_images_to_class(factory_sources, temp_dir, "factory_back")
    
    # Copy NFC backs to 'nfc_back' class
    nfc_sources = [base_path / "nfc-corners-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_back")
    
    # Train and save model with enhanced settings
    model_path = base_path / "models" / "backs_only_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path, 
        epochs=25,
        img_size=(1280, 720),  # Higher resolution
        enhance_edges_prob=0.3,
        use_tta=True, 
        progressive_resizing=False  # Skip progressive resizing
    )
    
    # Clean up
    clean_temp_dir(temp_dir)
    return learn

def check_gpu_memory():
    """Check and print available GPU memory"""
    if cuda.is_available():
        device = cuda.current_device()
        print(f"GPU: {cuda.get_device_name(device)}")
        
        # Print memory stats
        total_mem = cuda.get_device_properties(device).total_memory / 1e9  # Convert to GB
        reserved = cuda.memory_reserved(device) / 1e9
        allocated = cuda.memory_allocated(device) / 1e9
        free = total_mem - reserved
        
        print(f"Total GPU memory: {total_mem:.2f} GB")
        print(f"Reserved memory: {reserved:.2f} GB")
        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Free memory: {free:.2f} GB")
        
        # Plot memory usage
        labels = ['Total', 'Reserved', 'Allocated', 'Free']
        values = [total_mem, reserved, allocated, free]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
        plt.title('GPU Memory Usage (GB)')
        plt.ylabel('Memory (GB)')
        plt.show()
    else:
        print("No GPU available")

def main():
    # Set base path to your image dataset (using relative path)
    base_path = Path("data")
    
    # Create models directory if it doesn't exist
    models_dir = base_path / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Check GPU status before starting
    check_gpu_memory()
    
    # Run all three tests
    test_fronts_only(base_path)
    test_backs_only(base_path)
    test_combined_corners(base_path)

    print("All tests completed!")

if __name__ == "__main__":
    main()
