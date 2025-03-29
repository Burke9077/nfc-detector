import argparse
from pathlib import Path
import os
import sys
import shutil
from image_test_utils import *
from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

def verify_directories(data_path, work_path, models_path):
    """Verify all required directories exist or can be created"""
    # Create key directories
    directories = [
        work_path,                   # Working directory for temporary files
        work_path / "temp_test_dir", # Temporary processing directory
        work_path / "processed_images", # Processed images directory
        models_path,                 # Models output directory
    ]
    
    # Make sure source data directories exist in data_path
    source_directories = [
        data_path / "factory-cut-corners-backs",
        data_path / "factory-cut-corners-fronts",
        data_path / "nfc-corners-backs",
        data_path / "nfc-corners-fronts"
    ]
    
    # Verify each directory we need to create
    for directory in directories:
        try:
            # Create directory if it doesn't exist
            directory.mkdir(exist_ok=True, parents=True)
            print(f"✓ {directory} directory is ready")
        except Exception as e:
            print(f"✗ ERROR: Could not create {directory}: {e}")
            return False
    
    # Verify source directories exist (these should already exist, not be created)
    for src_dir in source_directories:
        if not src_dir.exists():
            print(f"✗ WARNING: Source directory {src_dir} does not exist!")
            print("  Please make sure your data is organized correctly.")
            
            # Ask if the user wants to continue
            if input("Continue anyway? (y/n): ").lower() != 'y':
                return False
    
    return True

def clean_work_dir(work_path):
    """Clean up the entire working directory"""
    if work_path.exists():
        try:
            shutil.rmtree(work_path)
            print(f"Cleaned up working directory: {work_path}")
        except Exception as e:
            print(f"Warning: Could not fully clean up {work_path}: {e}")
            # Try to clean up individual subdirectories
            for item in work_path.iterdir():
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                        print(f"  Cleaned up: {item}")
                    except Exception as e2:
                        print(f"  Could not clean up {item}: {e2}")

def find_latest_checkpoint(work_path, test_name):
    """Find the latest checkpoint for a specific test"""
    checkpoint_dir = work_path / "model_checkpoints"
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = list(checkpoint_dir.glob(f'best_model_stage*'))
    if not checkpoints:
        return None
        
    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest = checkpoints[0]
    print(f"Found checkpoint: {latest}")
    return latest

def test_orientation(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 01: Card Orientation Classification
    Classifies images into: normal or wrong-orientation
    """
    print("\n=== Running Test 01: Card Orientation Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "orientation")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define folder mapping to target classes
    # All standard card images (corners and sides) go to 'normal' class
    normal_folders = [
        "factory-cut-corners-backs", 
        "factory-cut-corners-fronts", 
        "nfc-corners-backs", 
        "nfc-corners-fronts",
        "factory-cut-sides-backs-die-cut", 
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-backs-rough-cut",
        "factory-cut-sides-fronts-rough-cut", 
        "nfc-sides-backs",
        "nfc-sides-fronts"
    ]
    
    # All wrong orientation images (corners and sides) go to 'wrong-orientation' class
    wrong_orientation_folders = [
        "corners-wrong-orientation",
        "sides-wrong-orientation"
    ]
    
    # Copy images from all normal folders (limiting to balance classes)
    print("\nProcessing normal orientation images:")
    max_per_folder = 200  # Lower limit per folder to balance classes
    normal_count = 0
    for folder in normal_folders:
        source = data_path / folder
        if source.exists():
            # Count images in source folder
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            
            # Sample if there are too many
            if folder_count > max_per_folder:
                import random
                sampled_images = random.sample(folder_images, max_per_folder)
                temp_sample_dir = work_path / f"temp_sample_{folder}"
                temp_sample_dir.mkdir(exist_ok=True, parents=True)
                for img in sampled_images:
                    shutil.copy(img, temp_sample_dir / img.name)
                copy_images_to_class([temp_sample_dir], temp_dir, "normal")
                copied_count = len(sampled_images)
            else:
                copy_images_to_class([source], temp_dir, "normal")
                copied_count = folder_count
            
            normal_count += copied_count
            print(f"  - Added {copied_count} images from {folder}")
    
    # Copy all wrong orientation images
    print("\nProcessing wrong-orientation images:")
    wrong_orient_count = 0
    for folder in wrong_orientation_folders:
        source = data_path / folder
        if source.exists():
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            copy_images_to_class([source], temp_dir, "wrong-orientation")
            wrong_orient_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for orientation model:")
    print(f"  Normal orientation images: {normal_count}")
    print(f"  Wrong orientation images: {wrong_orient_count}")
    
    # Train and save model with updated naming convention (01_)
    model_path = models_path / "01_orientation_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=15,  # Fewer epochs for simpler binary task
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_focus(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 02: Card Focus Classification
    Classifies images into: clear or blurry
    """
    print("\n=== Running Test 02: Card Focus Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "focus")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define folder mapping to target classes
    # All standard card images (corners and sides) go to 'clear' class
    clear_folders = [
        "factory-cut-corners-backs", 
        "factory-cut-corners-fronts", 
        "nfc-corners-backs", 
        "nfc-corners-fronts",
        "factory-cut-sides-backs-die-cut", 
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-backs-rough-cut",
        "factory-cut-sides-fronts-rough-cut", 
        "nfc-sides-backs",
        "nfc-sides-fronts"
    ]
    
    # All blurry images (both corners and sides) go to 'blurry' class
    blurry_folders = [
        "corners-blurry",
        "sides-blurry"
    ]
    
    # Copy images from clear folders (limiting to balance classes)
    print("\nProcessing clear images:")
    max_per_folder = 200  # Lower limit per folder to balance classes
    clear_count = 0
    for folder in clear_folders:
        source = data_path / folder
        if source.exists():
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            
            # Sample if there are too many
            if folder_count > max_per_folder:
                import random
                sampled_images = random.sample(folder_images, max_per_folder)
                temp_sample_dir = work_path / f"temp_sample_{folder}"
                temp_sample_dir.mkdir(exist_ok=True, parents=True)
                for img in sampled_images:
                    shutil.copy(img, temp_sample_dir / img.name)
                copy_images_to_class([temp_sample_dir], temp_dir, "clear")
                copied_count = len(sampled_images)
            else:
                copy_images_to_class([source], temp_dir, "clear")
                copied_count = folder_count
            
            clear_count += copied_count
            print(f"  - Added {copied_count} images from {folder}")
    
    # Copy all blurry images
    print("\nProcessing blurry images:")
    blurry_count = 0
    for folder in blurry_folders:
        source = data_path / folder
        if source.exists():
            folder_images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
            folder_count = len(folder_images)
            copy_images_to_class([source], temp_dir, "blurry")
            blurry_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for focus model:")
    print(f"  Clear images: {clear_count}")
    print(f"  Blurry images: {blurry_count}")
    
    # Train and save model
    model_path = models_path / "02_focus_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=15,  # Fewer epochs for simpler binary task
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_corner_front_back(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 10: Corner Front/Back Classification
    Classifies corner images as either front or back
    """
    print("\n=== Running Test 10: Corner Front/Back Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "corner_front_back")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define source folders for fronts and backs
    front_folders = [
        "factory-cut-corners-fronts",
        "nfc-corners-fronts" 
    ]
    
    back_folders = [
        "factory-cut-corners-backs",
        "nfc-corners-backs"
    ]
    
    # Copy front corner images (from both factory and NFC) to 'front' class
    print("\nProcessing front corner images:")
    front_count = 0
    for folder in front_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "front")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            front_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Copy back corner images (from both factory and NFC) to 'back' class
    print("\nProcessing back corner images:")
    back_count = 0
    for folder in back_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "back")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            back_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for corner front/back model:")
    print(f"  Front images: {front_count}")
    print(f"  Back images: {back_count}")
    
    # Train and save model with new numbering convention (10_)
    model_path = models_path / "10_corner_front_back_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=20,
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed for front/back detection
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_side_front_back(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 11: Side Front/Back Classification
    Classifies side images as either front or back regardless of cut type
    """
    print("\n=== Running Test 11: Side Front/Back Classification ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_front_back")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Define source folders for fronts and backs
    front_folders = [
        "factory-cut-sides-fronts-die-cut",
        "factory-cut-sides-fronts-rough-cut",  # Include if it exists
        "nfc-sides-fronts"
    ]
    
    back_folders = [
        "factory-cut-sides-backs-die-cut",
        "factory-cut-sides-backs-rough-cut",  # Include if it exists
        "nfc-sides-backs"
    ]
    
    # Copy front side images to 'front' class
    print("\nProcessing front side images:")
    front_count = 0
    for folder in front_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "front")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            front_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Copy back side images to 'back' class
    print("\nProcessing back side images:")
    back_count = 0
    for folder in back_folders:
        source = data_path / folder
        if source.exists():
            copy_images_to_class([source], temp_dir, "back")
            folder_count = len(list(source.glob("*.jpg")) + list(source.glob("*.png")))
            back_count += folder_count
            print(f"  - Added {folder_count} images from {folder}")
    
    # Summary of class distribution
    print("\nClass distribution for side front/back model:")
    print(f"  Front images: {front_count}")
    print(f"  Back images: {back_count}")
    
    # Train and save model
    model_path = models_path / "11_side_front_back_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=20,
        img_size=(720, 1280),
        enhance_edges_prob=0.0,  # No edge enhancement needed for front/back detection
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_corner_front_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 30: Factory vs NFC Corner Front Classification
    Compares factory-cut and NFC corners on the front of the card
    """
    print("\n=== Running Test 30: Factory vs NFC (Corner Front) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "corner_front_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory corner fronts to 'factory' class
    factory_sources = [data_path / "factory-cut-corners-fronts"]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corner fronts to 'nfc' class
    nfc_sources = [data_path / "nfc-corners-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "30_corner_front_factory_vs_nfc_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_corner_back_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 31: Factory vs NFC Corner Back Classification
    Compares factory-cut and NFC corners on the back of the card
    """
    print("\n=== Running Test 31: Factory vs NFC (Corner Back) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "corner_back_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory corner backs to 'factory' class
    factory_sources = [data_path / "factory-cut-corners-backs"]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corner backs to 'nfc' class
    nfc_sources = [data_path / "nfc-corners-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "31_corner_back_factory_vs_nfc_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_side_front_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 32: Factory vs NFC Side Front Classification
    Compares factory-cut and NFC sides on the front of the card
    """
    print("\n=== Running Test 32: Factory vs NFC (Side Front) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_front_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory side fronts to 'factory' class (combining die-cut and rough-cut)
    factory_sources = [
        data_path / "factory-cut-sides-fronts-die-cut",
        data_path / "factory-cut-sides-fronts-rough-cut"  # Include if it exists
    ]
    # Filter out non-existent paths
    factory_sources = [p for p in factory_sources if p.exists()]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC side fronts to 'nfc' class
    nfc_sources = [data_path / "nfc-sides-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "32_side_front_factory_vs_nfc_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
    return learn

def test_side_back_factory_vs_nfc(data_path, work_path, models_path, resume=False, recalculate_lr=False):
    """
    Test 33: Factory vs NFC Side Back Classification
    Compares factory-cut and NFC sides on the back of the card
    """
    print("\n=== Running Test 33: Factory vs NFC (Side Back) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "side_back_factory_vs_nfc")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory side backs to 'factory' class (combining die-cut and rough-cut)
    factory_sources = [
        data_path / "factory-cut-sides-backs-die-cut",
        data_path / "factory-cut-sides-backs-rough-cut"  # Include if it exists
    ]
    # Filter out non-existent paths
    factory_sources = [p for p in factory_sources if p.exists()]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC side backs to 'nfc' class
    nfc_sources = [data_path / "nfc-sides-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "33_side_back_factory_vs_nfc_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint,
        max_rotate=1.0,  # Minimal rotation as requested
        recalculate_lr=recalculate_lr
    )
    
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

def is_test_completed(models_path, model_name):
    """Check if a test has already successfully completed by looking for the output model file"""
    model_file = models_path / f"{model_name}.pkl"
    return model_file.exists()

def main():
    """Main function to run all tests"""
    # Parse command line arguments with improved help information
    parser = argparse.ArgumentParser(
        description='NFC Card Detector Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run all tests
  python 01_run_image_tests.py
  
  # Resume from previous run
  python 01_run_image_tests.py --resume
  
  # Skip completed models
  python 01_run_image_tests.py --skip-completed
  
  # Train only the orientation model
  python 01_run_image_tests.py --only orientation
  
  # Force recalculation of learning rates
  python 01_run_image_tests.py --recalculate-lr
'''
    )
    
    # Core arguments
    parser.add_argument('-r', '--resume', action='store_true', 
                      help='Resume from last run and skip completed tests')
    parser.add_argument('-s', '--skip-completed', action='store_true', 
                      help='Skip tests that have already successfully completed')
    parser.add_argument('--recalculate-lr', action='store_true',
                      help='Force recalculation of optimal learning rates')
    
    # Model selection argument
    test_group = parser.add_argument_group('Model Selection')
    test_group.add_argument('-o', '--only', type=str, 
                          choices=['orientation', 'focus', 'corner-front-back', 'side-front-back', 
                                  'corner-front', 'corner-back', 'side-front', 'side-back'],
                          help='Run only a specific test')
    
    # Display available models with descriptions
    models_group = parser.add_argument_group('Available Models')
    models_group.add_argument('--list-models', action='store_true',
                            help='List all available models with descriptions')
    
    args = parser.parse_args()
    
    # Handle the list-models argument first if specified
    if hasattr(args, 'list_models') and args.list_models:
        print("\nAvailable Models for Training:")
        print("-----------------------------")
        print("orientation          : Card orientation check (01) - Detects normal vs wrong orientation")
        print("focus                : Card focus check (02) - Detects clear vs blurry images")
        print("corner-front-back    : Corner front/back classifier (10) - Distinguishes front vs back for corners")
        print("side-front-back      : Side front/back classifier (11) - Distinguishes front vs back for sides")
        print("corner-front         : Corner front factory vs NFC (30) - Detects if a front corner is factory-cut or NFC")
        print("corner-back          : Corner back factory vs NFC (31) - Detects if a back corner is factory-cut or NFC")
        print("side-front           : Side front factory vs NFC (32) - Detects if a front side is factory-cut or NFC")
        print("side-back            : Side back factory vs NFC (33) - Detects if a back side is factory-cut or NFC")
        print("\nFor more information, run: python 01_run_image_tests.py -h")
        return
    
    print("Starting NFC Card Detector Training")
    print("===================================")
    
    # Set up distinct directories for different purposes
    data_path = Path("data").resolve()                   # Source images only
    work_path = Path("nfc_detector_work_dir").resolve()  # Working directory for temp files
    models_path = Path("nfc_models").resolve()           # Model output directory
    
    print(f"Data source path: {data_path}")
    print(f"Working directory: {work_path}")
    print(f"Models output directory: {models_path}")
    
    # Define current model filenames
    model_files = {
        "orientation": "01_orientation_model.pkl",
        "focus": "02_focus_model.pkl",
        "corner_front_back": "10_corner_front_back_model.pkl", 
        "side_front_back": "11_side_front_back_model.pkl",
        "corner_front_factory_vs_nfc": "30_corner_front_factory_vs_nfc_model.pkl",
        "corner_back_factory_vs_nfc": "31_corner_back_factory_vs_nfc_model.pkl",
        "side_front_factory_vs_nfc": "32_side_front_factory_vs_nfc_model.pkl",
        "side_back_factory_vs_nfc": "33_side_back_factory_vs_nfc_model.pkl"
    }
    
    # Check which tests have already been completed (if --resume or --skip-completed)
    completed_tests = []
    if args.resume or args.skip_completed:
        for test_name, model_file in model_files.items():
            if is_test_completed(models_path, model_file.replace('.pkl', '')):
                completed_tests.append(test_name)
                print(f"✓ {test_name.replace('_', ' ')} test has already completed successfully")
    
    # Check if work directory exists and offer to resume from checkpoints
    resume_training = False
    if work_path.exists() and (work_path / "model_checkpoints").exists():
        checkpoints = list((work_path / "model_checkpoints").glob('best_model_stage*'))
        if checkpoints and not args.resume:  # Don't ask if --resume was passed
            print(f"Found {len(checkpoints)} existing checkpoints in {work_path}")
            response = input("Do you want to resume from these checkpoints? (y/n): ").lower()
            resume_training = response == 'y'
        elif args.resume:
            resume_training = True
            print("Resuming from checkpoints (--resume flag detected)")
            
        if not resume_training and not args.resume:
            # User doesn't want to resume, clean up old working directory
            clean_work_dir(work_path)
    
    # Verify all required directories exist
    if not verify_directories(data_path, work_path, models_path):
        print("Directory verification failed. Exiting.")
        sys.exit(1)
    
    # Check GPU status before starting
    check_gpu_memory()
    
    success = True  # Track if all tests completed successfully
    
    try:
        # Run tests with explicit try/except for each
        print("\nBeginning test series...")
        
        # Run only specified test if --only is used
        if args.only:
            if args.only == 'orientation' and "orientation" not in completed_tests:
                test_orientation(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'focus' and "focus" not in completed_tests:
                test_focus(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'corner-front-back' and "corner_front_back" not in completed_tests:
                test_corner_front_back(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'side-front-back' and "side_front_back" not in completed_tests:
                test_side_front_back(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'corner-front' and "corner_front_factory_vs_nfc" not in completed_tests:
                test_corner_front_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'corner-back' and "corner_back_factory_vs_nfc" not in completed_tests:
                test_corner_back_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'side-front' and "side_front_factory_vs_nfc" not in completed_tests:
                test_side_front_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            elif args.only == 'side-back' and "side_back_factory_vs_nfc" not in completed_tests:
                test_side_back_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            else:
                print(f"Test '{args.only}' is already completed or invalid.")
            return
        
        # Otherwise run all tests that aren't completed
        
        # Orientation test (01)
        if "orientation" not in completed_tests:
            try:
                print("\nRunning orientation test (01)...")
                test_orientation(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in orientation test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping orientation test (already completed)")
            
        # Focus test (02)
        if "focus" not in completed_tests:
            try:
                print("\nRunning focus test (02)...")
                test_focus(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in focus test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping focus test (already completed)")
        
        # Corner front/back test (10)
        if "corner_front_back" not in completed_tests:
            try:
                print("\nRunning corner front/back test (10)...")
                test_corner_front_back(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in corner front/back test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping corner front/back test (already completed)")
            
        # Side front/back test (11)
        if "side_front_back" not in completed_tests:
            try:
                print("\nRunning side front/back test (11)...")
                test_side_front_back(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in side front/back test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping side front/back test (already completed)")
        
        # Corner front factory vs NFC test (30)
        if "corner_front_factory_vs_nfc" not in completed_tests:
            try:
                print("\nRunning corner front factory vs NFC test (30)...")
                test_corner_front_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in corner front factory vs NFC test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping corner front factory vs NFC test (already completed)")
            
        # Corner back factory vs NFC test (31)
        if "corner_back_factory_vs_nfc" not in completed_tests:
            try:
                print("\nRunning corner back factory vs NFC test (31)...")
                test_corner_back_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in corner back factory vs NFC test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping corner back factory vs NFC test (already completed)")
            
        # Side front factory vs NFC test (32)
        if "side_front_factory_vs_nfc" not in completed_tests:
            try:
                print("\nRunning side front factory vs NFC test (32)...")
                test_side_front_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in side front factory vs NFC test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping side front factory vs NFC test (already completed)")
            
        # Side back factory vs NFC test (33)
        if "side_back_factory_vs_nfc" not in completed_tests:
            try:
                print("\nRunning side back factory vs NFC test (33)...")
                test_side_back_factory_vs_nfc(data_path, work_path, models_path, resume_training, args.recalculate_lr)
            except Exception as e:
                success = False
                print(f"Error in side back factory vs NFC test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping side back factory vs NFC test (already completed)")
            
        # Count the current set of tests
        expected_test_count = len(model_files)
        if success and len(completed_tests) < expected_test_count:
            print("All tests completed successfully!")
        elif success:
            print("No tests were run (all were already completed)")
        else:
            print("Some tests failed - see errors above")
    
    finally:
        # Clean up working directory if all tests were successful
        if success and not args.resume:  # Don't clean up if using --resume flag
            print("\nAll tests completed successfully - cleaning up working directory...")
            clean_work_dir(work_path)
        elif not success:
            print("\nSome tests failed - preserving working directory for inspection")
            print(f"Working directory: {work_path}")
            print("You can resume using: python 01_run_image_tests.py --resume")

if __name__ == "__main__":
    main()