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

def test_combined_corners(data_path, work_path, models_path, resume=False):
    """
    Test 3: Factory corners (backs + fronts) vs NFC corners (backs + fronts)
    """
    print("\n=== Running Test 3: Factory vs NFC (Combined Front/Back) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "combined_corners")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory corners (both front and back) to 'factory' class
    factory_sources = [
        data_path / "factory-cut-corners-backs",
        data_path / "factory-cut-corners-fronts"
    ]
    copy_images_to_class(factory_sources, temp_dir, "factory")
    
    # Copy NFC corners (both front and back) to 'nfc' class
    nfc_sources = [
        data_path / "nfc-corners-backs",
        data_path / "nfc-corners-fronts"
    ]
    copy_images_to_class(nfc_sources, temp_dir, "nfc")
    
    # Train and save model with enhanced settings
    model_path = models_path / "combined_corners_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint
    )
    
    # We'll clean up everything at the end, not here
    return learn

def test_fronts_only(data_path, work_path, models_path, resume=False):
    """
    Test 1: Factory fronts vs NFC fronts
    """
    print("\n=== Running Test 1: Factory vs NFC (Fronts Only) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "fronts_only")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory fronts to 'factory_front' class
    factory_sources = [data_path / "factory-cut-corners-fronts"]
    copy_images_to_class(factory_sources, temp_dir, "factory_front")
    
    # Copy NFC fronts to 'nfc_front' class
    nfc_sources = [data_path / "nfc-corners-fronts"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_front")
    
    # Train and save model with enhanced settings
    model_path = models_path / "fronts_only_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path,
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint
    )
    
    # We'll clean up everything at the end, not here
    return learn
    
def test_backs_only(data_path, work_path, models_path, resume=False):
    """
    Test 2: Factory backs vs NFC backs
    """
    print("\n=== Running Test 2: Factory vs NFC (Backs Only) ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "backs_only")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy factory backs to 'factory_back' class
    factory_sources = [data_path / "factory-cut-corners-backs"]
    copy_images_to_class(factory_sources, temp_dir, "factory_back")
    
    # Copy NFC backs to 'nfc_back' class
    nfc_sources = [data_path / "nfc-corners-backs"]
    copy_images_to_class(nfc_sources, temp_dir, "nfc_back")
    
    # Train and save model with enhanced settings
    model_path = models_path / "backs_only_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path,
        epochs=25,
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True, 
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint
    )
    
    # We'll clean up everything at the end, not here
    return learn

def test_all_categories(data_path, work_path, models_path, resume=False):
    """
    Test 4: All four categories as separate classes (factory fronts, factory backs, NFC fronts, NFC backs)
    """
    print("\n=== Running Test 4: All Categories Separate ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, "all_categories")
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory in work_path
    temp_dir = setup_temp_dir(work_path)
    
    # Copy each category to its own class
    copy_images_to_class([data_path / "factory-cut-corners-fronts"], temp_dir, "factory_fronts")
    copy_images_to_class([data_path / "factory-cut-corners-backs"], temp_dir, "factory_backs")
    copy_images_to_class([data_path / "nfc-corners-fronts"], temp_dir, "nfc_fronts")
    copy_images_to_class([data_path / "nfc-corners-backs"], temp_dir, "nfc_backs")
    
    # Train and save model with enhanced settings
    model_path = models_path / "all_categories_model.pkl"
    learn = train_and_save_model(
        temp_dir, 
        model_path,
        work_path, 
        epochs=30,  # Slightly more epochs for this more complex task
        img_size=(720, 1280),
        enhance_edges_prob=0.3,
        use_tta=True,
        progressive_resizing=False,
        resume_from_checkpoint=checkpoint
    )
    
    # We'll clean up everything at the end, not here
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NFC card detection tests')
    parser.add_argument('--resume', action='store_true', help='Resume from last run and skip completed tests')
    parser.add_argument('--skip-completed', action='store_true', help='Skip tests that have completed successfully')
    args = parser.parse_args()
    
    print("Starting NFC Card Detector Training")
    print("===================================")
    
    # Set up distinct directories for different purposes
    data_path = Path("data").resolve()                   # Source images only
    work_path = Path("nfc_detector_work_dir").resolve()  # Working directory for temp files
    models_path = Path("nfc_models").resolve()           # Model output directory
    
    print(f"Data source path: {data_path}")
    print(f"Working directory: {work_path}")
    print(f"Models output directory: {models_path}")
    
    # Check which tests have already been completed (if --resume or --skip-completed)
    completed_tests = []
    if args.resume or args.skip_completed:
        if is_test_completed(models_path, "fronts_only_model"):
            completed_tests.append("fronts_only")
            print("✓ Fronts-only test has already completed successfully")
            
        if is_test_completed(models_path, "backs_only_model"):
            completed_tests.append("backs_only")
            print("✓ Backs-only test has already completed successfully")
            
        if is_test_completed(models_path, "combined_corners_model"):
            completed_tests.append("combined_corners")
            print("✓ Combined-corners test has already completed successfully")
            
        if is_test_completed(models_path, "all_categories_model"):
            completed_tests.append("all_categories")
            print("✓ All-categories test has already completed successfully")
    
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
        
        # Fronts-only test
        if "fronts_only" not in completed_tests:
            try:
                print("\nRunning fronts-only test...")
                test_fronts_only(data_path, work_path, models_path, resume_training)
            except Exception as e:
                success = False
                print(f"Error in fronts-only test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping fronts-only test (already completed)")
        
        # Backs-only test
        if "backs_only" not in completed_tests:
            try:
                print("\nRunning backs-only test...")
                test_backs_only(data_path, work_path, models_path, resume_training)
            except Exception as e:
                success = False
                print(f"Error in backs-only test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping backs-only test (already completed)")
        
        # Combined-corners test
        if "combined_corners" not in completed_tests:
            try:
                print("\nRunning combined-corners test...")
                test_combined_corners(data_path, work_path, models_path, resume_training)
            except Exception as e:
                success = False
                print(f"Error in combined-corners test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping combined-corners test (already completed)")
        
        # All-categories test
        if "all_categories" not in completed_tests:
            try:
                print("\nRunning all-categories test...")
                test_all_categories(data_path, work_path, models_path, resume_training)
            except Exception as e:
                success = False
                print(f"Error in all-categories test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping all-categories test (already completed)")
        
        if success and len(completed_tests) < 4:  # Updated to check for 4 tests
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
            print("You can resume using: python run_image_tests.py --resume")

if __name__ == "__main__":
    main()
