import argparse
from pathlib import Path
import os
import sys
import shutil
from image_test_utils import train_and_save_model
from utils.directory_utils import (verify_directories, clean_work_dir, 
                                  find_latest_checkpoint, is_test_completed, 
                                  setup_temp_dir)
from utils.dataset_utils import copy_images_to_class, balanced_copy_images
from utils.model_utils import check_gpu_memory

# Import modularized tests with valid Python module names
from utils.models.m01_orientation_classification import test_orientation
from utils.models.m02_focus_classification import test_focus
from utils.models.m10_corner_front_back_classification import test_corner_front_back
from utils.models.m11_side_front_back_classification import test_side_front_back
from utils.models.m30_corner_front_factory_vs_nfc_classification import test_corner_front_factory_vs_nfc
from utils.models.m31_corner_back_factory_vs_nfc_classification import test_corner_back_factory_vs_nfc
from utils.models.m32_side_front_factory_vs_nfc_classification import test_side_front_factory_vs_nfc
from utils.models.m33_side_back_factory_vs_nfc_classification import test_side_back_factory_vs_nfc

from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

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