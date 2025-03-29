import argparse
from pathlib import Path
import os
import sys
import shutil
import importlib
import pkgutil
from image_test_utils import train_and_save_model
from utils.directory_utils import (verify_directories, clean_work_dir, 
                                  find_latest_checkpoint, is_test_completed, 
                                  setup_temp_dir)
from utils.dataset_utils import copy_images_to_class, balanced_copy_images
from utils.model_utils import check_gpu_memory

# Instead of static imports, we'll dynamically discover model modules
import utils.models
from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

def discover_models():
    """Discover all model modules in the utils/models package and extract their metadata"""
    models = {}
    model_choices = []
    
    # Find all model modules in the utils.models package
    for _, name, _ in pkgutil.iter_modules(utils.models.__path__):
        if name.startswith('m') and '_' in name:
            try:
                # Import the module
                module = importlib.import_module(f'utils.models.{name}')
                
                # Extract metadata
                model_name = getattr(module, 'MODEL_NAME', name)
                model_number = getattr(module, 'MODEL_NUMBER', '00')
                model_description = getattr(module, 'MODEL_DESCRIPTION', 'No description available')
                
                # Get the main test function
                # Assuming the function is named test_X where X is the model_name
                test_func = getattr(module, f'test_{model_name}', None)
                
                if test_func:
                    # Convert to command-line friendly format (for choices)
                    cli_name = model_name.replace('_', '-')
                    model_choices.append(cli_name)
                    
                    # Store model info
                    models[cli_name] = {
                        'module': module,
                        'function': test_func,
                        'name': model_name,
                        'number': model_number,
                        'description': model_description,
                        'filename': f"{model_number}_{model_name}_model.pkl"
                    }
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load model from {name}: {e}")
    
    return models, sorted(model_choices)

def main():
    """Main function to run all tests"""
    # Discover available models
    models, model_choices = discover_models()
    
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
                          choices=model_choices,
                          help='Run only a specific test')
    
    # Display available models with descriptions
    models_group = parser.add_argument_group('Available Models')
    models_group.add_argument('--list-models', action='store_true',
                            help='List all available models with descriptions')
    
    args = parser.parse_args()
    
    # Handle the list-models argument first if specified
    if args.list_models:
        print("\nAvailable Models for Training:")
        print("-----------------------------")
        for cli_name, model_info in sorted(models.items(), key=lambda x: x[1]['number']):
            print(f"{cli_name:<20}: {model_info['description']} ({model_info['number']})")
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
    
    # Define model filenames based on discovered models
    model_files = {model_info['name']: model_info['filename'] for _, model_info in models.items()}
    
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
            model_info = models.get(args.only)
            if model_info and model_info['name'] not in completed_tests:
                print(f"\nRunning {model_info['name']} test ({model_info['number']})...")
                model_info['function'](data_path, work_path, models_path, resume_training, args.recalculate_lr)
            else:
                print(f"Test '{args.only}' is already completed or invalid.")
            return
        
        # Otherwise run all tests that aren't completed
        for cli_name, model_info in sorted(models.items(), key=lambda x: x[1]['number']):
            model_name = model_info['name']
            if model_name not in completed_tests:
                try:
                    print(f"\nRunning {model_name} test ({model_info['number']})...")
                    model_info['function'](data_path, work_path, models_path, resume_training, args.recalculate_lr)
                except Exception as e:
                    success = False
                    print(f"Error in {model_name} test: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\nSkipping {model_name} test (already completed)")
            
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