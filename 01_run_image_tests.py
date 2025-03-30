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
from tabulate import tabulate
from utils.model_metadata_utils import load_model_metadata, format_metadata_for_display

# Default configuration values
DEFAULT_MAX_IMAGES_PER_CLASS = 8000

# Dynamically discover model modules
import utils.models
from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

def determine_model_category(model_number):
    """Determine the model category based on model number"""
    try:
        # Extract first two digits and convert to integer for range comparison
        prefix = int(model_number[:2]) if len(model_number) >= 2 else -1
        
        # Categorize based on ranges
        if 0 <= prefix <= 9:  # 00-09 range
            return "QC & Prep"
        elif 10 <= prefix <= 19:  # 10-19 range
            return "Front/Back Detection"
        elif 30 <= prefix <= 39:  # 30-39 range
            return "Cut Classification"
        else:
            return "Other"
    except ValueError:
        # Handle case where model_number doesn't start with digits
        return "Other"

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
                
                # Get category based on model number
                model_category = getattr(module, 'MODEL_CATEGORY', 
                                        determine_model_category(model_number))
                
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
                        'category': model_category,
                        'filename': f"{model_number}_{model_name}_model.pkl"
                    }
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load model from {name}: {e}")
    
    return models, sorted(model_choices)

def parse_args():
    """Parse command line arguments"""
    # Discover available models
    models, model_choices = discover_models()
    
    # Select a sample model name for the help text example (first one in sorted list)
    example_model = model_choices[0] if model_choices else "example-model"
    
    # Parse command line arguments with improved help information
    parser = argparse.ArgumentParser(
        description='NFC Card Detector Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Examples:
  # Run all tests
  python 01_run_image_tests.py
  
  # Resume from previous run
  python 01_run_image_tests.py --resume
  
  # Skip completed models
  python 01_run_image_tests.py --skip-completed
  
  # Train only a specific model
  python 01_run_image_tests.py --only {example_model}
  
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
    parser.add_argument('--max-images', type=int, default=os.environ.get('MAX_IMAGES_PER_CLASS', DEFAULT_MAX_IMAGES_PER_CLASS),
                      help=f'Maximum images per class to use (default: {DEFAULT_MAX_IMAGES_PER_CLASS}, can also set MAX_IMAGES_PER_CLASS env var)')
    
    # Model selection argument
    test_group = parser.add_argument_group('Model Selection')
    test_group.add_argument('-o', '--only', type=str, 
                          choices=model_choices,
                          help='Run only a specific test')
    
    # Display available models with descriptions
    models_group = parser.add_argument_group('Available Models')
    models_group.add_argument('--list-models', action='store_true',
                            help='List all available models with descriptions')
    
    # Add the force-overwrite flag
    parser.add_argument('--force-overwrite', action='store_true', 
                        help='Force overwrite existing models even if they have better metrics')
    
    return parser.parse_args()

def get_accuracy_emoji(accuracy):
    """Return an emoji based on the accuracy value"""
    if accuracy is None:
        return ""
    
    try:
        accuracy = float(accuracy)
        if accuracy > 0.999:
            return "😄"  # Really happy face for excellent accuracy
        elif accuracy > 0.97:
            return "🙂"  # Slightly smiling face for good accuracy
        elif accuracy > 0.90:
            return "😐"  # Straight face for acceptable accuracy
        elif accuracy > 0.80:
            return "🙁"  # Frowny face for mediocre accuracy
        else:
            return "😠"  # Angry face for poor accuracy
    except (ValueError, TypeError):
        return ""

def format_metrics_with_emoji(metadata):
    """Format metadata with emoji indicators for accuracy"""
    base_display = format_metadata_for_display(metadata)
    
    # If no metadata or metrics, return empty string
    if not metadata or 'metrics' not in metadata:
        return base_display
    
    # Get accuracy from metrics
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        # Use TTA accuracy if regular accuracy isn't available
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    # Get appropriate emoji based on accuracy
    emoji = get_accuracy_emoji(accuracy)
    
    # Add emoji to the display
    if emoji and base_display:
        return f"{emoji} {base_display}"
    
    return base_display

def list_models(models_path, available_models):
    """List all available models with their metadata"""
    print("\nAvailable Models:")
    print("================")
    
    # Table headers
    headers = ["Model Name", "Number", "Category", "Status", "CLI Parameter", "Metrics"]
    table_data = []
    
    # Sort models by their number for consistent display
    for cli_param, model_info in sorted(available_models.items(), key=lambda x: x[1]['number']):
        display_name = model_info['name'].replace('_', ' ').title()
        model_number = model_info['number']
        model_path = models_path / f"{model_number}_{model_info['name']}_model.pkl"
        
        # Determine category based on model number
        category = determine_model_category(model_number)
        
        # Check if model exists and get status
        if model_path.exists():
            status = "✓ Trained"
        else:
            status = "✗ Not trained"
        
        # Default empty metrics display
        metrics_display = ""
        
        # Get metrics from model metadata if available
        if model_path.exists():
            metadata = load_model_metadata(model_path)
            # Use the new function to get metrics with emoji
            metrics_display = format_metrics_with_emoji(metadata)
        
        # Add row to table data with CLI parameter
        table_data.append([display_name, model_number, category, status, cli_param, metrics_display])
    
    # Display the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\nAccuracy ratings: 😄 (>99.9%), 🙂 (97-99.9%), 😐 (90-97%), 🙁 (80-90%), 😠 (<80%)")
    print("\nTo train a specific model: python 01_run_image_tests.py --only MODEL-NAME")
    print("For more information, run: python 01_run_image_tests.py -h")

def main():
    """Main function to run all tests"""
    args = parse_args()
    
    # Discover available models
    models, model_choices = discover_models()
    
    # Handle the list-models argument first if specified
    if args.list_models:
        list_models(Path("nfc_models").resolve(), models)
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
    
    # Check GPU status, enforce CUDA and check memory
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
                # Use function signature inspection to only pass appropriate arguments
                try:
                    # Try with all parameters
                    model_info['function'](data_path, work_path, models_path, 
                                          resume_training, args.recalculate_lr, args.force_overwrite)
                except TypeError as e:
                    # If that fails, try without force_overwrite
                    if "takes from" in str(e) and "but 6 were given" in str(e):
                        print(f"Warning: {model_info['name']} function doesn't support force_overwrite parameter")
                        model_info['function'](data_path, work_path, models_path, 
                                             resume_training, args.recalculate_lr)
            else:
                print(f"Test '{args.only}' is already completed or invalid.")
            return
        
        # Otherwise run all tests that aren't completed
        for cli_name, model_info in sorted(models.items(), key=lambda x: x[1]['number']):
            model_name = model_info['name']
            if model_name not in completed_tests:
                try:
                    print(f"\nRunning {model_name} test ({model_info['number']})...")
                    try:
                        # Try with all parameters
                        model_info['function'](data_path, work_path, models_path, 
                                             resume_training, args.recalculate_lr, args.force_overwrite)
                    except TypeError as e:
                        # If that fails, try without force_overwrite
                        if "takes from" in str(e) and "but 6 were given" in str(e):
                            print(f"Warning: {model_name} function doesn't support force_overwrite parameter")
                            model_info['function'](data_path, work_path, models_path, 
                                                 resume_training, args.recalculate_lr)
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