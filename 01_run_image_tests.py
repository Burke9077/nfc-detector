import argparse
from pathlib import Path
import os
import sys
import shutil
import importlib
import pkgutil
from utils.model_utils import train_and_save_model
from utils.directory_utils import (verify_directories, clean_work_dir, 
                                  find_latest_checkpoint, is_test_completed, 
                                  setup_temp_dir)
from utils.dataset_utils import copy_images_to_class, balanced_copy_images
from utils.model_utils import check_gpu_memory 
from tabulate import tabulate
from utils.model_metadata_utils import load_model_metadata, format_metadata_for_display
from utils.model_performance_utils import (get_best_accuracy_from_metadata, get_accuracy_emoji,
                                          get_model_quality_category, format_performance_change,
                                          compare_model_performance, format_metrics_with_emoji,
                                          should_rerun_model_by_quality)
from utils.model_discovery_utils import discover_models, determine_model_category, list_models

# Default configuration values
DEFAULT_MAX_IMAGES_PER_CLASS = 8000

# Store model performance changes to display at the end
model_performance_tracker = {}

def display_model_performance_summary():
    """Display a summary table of model performance changes"""
    if not model_performance_tracker:
        print("\nNo model performance changes to report.")
        return
    
    print("\nModel Performance Summary:")
    print("=========================")
    
    # Table headers
    headers = ["Model", "New Score", "Previous Score", "Change", "Secondary Improvements"]
    table_data = []
    
    # Add rows for each model
    for model_name, perf in sorted(model_performance_tracker.items()):
        # Format model name for display
        display_name = model_name.replace('_', ' ').title()
        
        # Get emoji for new accuracy
        new_emoji = get_accuracy_emoji(perf['new_accuracy'])
        new_score = f"{new_emoji} {perf['new_accuracy']:.4f}" if perf['new_accuracy'] is not None else "N/A"
        
        # Get emoji for old accuracy (if available)
        old_emoji = get_accuracy_emoji(perf['old_accuracy'])
        old_score = f"{old_emoji} {perf['old_accuracy']:.4f}" if perf['old_accuracy'] is not None else "N/A"
        
        # Format the change with sign and percent
        change_display = format_performance_change(perf['change'], perf['percent_change'])
        
        # Get secondary improvements if any
        secondary = perf.get('secondary_improvement', 'None')
        
        # Add row to table
        table_data.append([display_name, new_score, old_score, change_display, secondary])
    
    # Display the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Show summary statistics
    improvements = sum(1 for perf in model_performance_tracker.values() if perf['is_improvement'])
    no_change = sum(1 for perf in model_performance_tracker.values() if perf['change'] == 0)
    regressions = sum(1 for perf in model_performance_tracker.values() if perf['change'] is not None and perf['change'] < 0)
    new_models = sum(1 for perf in model_performance_tracker.values() if perf['old_accuracy'] is None)
    
    print(f"\nSummary: {improvements} improvements, {regressions} regressions, {no_change} unchanged, {new_models} new models")

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
  
  # Train multiple specific models
  python 01_run_image_tests.py --only {example_model},corner-front-back
  
  # Retrain all models with poor or mediocre quality
  python 01_run_image_tests.py --run-below-quality mediocre
  
  # Force recalculation of learning rates
  python 01_run_image_tests.py --recalculate-lr
  
  # Force overwrite existing models even if they have better performance
  python 01_run_image_tests.py --force-overwrite
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
                          help='Run specific test(s), separated by commas if multiple')
    test_group.add_argument('--run-below-quality', type=str, 
                          choices=['good', 'acceptable', 'mediocre', 'poor'],
                          help='Run all models below a certain quality threshold')
    
    # Display available models with descriptions
    models_group = parser.add_argument_group('Available Models')
    models_group.add_argument('--list-models', action='store_true',
                            help='List all available models with descriptions')
    
    # Add the force-overwrite flag with improved description
    parser.add_argument('--force-overwrite', action='store_true', 
                        help='Force overwrite existing models even if they have better metrics')
    
    return parser.parse_args()

def main():
    """Main function to run all tests"""
    args = parse_args()
    
    # Discover available models
    models, model_choices = discover_models()
    
    # Reset the performance tracker at the start of each run
    global model_performance_tracker
    model_performance_tracker = {}
    
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
    
    # Build a list of models to run based on quality if --run-below-quality is specified
    models_to_run = []
    if args.run_below_quality:
        print(f"\nChecking for models below '{args.run_below_quality}' quality...")
        quality_threshold = args.run_below_quality
        for cli_name, model_info in models.items():
            model_name = model_info['name']
            model_path = models_path / f"{model_info['number']}_{model_name}_model.pkl"
            if should_rerun_model_by_quality(model_path, quality_threshold):
                models_to_run.append(cli_name)
                if model_path.exists():
                    print(f"Model {model_name} exists but quality is below threshold - will retrain")
                else:
                    print(f"Model {model_name} doesn't exist - will train")
        
        if not models_to_run:
            print("No models found that need retraining based on quality threshold.")
            return
    
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
        
        # Run only specified test(s) if --only is used
        if args.only:
            # Split by comma to handle multiple models
            requested_models = [m.strip() for m in args.only.split(',')]
            
            for model_name in requested_models:
                model_info = models.get(model_name)
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
                    print(f"Test '{model_name}' is already completed or invalid.")
            return
        
        # Run models based on quality threshold, if specified
        if args.run_below_quality:
            for cli_name in models_to_run:
                model_info = models.get(cli_name)
                model_name = model_info['name']
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
        # Display performance summary before cleaning up
        if model_performance_tracker:
            display_model_performance_summary()
            
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