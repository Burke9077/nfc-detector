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
# Import ML configuration settings
from utils.ml_config import (optimize_cuda_settings, get_recommended_architecture, 
                           get_optimal_batch_size, DEFAULT_ARCHITECTURE, DEFAULT_EPOCHS)

# Default configuration values
DEFAULT_MAX_IMAGES_PER_CLASS = 8000

# Dynamically discover model modules
import utils.models
from fastai.vision.all import *
from collections import Counter
import torch.cuda as cuda
import matplotlib.pyplot as plt

# Apply CUDA optimizations at startup
optimize_cuda_settings()

# Store model performance changes to display at the end
model_performance_tracker = {}

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

def compare_model_performance(new_model_metadata, existing_model_path, model_name):
    """
    Compare the performance of a newly trained model with an existing one.
    Returns True if new model should replace the existing one.
    Also tracks performance changes for later reporting.
    """
    # If existing model doesn't exist, always save new model
    if not existing_model_path.exists():
        # Store info about new model (no comparison available)
        model_performance_tracker[model_name] = {
            'new_accuracy': get_best_accuracy_from_metadata(new_model_metadata),
            'old_accuracy': None,
            'change': None,
            'percent_change': None,
            'is_improvement': True  # New model is always an improvement if no previous model
        }
        return True
    
    # Load existing model metadata
    existing_metadata = load_model_metadata(existing_model_path)
    
    # If existing metadata is invalid or missing metrics, replace it
    if not existing_metadata or 'metrics' not in existing_metadata:
        model_performance_tracker[model_name] = {
            'new_accuracy': get_best_accuracy_from_metadata(new_model_metadata),
            'old_accuracy': None,
            'change': None,
            'percent_change': None,
            'is_improvement': True
        }
        return True
    
    # If new metadata is invalid, don't replace existing model
    if not new_model_metadata or 'metrics' not in new_model_metadata:
        print("Warning: New model has no metrics. Keeping existing model.")
        return False
    
    # Get accuracy metrics from both models
    new_accuracy = get_best_accuracy_from_metadata(new_model_metadata)
    existing_accuracy = get_best_accuracy_from_metadata(existing_metadata)
    
    # Calculate accuracy change
    accuracy_change = new_accuracy - existing_accuracy
    percent_change = (accuracy_change / existing_accuracy) * 100 if existing_accuracy > 0 else 0
    
    # Store performance comparison info
    model_performance_tracker[model_name] = {
        'new_accuracy': new_accuracy,
        'old_accuracy': existing_accuracy,
        'change': accuracy_change,
        'percent_change': percent_change,
        'is_improvement': new_accuracy > existing_accuracy
    }
    
    # Compare accuracies and decide
    if new_accuracy > existing_accuracy:
        print(f"New model accuracy ({new_accuracy:.4f}) is better than existing ({existing_accuracy:.4f}). Replacing model.")
        return True
    else:
        print(f"New model accuracy ({new_accuracy:.4f}) is not better than existing ({existing_accuracy:.4f}). Keeping existing model.")
        print("To force overwrite, use --force-overwrite flag.")
        return False

def get_best_accuracy_from_metadata(metadata):
    """Extract the best accuracy from model metadata"""
    if not metadata or 'metrics' not in metadata:
        return 0.0
    
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    try:
        return float(accuracy) if accuracy is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def format_performance_change(change, percent_change):
    """Format the accuracy change with appropriate sign and color indicators"""
    if change is None:
        return "N/A"
    
    change_sign = '+' if change >= 0 else ''
    return f"{change_sign}{change:.4f} ({change_sign}{percent_change:.2f}%)"

def display_model_performance_summary():
    """Display a summary table of model performance changes"""
    if not model_performance_tracker:
        print("\nNo model performance changes to report.")
        return
    
    print("\nModel Performance Summary:")
    print("=========================")
    
    # Table headers
    headers = ["Model", "New Score", "Previous Score", "Change (Absolute/Percent)"]
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
        
        # Add row to table
        table_data.append([display_name, new_score, old_score, change_display])
    
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
  
  # Automatically try multiple architectures and select the best one
  python 01_run_image_tests.py --auto-architecture
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
    
    # Architecture auto-selection arguments
    arch_group = parser.add_argument_group('Architecture Selection')
    arch_group.add_argument('--auto-architecture', action='store_true',
                          help='Automatically try multiple architectures and select the best one')
    arch_group.add_argument('--max-architectures', type=int, default=3,
                          help='Maximum number of architectures to try (default: 3)')
    
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

def get_accuracy_emoji(accuracy):
    """Return an emoji based on the accuracy value"""
    if accuracy is None:
        return ""
    
    try:
        accuracy = float(accuracy)
        if accuracy > 0.999:
            return "✅"  # Green check box for excellent accuracy
        elif accuracy > 0.97:
            return "😄"  # Slightly smiling face for good accuracy
        elif accuracy > 0.90:
            return "😐"  # Straight face for acceptable accuracy
        elif accuracy > 0.80:
            return "🙁"  # Frowny face for mediocre accuracy
        else:
            return "😭"  # Crying face for poor accuracy
    except (ValueError, TypeError):
        return ""

def get_model_quality_category(accuracy):
    """Return quality category based on accuracy value"""
    if accuracy is None:
        return None
    
    try:
        accuracy = float(accuracy)
        if accuracy > 0.999:
            return "excellent" 
        elif accuracy > 0.97:
            return "good"      
        elif accuracy > 0.90:
            return "acceptable"
        elif accuracy > 0.80:
            return "mediocre" 
        else:
            return "poor"   
    except (ValueError, TypeError):
        return None

def should_rerun_model_by_quality(model_path, quality_threshold):
    """Check if model should be rerun based on quality threshold"""
    if not model_path.exists():
        return True  # Model doesn't exist, so run it
    
    # Load metadata
    metadata = load_model_metadata(model_path)
    if not metadata or 'metrics' not in metadata:
        return True  # No metrics, so run it
    
    # Get accuracy
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    # Get quality category
    quality = get_model_quality_category(accuracy)
    
    # Quality thresholds in descending order
    quality_levels = ["excellent", "good", "acceptable", "mediocre", "poor"]
    
    # If quality is None, rerun it
    if quality is None:
        return True
    
    # Find index of threshold and current quality
    threshold_index = quality_levels.index(quality_threshold)
    quality_index = quality_levels.index(quality)
    
    # If current quality is at threshold or worse, rerun it
    return quality_index >= threshold_index

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
    display_parts = [emoji] if emoji else []
    
    # Add architecture info if available
    if 'metrics' in metadata and 'architecture' in metadata['metrics']:
        arch = metadata['metrics']['architecture']
        display_parts.append(f"({arch})")
    
    # Add the base metrics display
    if base_display:
        display_parts.append(base_display)
    
    # Join all parts with spaces
    return " ".join(display_parts)

def format_architecture_comparison(comparison_data):
    """
    Format architecture comparison data for display
    
    Args:
        comparison_data: Dict mapping architecture names to accuracy values
        
    Returns:
        str: Formatted comparison string
    """
    if not comparison_data:
        return ""
        
    # Sort architectures by accuracy (descending)
    sorted_archs = sorted(comparison_data.items(), key=lambda x: x[1], reverse=True)
    
    # Format each architecture with its accuracy
    formatted_items = [f"{arch}: {acc:.4f}" for arch, acc in sorted_archs]
    
    # Join with commas and highlight the best one with an asterisk
    best_arch, _ = sorted_archs[0]
    formatted_items[0] = f"*{formatted_items[0]}*"  # Mark best with asterisks
    
    return f"Compared: {', '.join(formatted_items)}"

def list_models(models_path, available_models):
    """List all available models with their metadata"""
    print("\nAvailable Models:")
    print("================")
    
    # Table headers
    headers = ["Model Name", "Number", "Category", "Status", "CLI Parameter", "Metrics", "Architecture"]
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
        architecture_info = ""
        
        # Get metrics from model metadata if available
        if model_path.exists():
            metadata = load_model_metadata(model_path)
            # Use the function to get metrics with emoji
            metrics_display = format_metrics_with_emoji(metadata)
            
            # Extract architecture information if available
            if metadata and 'metrics' in metadata:
                if 'architecture' in metadata['metrics']:
                    architecture_info = metadata['metrics']['architecture']
                    
                # Add comparison info if multiple architectures were tried
                if 'architecture_comparison' in metadata['metrics']:
                    comp_info = format_architecture_comparison(metadata['metrics']['architecture_comparison'])
                    if comp_info:
                        architecture_info = f"{architecture_info}\n{comp_info}"
        
        # Add row to table data with CLI parameter
        table_data.append([display_name, model_number, category, status, cli_param, metrics_display, architecture_info])
    
    # Display the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\nTo train a specific model: python 01_run_image_tests.py --only MODEL-NAME")
    print("For more information, run: python 01_run_image_tests.py -h")

def get_dataset_info(data_path, model_name):
    """Get dataset size information to help select optimal model architecture"""
    try:
        # Count images in all subdirectories
        total_images = 0
        num_classes = 0
        
        # Look for directories that might contain class folders
        potential_data_dirs = [data_path]
        for subdir in ['train', 'valid', 'test']:
            if (data_path / subdir).exists() and (data_path / subdir).is_dir():
                potential_data_dirs.append(data_path / subdir)
        
        # Count classes and images
        class_counts = {}
        for data_dir in potential_data_dirs:
            for item in data_dir.iterdir():
                if item.is_dir():
                    # Count images in this class directory
                    img_count = len([f for f in item.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    if img_count > 0:
                        class_counts[item.name] = class_counts.get(item.name, 0) + img_count
                        total_images += img_count
        
        num_classes = len(class_counts)
        
        # Log dataset statistics
        print(f"Dataset for {model_name}: {total_images} images across {num_classes} classes")
        
        return total_images, num_classes
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return 1000, 2  # Default values

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
                    
                    # Prepare arguments for the function
                    func_args = {
                        'data_path': data_path,
                        'work_path': work_path,
                        'models_path': models_path,
                        'resume_training': resume_training,
                        'recalculate_lr': args.recalculate_lr,
                        'force_overwrite': args.force_overwrite
                    }
                    
                    # Add architecture auto-selection parameters if enabled
                    if args.auto_architecture:
                        func_args['try_multiple_architectures'] = True
                        func_args['max_architectures'] = args.max_architectures
                    
                    try:
                        # Try to call the function with the arguments that it accepts
                        import inspect
                        sig = inspect.signature(model_info['function'])
                        valid_args = {k: v for k, v in func_args.items() if k in sig.parameters}
                        model_info['function'](**valid_args)
                    except TypeError as e:
                        # Fall back to older function signature if needed
                        if "got an unexpected keyword argument" in str(e):
                            print(f"Warning: {model_info['name']} function doesn't support all parameters")
                            model_info['function'](data_path, work_path, models_path, 
                                                 resume_training, args.recalculate_lr, args.force_overwrite)
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
                    
                    # Prepare arguments for the function 
                    func_args = {
                        'data_path': data_path,
                        'work_path': work_path, 
                        'models_path': models_path,
                        'resume_training': resume_training,
                        'recalculate_lr': args.recalculate_lr,
                        'force_overwrite': args.force_overwrite
                    }
                    
                    # Add architecture auto-selection parameters if enabled
                    if args.auto_architecture:
                        func_args['try_multiple_architectures'] = True
                        func_args['max_architectures'] = args.max_architectures
                    
                    try:
                        # Try to call the function with the arguments that it accepts
                        import inspect
                        sig = inspect.signature(model_info['function'])
                        valid_args = {k: v for k, v in func_args.items() if k in sig.parameters}
                        model_info['function'](**valid_args)
                    except TypeError as e:
                        # Fall back to older function signature if needed
                        if "got an unexpected keyword argument" in str(e):
                            print(f"Warning: {model_name} function doesn't support all parameters")
                            model_info['function'](data_path, work_path, models_path, 
                                                 resume_training, args.recalculate_lr, args.force_overwrite)
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