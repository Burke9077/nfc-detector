"""
Test Utilities for NFC Detector

This module provides high-level utilities for standardizing model testing workflows:
- Standard test setup and execution
- Environment variable handling
- Path management for models, checkpoints and visualizations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastai.learner import Learner

from utils.directory_utils import find_latest_checkpoint, setup_temp_dir
from utils.dataset_utils import prepare_balanced_dataset
from utils.model_metadata_utils import save_model_metadata, load_model_metadata, is_model_better
from image_test_utils import train_and_save_model

# Import from 01_run_image_tests if available, or define functions locally
try:
    from utils.model_performance_utils import get_best_accuracy_from_metadata
except ImportError:
    # Fallback implementation if not imported
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

def track_model_performance(new_metrics, model_name, model_path):
    """
    Track the performance of a new model compared to an existing one.
    Updates the global model_performance_tracker if running from main script.
    
    Args:
        new_metrics: The metrics from the new model
        model_name: The name of the model
        model_path: Path to existing model (if any)
        
    Returns:
        True if new model is better or no old model exists
    """
    # Get module that contains model_performance_tracker
    try:
        import sys
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'model_performance_tracker'):
            performance_tracker = main_module.model_performance_tracker
            
            # Create a minimal metadata dict with just metrics
            new_model_metadata = {'metrics': new_metrics} if new_metrics else None
            
            # If existing model doesn't exist, always return True
            if not model_path.exists():
                # Store info about new model (no comparison available)
                new_accuracy = get_best_accuracy_from_metadata(new_model_metadata)
                performance_tracker[model_name] = {
                    'new_accuracy': new_accuracy,
                    'old_accuracy': None,
                    'change': None,
                    'percent_change': None,
                    'is_improvement': True  # New model is always an improvement if no previous model
                }
                return True
            
            # Load existing model metadata
            existing_metadata = load_model_metadata(model_path)
            
            # If existing metadata is invalid or missing metrics, replace it
            if not existing_metadata or 'metrics' not in existing_metadata:
                new_accuracy = get_best_accuracy_from_metadata(new_model_metadata)
                performance_tracker[model_name] = {
                    'new_accuracy': new_accuracy,
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
            performance_tracker[model_name] = {
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
    except (ImportError, AttributeError):
        # Fallback to simple comparison if not running from main script
        return is_model_better(new_metrics, load_model_metadata(model_path)['metrics'] if model_path.exists() else None)

def run_classification_test(
    test_name: str,
    model_name: str,
    model_number: str,
    data_path: Path,
    work_path: Path,
    models_path: Path,
    class_folders_dict: Dict[str, List[str]],
    train_params: Dict[str, Any],
    resume: bool = False,
    recalculate_lr: bool = False,
    force_overwrite: bool = False
) -> Learner:
    """
Run a standard classification test with consistent workflow across all tests.
        
    Args:
        test_name: Human-readable name for the test (for display)
        model_name: Model identifier for file naming
        model_number: Model number (e.g., "01") for file prefixes
        data_path: Root path for input dataset folders
        work_path: Working directory for temp files and checkpoints
        models_path: Directory where final models will be saved
        class_folders_dict: Dict mapping class names to lists of folder names
        train_params: Dict of parameters for train_and_save_model
        resume: Whether to resume from checkpoint
        recalculate_lr: Whether to recalculate learning rate
        force_overwrite: Whether to overwrite existing models regardless of performance
    
    Returns:
        The trained FastAI learner object
    """
    print(f"\n=== Running {test_name} ===")
    
    # Check for existing checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = find_latest_checkpoint(work_path, model_name)
        if checkpoint:
            print(f"Will resume training from checkpoint: {checkpoint}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Setup temp directory
    temp_dir = setup_temp_dir(work_path)
    
    # Convert folder names to full paths
    class_paths = {}
    for class_name, folders in class_folders_dict.items():
        class_paths[class_name] = [data_path / folder for folder in folders]
    
    # Get max_images_per_class from environment variable or default
    max_images_per_class = int(os.environ.get('MAX_IMAGES_PER_CLASS', 800))
    
    # Visualize balance if requested
    visualize = os.environ.get('VISUALIZE_BALANCE', '0') == '1'
    viz_path = work_path / f"{model_number}_{model_name}_balance.png" if visualize else None
    
    # Prepare dataset with dynamic balancing
    class_counts = prepare_balanced_dataset(
        class_paths,
        temp_dir,
        max_images_per_class=max_images_per_class,
        visualize=visualize,
        viz_path=viz_path,
        model_name=f"{model_number}_{model_name}"
    )
    
    # Define model path
    model_path = models_path / f"{model_number}_{model_name}_model.pkl"
    
    # Train and save model (now returns metrics dictionary as well as the learner)
    learn, metrics = train_and_save_model(
        temp_dir,
        model_path,
        work_path,
        **train_params,
        resume_from_checkpoint=checkpoint,
        recalculate_lr=recalculate_lr,
        save_model=False  # We'll handle model saving ourselves
    )
    
    # Track performance and decide whether to save (replaces previous comparison logic)
    should_save = force_overwrite or track_model_performance(metrics, model_name, model_path)
    
    # Save the model if it's better or if forced
    if should_save:
        try:
            print(f"Saving model to {model_path}")
            learn.export(model_path)
            save_model_metadata(model_path, metrics)
        except RuntimeError as e:
            if "selected index k out of range" in str(e):
                print("Warning: Couldn't plot top losses - validation set too small. Saving model anyway.")
                # Make sure we still export the model even if visualization failed
                if not os.path.exists(model_path):
                    learn.export(model_path)
                    save_model_metadata(model_path, metrics)
            else:
                # Re-raise any other RuntimeErrors
                raise
    
    return learn
