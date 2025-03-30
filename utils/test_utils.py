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
    
    # If the model exists, check if new model is better
    should_save = True
    if model_path.exists() and not force_overwrite:
        # Load old model's metadata
        old_metadata = load_model_metadata(model_path)
        if old_metadata and 'metrics' in old_metadata:
            old_metrics = old_metadata['metrics']
            
            # Compare metrics
            if not is_model_better(metrics, old_metrics):
                print(f"New model is not better than existing model. Keeping the old model.")
                print(f"New metrics: {metrics}")
                print(f"Old metrics: {old_metrics}")
                should_save = False
    
    # Save the model if it's better or if forced
    if should_save or force_overwrite:
        print(f"Saving model to {model_path}")
        learn.export(model_path)
        save_model_metadata(model_path, metrics)
    
    return learn
