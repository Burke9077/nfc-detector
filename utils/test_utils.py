"""
Test Utilities for NFC Detector

This module provides high-level utilities for standardizing model testing workflows:
- Standard test setup and execution
- Environment variable handling
- Path management for models, checkpoints and visualizations
"""

import os
from pathlib import Path
from utils.directory_utils import find_latest_checkpoint, setup_temp_dir
from utils.dataset_utils import prepare_balanced_dataset
from image_test_utils import train_and_save_model

def run_classification_test(
    test_name,
    model_name,
    model_number,
    data_path,
    work_path,
    models_path,
    class_folders_dict,
    train_params,
    resume=False,
    recalculate_lr=False
):
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
    
    # Train and save model
    model_path = models_path / f"{model_number}_{model_name}_model.pkl"
    learn = train_and_save_model(
        temp_dir,
        model_path,
        work_path,
        **train_params,
        resume_from_checkpoint=checkpoint,
        recalculate_lr=recalculate_lr
    )
    
    return learn
