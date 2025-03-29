"""
Directory Utilities for NFC Detector

This module provides utility functions for directory management, including:
- Directory creation and verification
- Working directory cleanup
- Checkpoint finding and management
- Test completion checking

These utilities support the image classification pipeline by ensuring
proper directory structure and managing temporary files.
"""

from pathlib import Path
import shutil

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
        
    # Look for the best_model checkpoint (no longer using stages)
    checkpoint = checkpoint_dir / 'best_model'
    if checkpoint.exists():
        print(f"Found checkpoint: {checkpoint}")
        return checkpoint
    
    # For backward compatibility, also check for stage-based checkpoints
    checkpoints = list(checkpoint_dir.glob(f'best_model_stage*'))
    if checkpoints:
        # Sort by modification time, newest first
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = checkpoints[0]
        print(f"Found legacy stage checkpoint: {latest}")
        return latest
        
    return None

def is_test_completed(models_path, model_name):
    """Check if a test has already successfully completed by looking for the output model file"""
    model_file = models_path / f"{model_name}.pkl"
    return model_file.exists()

def setup_temp_dir(work_path):
    """Create and return a temporary directory for image processing"""
    temp_dir = work_path / "temp_test_dir"
    if temp_dir.exists():
        # Clean up previous test files
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir
