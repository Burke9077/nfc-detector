"""
NFC Detector Utilities Package

This package contains utility modules for the NFC Detector project:
- directory_utils: Directory management and file operations
- dataset_utils: Dataset preparation and organization
- model_utils: Model training support and GPU monitoring

Import commonly used functions directly from the utils package.
"""

# Import common functions for easier access
from .directory_utils import (
    verify_directories, clean_work_dir, find_latest_checkpoint,
    is_test_completed, setup_temp_dir
)

from .dataset_utils import copy_images_to_class, balanced_copy_images
from .model_utils import check_gpu_memory

# Define what's available when importing with *
__all__ = [
    # From directory_utils
    'verify_directories', 'clean_work_dir', 'find_latest_checkpoint',
    'is_test_completed', 'setup_temp_dir',
    
    # From dataset_utils
    'copy_images_to_class', 'balanced_copy_images',
    
    # From model_utils
    'check_gpu_memory',
]
