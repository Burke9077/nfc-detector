"""
Model Utilities for NFC Detector

This module provides helper functions for model training and evaluation.
"""

import torch
import torch.cuda as cuda
import matplotlib.pyplot as plt

def check_gpu_memory():
    """
    Check if GPU is available and print memory info.
    No GUI elements - only console output.
    Warns if less than 8GB GPU memory is available.
    """
    if not cuda.is_available():
        print("CUDA not available, using CPU only")
        return None
    
    # Get device information
    device = torch.device("cuda")
    gpu_name = cuda.get_device_name(0)
    
    # Get memory information
    total_memory = round(cuda.get_device_properties(0).total_memory / 1e9, 2)  # Convert to GB
    allocated_memory = round(cuda.memory_allocated(0) / 1e9, 2)  # Convert to GB
    free_memory = round(total_memory - allocated_memory, 2)
    
    # Print GPU information to console
    print(f"\nGPU Information:")
    print(f"  Device: {gpu_name}")
    print(f"  Total Memory: {total_memory:.2f} GB")
    print(f"  Currently Allocated: {allocated_memory:.2f} GB")
    print(f"  Free Memory: {free_memory:.2f} GB")
    
    # Check if GPU has at least 8GB available
    if total_memory < 8.0:
        print(f"  WARNING: GPU has less than 8GB total memory ({total_memory:.2f} GB). Performance may be impacted.")
    elif free_memory < 8.0:
        print(f"  WARNING: Less than 8GB GPU memory available ({free_memory:.2f} GB). Performance may be impacted.")
    else:
        print(f"  GPU memory is sufficient for training.")
    
    return device
