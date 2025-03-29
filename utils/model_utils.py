"""
Model Utilities for NFC Detector

This module provides helper functions for model training and evaluation:
- GPU memory checking and visualization
- Training resource monitoring

These utilities support the model training process by providing insights
into available computational resources.
"""

import torch.cuda as cuda
import matplotlib.pyplot as plt

def check_gpu_memory():
    """Check if GPU is available and print memory info"""
    if not cuda.is_available():
        print("CUDA not available, using CPU only")
        return
    
    # Get device information
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    gpu_name = cuda.get_device_name(0) if cuda.is_available() else "CPU"
    
    # Get memory information
    if cuda.is_available():
        total_memory = round(cuda.get_device_properties(0).total_memory / 1e9, 2)  # Convert to GB
        allocated_memory = round(cuda.memory_allocated(0) / 1e9, 2)  # Convert to GB
        free_memory = round(total_memory - allocated_memory, 2)
        
        # Print GPU information to console
        print(f"\nGPU Information:")
        print(f"  Device: {gpu_name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Currently Allocated: {allocated_memory:.2f} GB")
        print(f"  Free Memory: {free_memory:.2f} GB")
        
        # Warn if memory is low
        if free_memory < 2.0:  # Less than 2GB available
            print(f"  WARNING: Low GPU memory available ({free_memory:.2f} GB). Performance may be affected.")
    else:
        print("Using CPU for computation (No GPU detected)")
    
    return device
