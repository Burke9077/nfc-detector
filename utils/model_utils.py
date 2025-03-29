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
    """Check and print available GPU memory"""
    if cuda.is_available():
        device = cuda.current_device()
        print(f"GPU: {cuda.get_device_name(device)}")
        
        # Print memory stats
        total_mem = cuda.get_device_properties(device).total_memory / 1e9  # Convert to GB
        reserved = cuda.memory_reserved(device) / 1e9
        allocated = cuda.memory_allocated(device) / 1e9
        free = total_mem - reserved
        
        print(f"Total GPU memory: {total_mem:.2f} GB")
        print(f"Reserved memory: {reserved:.2f} GB")
        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Free memory: {free:.2f} GB")
        
        # Plot memory usage
        labels = ['Total', 'Reserved', 'Allocated', 'Free']
        values = [total_mem, reserved, allocated, free]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
        plt.title('GPU Memory Usage (GB)')
        plt.ylabel('Memory (GB)')
        plt.show()
    else:
        print("No GPU available")
