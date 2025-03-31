"""
GPU utility functions for checking and displaying GPU status information.
"""

import torch
import io
from contextlib import redirect_stdout

def check_gpu_status():
    """
    Check and display GPU information. Returns True if CUDA is available.
    Provides detailed output about CUDA availability and GPU specifications.
    """
    print("Checking GPU status...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. A CUDA-enabled GPU is required for this application.")
        print("   Please ensure you have:")
        print("   1. A compatible NVIDIA GPU")
        print("   2. Proper NVIDIA drivers installed")
        print("   3. CUDA toolkit installed and configured")
        return False
    
    # CUDA is available, show details
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA is available. Found {device_count} GPU(s).")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"  GPU #{i}: {device_name} (CUDA Capability {device_capability[0]}.{device_capability[1]})")
        
        # Get memory info
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
        reserved = torch.cuda.memory_reserved(i) / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        free = total_mem - reserved
        
        print(f"     Memory: {total_mem:.2f} GB total, {free:.2f} GB free")
    
    # Set the current device to 0
    torch.cuda.set_device(0)
    print(f"✓ Using GPU #{0}: {torch.cuda.get_device_name(0)}")
    
    return True

def check_gpu_status_with_capture():
    """
    Check GPU status and capture the output as a string.
    Returns a tuple of (has_cuda, output_text).
    """
    # Capture the output of the check_gpu_status function
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        has_cuda = check_gpu_status()
    
    # Return both the CUDA status and the captured output
    return has_cuda, captured_output.getvalue()
