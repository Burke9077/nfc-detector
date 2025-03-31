"""
Camera utility functions for detecting and retrieving information about attached camera devices.
"""

import cv2
import subprocess
import platform
import re
import sys
import time

def get_device_info(device_id):
    """
    Try to get additional device information using system-specific commands.
    This is platform-dependent and may not work on all systems.
    
    Args:
        device_id (int): The device ID of the camera
        
    Returns:
        dict: Dictionary with camera information including name and manufacturer
    """
    device_info = {"name": f"Camera {device_id}", "manufacturer": "Unknown"}
    
    system = platform.system()
    
    try:
        if system == "Windows":
            # On Windows, use PowerShell to query device information
            cmd = ["powershell", "-Command", 
                  "Get-PnpDevice -Class 'Camera' | Format-List FriendlyName, Manufacturer"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout:
                # Parse output to find camera names and manufacturers
                devices = result.stdout.split("\r\n\r\n")
                # Try to match to the device_id (this is imperfect as we can't directly link them)
                if device_id < len(devices) and "FriendlyName" in devices[device_id]:
                    matches = re.search(r"FriendlyName\s*:\s*(.+)", devices[device_id])
                    if matches:
                        device_info["name"] = matches.group(1).strip()
                    
                    matches = re.search(r"Manufacturer\s*:\s*(.+)", devices[device_id])
                    if matches:
                        device_info["manufacturer"] = matches.group(1).strip()
        
        elif system == "Linux":
            # On Linux, try using v4l2-ctl
            cmd = ["v4l2-ctl", "--list-devices"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout:
                devices = result.stdout.split("\n\n")
                if device_id < len(devices):
                    device_info["name"] = devices[device_id].split("\n")[0].strip()
        
        elif system == "Darwin":  # macOS
            # On macOS, limited options without additional libraries
            pass
            
    except (subprocess.SubprocessError, IndexError, FileNotFoundError) as e:
        # If anything goes wrong, we'll just use the default info
        pass
        
    return device_info

def list_video_devices():
    """
    List all available video devices with minimal checking.
    
    Returns:
        dict: Dictionary of device IDs and their basic information including
              resolution, fps, name, and manufacturer
    """
    devices = {}
    print("Scanning for video devices...")
    
    # Try a reasonable range of device IDs
    for i in range(10):  # Usually video devices are 0-9
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get basic device information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Simplified device info - just use a generic name
                device_name = f"Camera {i}"
                
                # Store basic info without extra checks
                devices[i] = {
                    'id': i,
                    'resolution': (width, height),
                    'fps': fps,
                    'name': device_name,
                    'manufacturer': "Unknown"
                }
                
                print(f"  Found camera: Device {i} ({width}x{height})")
                
                # Release the device
                cap.release()
        except Exception as e:
            pass
    
    print(f"Found {len(devices)} camera device(s)")
    return devices
