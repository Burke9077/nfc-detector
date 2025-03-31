# UI Utilities

This directory contains utility modules for user interface-related functionality in the NFC Detector application.

## Available Modules

### window_utils.py (./utils/ui/window_utils.py)
This module provides utilities for managing window positioning and geometry in PyQt5 applications:
- **Position validation**: Use `is_position_on_screen()` to check if a window position would be visible on screen
- **Window centering**: Use `get_centered_position()` to find the center position on the primary screen
- **Geometry persistence**: Use `restore_window_geometry()` and `save_window_geometry()` to save and restore window positions
- **Multi-monitor support**: All functions handle setups with multiple monitors correctly

Include this file when building PyQt5-based UIs that need to remember their positions, ensure windows appear on-screen, or handle multiple monitor configurations properly.

### gpu_utils.py (./utils/ui/gpu_utils.py)
This module provides utilities for checking and displaying GPU status information:
- **GPU availability**: Use `check_gpu_status()` to verify if a CUDA-capable GPU is available
- **Detailed information**: Get comprehensive details about available GPUs including CUDA capability and memory
- **Output capture**: Use `check_gpu_status_with_capture()` to get both status and formatted text output

Include this file when your application requires GPU acceleration and needs to provide feedback to the user about GPU availability and specifications.

### camera_utils.py (./utils/ui/camera_utils.py)
This module provides utilities for detecting and retrieving information about attached camera devices:
- **Device listing**: Use `list_video_devices()` to find all connected camera devices and their properties
- **Device information**: Use `get_device_info()` to retrieve additional platform-specific details about a camera
- **Cross-platform support**: Works on Windows, Linux, and provides fallbacks for unsupported platforms

Include this file when your application needs to enumerate available cameras or gather information about them before opening a video stream.
