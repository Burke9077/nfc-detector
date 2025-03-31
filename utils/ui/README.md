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
