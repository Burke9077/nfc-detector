"""
Window geometry and positioning utilities for PyQt5 applications.
Provides functions to manage window positions, ensure they're visible on screen,
and save/restore geometry from QSettings.
"""

from PyQt5.QtCore import QRect, QPoint, QSize, QSettings
from PyQt5.QtWidgets import QApplication

def is_position_on_screen(pos, size, screens=None):
    """
    Check if a window position is visible on any screen.
    Returns True if at least 50% of the window would be visible.
    
    Args:
        pos (QPoint): The position of the window
        size (QSize): The size of the window
        screens (list, optional): List of QScreens. If None, uses QApplication.screens()
    
    Returns:
        bool: True if the window would be mostly visible
    """
    if screens is None:
        # Get all screens from the application
        screens = QApplication.screens()
    
    # Create rect for the window
    window_rect = QRect(pos, size)
    
    # Check if the window is visible on any screen
    for screen in screens:
        # Get the geometry of the screen
        screen_geometry = screen.availableGeometry()
        
        # Calculate the intersection area
        intersection = screen_geometry.intersected(window_rect)
        window_area = window_rect.width() * window_rect.height()
        
        # If at least 50% of the window is visible, consider it valid
        if intersection.width() * intersection.height() >= window_area * 0.5:
            return True
    
    # No screen has enough visible area for this window
    return False

def get_centered_position(screens=None):
    """
    Get a centered position for the window on the primary screen.
    
    Args:
        screens (list, optional): List of QScreens. If None, uses QApplication.screens()
    
    Returns:
        QPoint: Center position on primary screen
    """
    if screens is None:
        screens = QApplication.screens()
    
    # Use primary screen
    primary_screen = screens[0]
    screen_geometry = primary_screen.availableGeometry()
    
    # Return the center position
    return screen_geometry.center()

def restore_window_geometry(window, app_name, window_name):
    """
    Restore window position and size from settings.
    
    Args:
        window (QWidget): The window to restore geometry for
        app_name (str): Application name for settings
        window_name (str): Window name for settings
    """
    settings = QSettings(app_name, window_name)
    
    # Set default values if settings don't exist yet
    if not settings.contains("geometry/size"):
        # No saved settings, use default size
        return
    
    # Get the saved values
    pos = settings.value("geometry/pos", QPoint(100, 100), type=QPoint)
    size = settings.value("geometry/size", QSize(800, 600), type=QSize)
    
    # Check if the position is still valid on the current screens
    if is_position_on_screen(pos, size):
        # Position is valid, restore it
        window.resize(size)
        window.move(pos)
    else:
        # Position is not valid, use a safe position
        screens = QApplication.screens()
        center = get_centered_position(screens)
        
        # Set window to a reasonable size
        window.resize(min(size.width(), 1024), min(size.height(), 768))
        
        # Move to center, adjusting for the window's size
        window.move(center.x() - window.width()//2, center.y() - window.height()//2)

def save_window_geometry(window, app_name, window_name):
    """
    Save window position and size to settings.
    
    Args:
        window (QWidget): The window to save geometry for
        app_name (str): Application name for settings
        window_name (str): Window name for settings
    """
    settings = QSettings(app_name, window_name)
    settings.setValue("geometry/pos", window.pos())
    settings.setValue("geometry/size", window.size())
"""
Window Utilities for NFC Detector

This module provides functions for managing window positioning and geometry,
ensuring windows are properly displayed on the user's screen.
"""
