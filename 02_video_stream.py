# Set up the compatibility layer before any other imports
# This ensures fastcore.transform is available when FastAI attempts to unpickle models
import sys
import types
import pip

# Create and install the fastcore.transform compatibility layer
if "fastcore.transform" not in sys.modules or "fastcore.dispatch" not in sys.modules:
    try:
        # First ensure fasttransform is installed
        try:
            import fasttransform
        except ImportError:
            print("Installing fasttransform package...")
            pip.main(['install', 'git+https://github.com/AnswerDotAI/fasttransform.git'])
            print("✓ fasttransform installed successfully.")
            import fasttransform
        
        # Create compatibility module for fastcore.transform and populate it
        print("Setting up fastcore.transform compatibility layer...")
        fastcore_transform = types.ModuleType("fastcore.transform")
        
        # Create minimal class definitions for essential Transform classes
        # Only try to import what actually exists, with fallbacks for missing classes
        
        # Start with Pipeline which is the main one needed for model loading
        if hasattr(fasttransform, 'Pipeline'):
            from fasttransform import Pipeline
            fastcore_transform.Pipeline = Pipeline
        else:
            # Create a minimal Pipeline implementation if it doesn't exist
            class Pipeline:
                def __init__(self, *args, **kwargs): pass
            fastcore_transform.Pipeline = Pipeline
            print("  Created Pipeline placeholder class")
        
        # Try to import other classes with fallbacks
        try:
            from fasttransform import Transform
            fastcore_transform.Transform = Transform
        except (ImportError, AttributeError):
            # Create minimal Transform class if needed
            class Transform:
                def __init__(self, *args, **kwargs): pass
            fastcore_transform.Transform = Transform
            print("  Created Transform placeholder class")
        
        # Add other required classes with fallbacks
        class_names = ['DisplayedTransform', 'ItemTransform', 'TupleTransform']
        for class_name in class_names:
            try:
                cls = getattr(fasttransform, class_name)
                setattr(fastcore_transform, class_name, cls)
            except (ImportError, AttributeError):
                # Create a minimal placeholder class
                placeholder = type(class_name, (), {"__init__": lambda self, *args, **kwargs: None})
                setattr(fastcore_transform, class_name, placeholder)
                print(f"  Created {class_name} placeholder class")
        
        # For functions, try to import them or create simple fallbacks
        function_names = ['drop_none', 'retain_type', 'get_func']
        for func_name in function_names:
            try:
                func = getattr(fasttransform, func_name)
                setattr(fastcore_transform, func_name, func)
            except (ImportError, AttributeError):
                # Create a simple pass-through function
                setattr(fastcore_transform, func_name, lambda *args, **kwargs: args[0] if args else None)
                print(f"  Created {func_name} placeholder function")
        
        # Register the transform module with Python
        sys.modules["fastcore.transform"] = fastcore_transform
        
        # Create and set up fastcore.dispatch compatibility module
        print("Setting up fastcore.dispatch compatibility layer...")
        fastcore_dispatch = types.ModuleType("fastcore.dispatch")
        
        # Import TypeDispatch or create a fallback
        try:
            from plum import Function
            # Use Function from plum as a replacement for TypeDispatch
            class TypeDispatch:
                def __init__(self, *funcs, **kwargs):
                    self.func = Function(funcs[0] if funcs else lambda: None)
                    for f in funcs[1:]:
                        self.func.register(f)
                
                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)
                
                def add(self, f):
                    self.func.register(f)
                    return self
                
                # Add other necessary methods for compatibility
                def returns(self, x):
                    # Simple implementation to avoid complex logic
                    return None
                
                def __getitem__(self, key):
                    # Simple implementation to avoid complex logic
                    return lambda x
                
                def __repr__(self):
                    return str(self.func.methods)
            
            fastcore_dispatch.TypeDispatch = TypeDispatch
            print("  Created TypeDispatch compatibility class using plum.Function")
            
            # Add typedispatch decorator
            def typedispatch(f=None):
                # Simple typedispatch decorator using plum's dispatch
                from plum import dispatch
                if f is None: return dispatch
                return dispatch(f)
                
            fastcore_dispatch.typedispatch = typedispatch
            print("  Created typedispatch compatibility function")
            
        except (ImportError, AttributeError) as e:
            print(f"  Error setting up TypeDispatch: {str(e)}")
            # Create minimal TypeDispatch class as fallback
            class TypeDispatch:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, *args, **kwargs): return args[0] if args else None
                def add(self, f): return self
                def returns(self, x): return None
                def __getitem__(self, key): return lambda x: x
                def __repr__(self): return "TypeDispatch(placeholder)"
                
            fastcore_dispatch.TypeDispatch = TypeDispatch
            fastcore_dispatch.typedispatch = lambda f=None: (lambda g: g) if f is None else f
            print("  Created TypeDispatch fallback placeholder class")
        
        # Register the dispatch module with Python
        sys.modules["fastcore.dispatch"] = fastcore_dispatch
            
        # Also create fastcore.basics if needed
        if "fastcore.basics" not in sys.modules:
            fastcore_basics = types.ModuleType("fastcore.basics")
            sys.modules["fastcore.basics"] = fastcore_basics
            
        print("✓ Compatibility layer installed successfully")
    except Exception as e:
        print(f"ERROR setting up compatibility layer: {str(e)}")
        print("Models may fail to load.")

# Now continue with normal imports
import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import subprocess
import platform
import re
import torch
import torch.cuda as cuda
import io
from contextlib import redirect_stdout
import datetime
import shutil

# Import our window utility module
from utils.ui.window_utils import (
    is_position_on_screen, 
    get_centered_position, 
    restore_window_geometry,
    save_window_geometry
)

# Add FastAI imports for model loading and inference
from fastai.vision.all import load_learner, PILImage
import pandas as pd

# Add PyQt5 imports - exit if not installed
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                              QVBoxLayout, QHBoxLayout, QLabel, QStatusBar, 
                              QDialog, QTextEdit, QComboBox, QDialogButtonBox,
                              QTabWidget, QGroupBox, QRadioButton, QScrollArea,
                              QMessageBox, QButtonGroup, QTableWidget, QTableWidgetItem,
                              QHeaderView, QCheckBox)
    from PyQt5.QtCore import Qt, QTimer, QSize, QSettings, QPoint, QRect
    from PyQt5.QtGui import QImage, QPixmap, QColor, QScreen
except ImportError:
    print("ERROR: PyQt5 is required but not installed.")
    print("Please install it with: pip install PyQt5")
    print("Then run this script again.")
    sys.exit(1)

def get_device_info(device_id):
    """
    Try to get additional device information using system-specific commands.
    This is platform-dependent and may not work on all systems.
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
    Returns a dictionary of device IDs and their basic information.
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

class VideoPreviewWidget(QWidget):
    """Widget to display a live video preview of a camera."""
    
    def __init__(self, device_id, parent=None):
        super().__init__(parent)
        self.device_id = device_id
        self.cap = None
        self.running = False
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Video display
        self.video_label = QLabel("Opening camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        layout.addWidget(self.video_label)
        
        # Start video capture
        self.cap = cv2.VideoCapture(self.device_id)
        if self.cap.isOpened():
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(100)  # Update every 100ms (slower for previews)
            self.running = True
        else:
            self.video_label.setText(f"Failed to open camera {device_id}")
    
    def update_frame(self):
        """Update the video frame display."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                
                # Convert to QImage and then QPixmap
                qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Scale pixmap to fit the label
                pixmap = pixmap.scaled(self.video_label.size(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Update the video label
                self.video_label.setPixmap(pixmap)
    
    def close_video(self):
        """Stop the video and release resources."""
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

class MicroscopeUI(QMainWindow):
    """
    PyQt5-based UI for the microscope video stream.
    Provides a robust interface with proper video display and button controls.
    """
    def __init__(self, device_id, target_resolution=(1280, 720)):
        super().__init__()
        
        # Store parameters
        self.device_id = device_id
        self.target_resolution = target_resolution
        self.frame_count = 0
        self.last_frame = None
        
        # Load models for inference
        self.models = find_and_load_models()
        
        # Set up the UI
        self.setWindowTitle(f"USB Microscope (Device {device_id})")
        
        # Default minimum size
        self.setMinimumSize(1024, 768)  # Larger window for model results
        
        # Restore saved window position and size
        restore_window_geometry(self, "NFC-Detector", "MicroscopeUI")
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create splitter layout: video on left, results on right
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 3)  # Give more space to the camera and results area
        
        # Left side: Video display
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        top_layout.addWidget(video_widget, 2)  # Video takes 2/3 of the width
        
        # Right side: Model inference results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Results label
        results_label = QLabel("Model Predictions")
        results_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_label)
        
        # Results table
        self.results_table = QTableWidget(0, 3)  # Rows will be added dynamically, 3 columns
        self.results_table.setHorizontalHeaderLabels(["Model", "Prediction", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        top_layout.addWidget(results_widget, 1)  # Results takes 1/3 of the width
        
        # Button panel
        button_layout = QHBoxLayout()
        
        # Capture button
        self.capture_btn = QPushButton("Capture Frame and Analyze (C)")
        self.capture_btn.setMinimumHeight(50)
        self.capture_btn.clicked.connect(self.capture_frame)
        button_layout.addWidget(self.capture_btn)
        
        # Quit button
        self.quit_btn = QPushButton("Quit (Q)")
        self.quit_btn.setMinimumHeight(50)
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.quit_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status bar for feedback
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Set up video capture
        self.setup_camera()
        
        # Set up keyboard shortcuts
        self.capture_btn.setShortcut("C")
        self.quit_btn.setShortcut("Q")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save window position and size
        save_window_geometry(self, "NFC-Detector", "MicroscopeUI")
        
        # Stop the timer and release the camera
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        event.accept()
        
    def setup_camera(self):
        """Initialize the camera and video timer"""
        self.cap = cv2.VideoCapture(self.device_id)
        
        # No need to try setting resolution - just use whatever the camera provides
        if not self.cap.isOpened():
            self.statusBar.showMessage("Error: Failed to open camera")
            return
        
        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.statusBar.showMessage(f"Streaming at {width}x{height}")
        
        # Create timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (approx 30 FPS)
    
    def update_frame(self):
        """Update the video frame display"""
        ret, frame = self.cap.read()
        if ret:
            # Store the frame for capture
            self.last_frame = frame
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            
            # Convert to QImage and then QPixmap
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale pixmap if needed while maintaining aspect ratio
            pixmap = pixmap.scaled(self.video_label.size(), 
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Update the video label
            self.video_label.setPixmap(pixmap)
    
    def capture_frame(self):
        """Capture the current frame, run inference, and prompt for labeling"""
        if self.last_frame is not None:
            # Store a copy of the captured frame to avoid modification during processing
            captured_frame = self.last_frame.copy()
            
            # Update status
            self.statusBar.showMessage("Frame captured - Running analysis...")
            
            # Run inference if models are loaded
            if self.models:
                # Run inference on the captured frame
                results = run_inference(captured_frame, self.models)
                
                # Update results table
                self.update_results_table(results)
                
                # Update status
                self.statusBar.showMessage("Analysis complete")
            else:
                self.statusBar.showMessage("No models loaded for analysis")
            
            # Show dialog for image labeling - using the same captured frame
            dialog = ImageLabelingDialog(captured_frame, self)
            if dialog.exec_() == QDialog.Accepted:
                labels = dialog.get_image_labels()
                if labels:
                    # Save image to appropriate folders
                    saved_paths = save_labeled_image(captured_frame, labels)
                    paths_str = ", ".join(str(p) for p in saved_paths)
                    self.statusBar.showMessage(f"Image saved with labels: {', '.join(labels)}")
                    print(f"Image saved to: {paths_str}")
                else:
                    self.statusBar.showMessage("No labels selected, image not categorized")
            else:
                self.statusBar.showMessage("Image labeling canceled")
    
    def update_results_table(self, results):
        """Update the results table with model predictions"""
        # Clear existing results
        self.results_table.setRowCount(0)
        
        # Add new results
        for i, (model_name, result) in enumerate(results.items()):
            # Add a row for each model
            self.results_table.insertRow(i)
            
            # Add model name
            model_item = QTableWidgetItem(model_name)
            self.results_table.setItem(i, 0, model_item)
            
            # Add prediction
            pred_item = QTableWidgetItem(result['prediction'])
            self.results_table.setItem(i, 1, pred_item)
            
            # Add confidence with color based on value and 2 decimal places
            confidence = result['confidence']
            conf_item = QTableWidgetItem(f"{confidence:.2f}%")
            
            # Color code by confidence: red (<70%), yellow (70-85%), green (>85%)
            if confidence < 70:
                conf_item.setBackground(QColor(255, 200, 200))  # Light red
            elif confidence < 85:
                conf_item.setBackground(QColor(255, 255, 200))  # Light yellow
            else:
                conf_item.setBackground(QColor(200, 255, 200))  # Light green
                
            self.results_table.setItem(i, 2, conf_item)
            
            # Add tooltips with all class probabilities (now with 2 decimal places)
            all_probs_text = "\n".join(f"{c}: {p:.2f}%" for c, p in result['all_probs'])
            model_item.setToolTip(all_probs_text)
            pred_item.setToolTip(all_probs_text)
            conf_item.setToolTip(all_probs_text)
            
            # Print the complete probabilities to the console for debugging
            print(f"\nModel: {model_name}")
            print("All class probabilities:")
            for c, p in result['all_probs']:
                print(f"  {c}: {p:.2f}%")

class SetupDialog(QDialog):
    """Dialog to show GPU status and allow camera selection."""
    
    def __init__(self, devices=None, parent=None):
        super().__init__(parent)
        self.devices = devices or {}
        self.selected_device = None
        self.preview_widgets = {}
        
        self.setWindowTitle("NFC Detector Setup")
        self.setMinimumSize(800, 600)
        
        # Restore saved window position and size (before creating layout)
        restore_window_geometry(self, "NFC-Detector", "SetupDialog")
        
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # GPU status tab
        self.gpu_tab = QWidget()
        self.tabs.addTab(self.gpu_tab, "GPU Status")
        
        # Camera selection tab
        self.camera_tab = QWidget()
        self.tabs.addTab(self.camera_tab, "Camera Selection")
        
        # Setup GPU tab
        self.setup_gpu_tab()
        
        # Setup Camera tab
        self.setup_camera_tab()
        
        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        # Set up the GPU status check first
        self.check_gpu_status()
    
    def accept(self):
        """Handle OK button."""
        save_window_geometry(self, "NFC-Detector", "SetupDialog")
        self.cleanup()
        super().accept()
    
    def reject(self):
        """Handle Cancel button."""
        save_window_geometry(self, "NFC-Detector", "SetupDialog")
        self.cleanup()
        super().reject()
        
    def setup_gpu_tab(self):
        """Set up the GPU status tab."""
        layout = QVBoxLayout(self.gpu_tab)
        
        # Text area for GPU info
        self.gpu_text = QTextEdit()
        self.gpu_text.setReadOnly(True)
        layout.addWidget(self.gpu_text)
    
    def setup_camera_tab(self):
        """Set up the camera selection tab."""
        layout = QVBoxLayout(self.camera_tab)
        
        # Info text
        info_label = QLabel("Select a camera to use:")
        layout.addWidget(info_label)
        
        if not self.devices:
            layout.addWidget(QLabel("No cameras detected"))
            return
        
        # Create a scroll area for camera options
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Camera option group
        self.camera_group = QGroupBox("Available Cameras")
        camera_layout = QVBoxLayout()
        self.camera_group.setLayout(camera_layout)
        
        # Create a button group for the radio buttons to ensure mutual exclusivity
        self.radio_group = QButtonGroup(self)
        
        # Add radio buttons for each camera
        self.camera_radios = {}
        for i, (dev_id, info) in enumerate(sorted(self.devices.items())):
            # Camera info with preview
            camera_widget = QWidget()
            cam_layout = QHBoxLayout(camera_widget)
            
            # Radio button for selection
            radio = QRadioButton(f"Device {dev_id}: {info['name']}")
            self.camera_radios[dev_id] = radio
            self.radio_group.addButton(radio, dev_id)  # Add to button group with ID
            cam_layout.addWidget(radio, 1)
            
            # Preview button
            preview_btn = QPushButton("Preview")
            preview_btn.setProperty("device_id", dev_id)
            preview_btn.clicked.connect(self.toggle_preview)
            cam_layout.addWidget(preview_btn)
            
            camera_layout.addWidget(camera_widget)
            
            # Container for the preview (initially empty)
            preview_container = QWidget()
            preview_layout = QVBoxLayout(preview_container)
            self.preview_widgets[dev_id] = {"container": preview_container, "widget": None}
            camera_layout.addWidget(preview_container)
        
        # Select first camera by default
        if self.camera_radios:
            first_id = list(self.camera_radios.keys())[0]
            self.camera_radios[first_id].setChecked(True)
            self.selected_device = first_id
        
        # Add camera group to scroll area
        scroll_layout.addWidget(self.camera_group)
        
        # Note about camera device ordering (replacing the swap functionality)
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_label = QLabel("Note: Camera device numbers may not match their physical order.")
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        scroll_layout.addWidget(info_widget)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
    
    def toggle_preview(self):
        """Toggle camera preview when button is clicked.""" 
        # Get the device ID from the sender button
        button = self.sender()
        dev_id = button.property("device_id")
        
        # Get the container and current widget
        container = self.preview_widgets[dev_id]["container"]
        widget = self.preview_widgets[dev_id]["widget"]
        
        # If preview is active, close it
        if widget and widget.running:
            button.setText("Preview")
            widget.close_video()
            widget.deleteLater()
            self.preview_widgets[dev_id]["widget"] = None
        else:
            # Start preview
            button.setText("Stop Preview")
            preview = VideoPreviewWidget(dev_id)
            container_layout = container.layout()
            
            # Clear any existing widgets in the container
            while container_layout.count():
                item = container_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Add the new preview widget
            container_layout.addWidget(preview)
            self.preview_widgets[dev_id]["widget"] = preview
            
            # Select this camera
            self.camera_radios[dev_id].setChecked(True)
            self.selected_device = dev_id
    
    def check_gpu_status(self):
        """Check GPU status and display in the text area.""" 
        # Capture the output of the check_gpu_status function
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            has_cuda = check_gpu_status_internal()
        
        # Set the text in the GPU info area
        self.gpu_text.setText(captured_output.getvalue())
        
        # Enable/disable the OK button based on CUDA availability
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(has_cuda)
        
        # If no CUDA, show a warning
        if not has_cuda:
            self.gpu_text.append("\n\nWARNING: No CUDA-capable GPU detected. This application requires a GPU.")
            self.gpu_text.append("You can override this requirement by using the --skip-gpu-check flag.")
    
    def get_selected_device(self):
        """Return the selected camera device ID.""" 
        selected_id = self.radio_group.checkedId()
        if selected_id != -1:  # -1 means no button selected
            return selected_id
        
        # Fallback to old method if button group fails
        for dev_id, radio in self.camera_radios.items():
            if radio.isChecked():
                return dev_id
                
        return self.selected_device

    def cleanup(self):
        """Clean up all video previews when dialog closes.""" 
        for dev_id, preview_data in self.preview_widgets.items():
            if preview_data["widget"] and preview_data["widget"].running:
                preview_data["widget"].close_video()

def check_gpu_status_internal():
    """Check and display GPU information. Returns True if CUDA is available.""" 
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

def display_video_stream(device_id, target_resolution=(1280, 720)):
    """
    Display video stream using PyQt5 UI.
    """
    # Initialize Qt organization and application names for settings
    QApplication.setOrganizationName("NFC-Detector")
    QApplication.setApplicationName("MicroscopeUI")
    
    # Create and show the UI
    window = MicroscopeUI(device_id, target_resolution)
    window.show()
    
    # Bring window to front and give it focus
    window.raise_()
    window.activateWindow()
    
    # Return the window instance so we can keep a reference to it
    return window

def find_and_load_models(models_dir="nfc_models"):
    """Find and load all model files in the models directory with enhanced compatibility handling"""
    models_dir = Path(models_dir)
    models = {}
    
    # Check if directory exists
    if not models_dir.exists():
        print(f"Warning: Models directory {models_dir} not found")
        return models
        
    # Find all .pkl files in models directory
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        print(f"Warning: No model files found in {models_dir}")
        return models
        
    print(f"Found {len(model_files)} model file(s):")
    
    # Load each model
    for model_file in model_files:
        model_name = model_file.stem  # Get filename without extension
        print(f"  Loading {model_name}...")
        try:
            # Use enhanced safe model loading with additional diagnostic info
            model = safe_load_model(model_file)
            
            # Print model info
            print(f"    ✓ Model loaded successfully: {len(model.dls.vocab)} classes - {model.dls.vocab}")
            
            # Store the model
            models[model_name] = model
        except Exception as e:
            print(f"    ✗ Error loading model {model_name}: {str(e)}")
            
            # Additional diagnostic info for debugging
            if "fastcore.transform" in str(e):
                print(f"    Debug: 'fastcore.transform' module present: {'fastcore.transform' in sys.modules}")
                if "fastcore.transform" in sys.modules:
                    print(f"    Debug: Contains Pipeline: {hasattr(sys.modules['fastcore.transform'], 'Pipeline')}")
                    pipeline_class = getattr(sys.modules['fastcore.transform'], 'Pipeline', None)
                    if pipeline_class:
                        print(f"    Debug: Pipeline class source: {pipeline_class.__module__}")
            
            # Try to provide more helpful information
            import traceback
            traceback.print_exc()
    
    return models

def safe_load_model(model_path):
    """
    Enhanced model loading function with better error diagnostics.
    """
    try:
        # First attempt: direct loading
        print(f"    Attempting to load model directly...")
        return load_learner(model_path)
    except Exception as e:
        print(f"    Direct load failed: {str(e)}")
        
        # If we're dealing with the fastcore.transform error, try to recover
        if "Pipeline" in str(e) and "fastcore.transform" in str(e):
            print(f"    Trying alternative loading approach...")
            try:
                # Get a fresh import context
                import importlib
                if "fastcore.transform" in sys.modules:
                    del sys.modules["fastcore.transform"]
                
                # Create the module again
                fastcore_transform = types.ModuleType("fastcore.transform")
                
                # Import individual components from fasttransform
                from fasttransform import Pipeline
                fastcore_transform.Pipeline = Pipeline
                
                # Register with sys.modules
                sys.modules["fastcore.transform"] = fastcore_transform
                
                # Try loading now that we have the module in place
                return load_learner(model_path)
            except Exception as recovery_error:
                print(f"    Alternative approach failed: {str(recovery_error)}")
                raise recovery_error
        else:
            # Not the fastcore.transform error, just raise
            raise e

def run_inference(image, models):
    """
    Run inference on an image using all loaded models
    Returns a dictionary of results by model
    """
    results = {}
    
    # Convert OpenCV image (BGR) to PIL Image (RGB) for FastAI
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.create(rgb_image)
    
    # Run inference with each model
    for model_name, model in models.items():
        try:
            # Get prediction - FastAI returns (prediction, prediction_index, probabilities)
            pred, pred_idx, probs = model.predict(pil_img)
            
            # Print raw probabilities for debugging
            print(f"\nModel: {model_name}")
            print(f"Raw prediction probabilities: {probs}")
            print(f"Selected class: {pred} (index {pred_idx})")
            
            # Get class names from the model's vocabulary
            class_names = model.dls.vocab
            
            # Create a list of (class_name, probability) tuples sorted by probability
            class_probs = []
            for i in range(len(class_names)):
                # Convert tensor element to Python float before multiplying
                prob_value = float(probs[i].item()) * 100.0
                class_probs.append((str(class_names[i]), prob_value))
            
            # Sort by probability, highest first
            class_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Store results
            confidence = float(probs[pred_idx].item()) * 100.0
            print(f"Confidence for {pred}: {confidence:.2f}%")
            
            results[model_name] = {
                'prediction': str(pred),
                'confidence': confidence,
                'all_probs': class_probs
            }
            
        except Exception as e:
            print(f"Error running inference with model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'prediction': "ERROR",
                'confidence': 0.0,
                'all_probs': [("ERROR", 0.0)]
            }
    
    return results

class ImageLabelingDialog(QDialog):
    """
    Dialog for labeling captured images with appropriate categories.
    Appears after an image is captured to collect metadata about the image.
    """
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.selected_labels = []
        
        # Set dialog properties
        self.setWindowTitle("Image Classification")
        self.setMinimumSize(600, 700)  # Increased height for better visibility
        
        # Restore saved window position and size (before creating layout)
        restore_window_geometry(self, "NFC-Detector", "ImageLabeling")
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Add image preview
        preview_label = QLabel("Image Preview:")
        main_layout.addWidget(preview_label)
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(320, 240)
        main_layout.addWidget(self.image_preview)
        
        # Display the image preview
        self.display_image_preview()
        
        # Top section: Image Type Groups
        image_type_group = QGroupBox("Image Type Selection")
        image_type_layout = QVBoxLayout()
        image_type_group.setLayout(image_type_layout)
        
        # Create button group for mutually exclusive card type selection
        self.card_type_group = QButtonGroup(self)
        
        # 1. Corner group with its special issue checkbox
        corner_section = QGroupBox("Card Corner")
        corner_layout = QHBoxLayout()
        corner_section.setLayout(corner_layout)
        
        self.corner_radio = QRadioButton("Card Corner")
        self.card_type_group.addButton(self.corner_radio, 1)
        self.corner_special_check = QCheckBox("Special Issue (Blurry/Wrong Orientation)")
        # Special issue checkbox starts disabled until its radio button is selected
        self.corner_special_check.setEnabled(False)
        
        corner_layout.addWidget(self.corner_radio)
        corner_layout.addWidget(self.corner_special_check, 1)  # Stretch to fill available space
        image_type_layout.addWidget(corner_section)
        
        # 2. Side group with its special issue checkbox
        side_section = QGroupBox("Card Side")
        side_layout = QHBoxLayout()
        side_section.setLayout(side_layout)
        
        self.side_radio = QRadioButton("Card Side")
        self.card_type_group.addButton(self.side_radio, 2)
        self.side_special_check = QCheckBox("Special Issue (Blurry/Wrong Orientation)")
        # Special issue checkbox starts disabled until its radio button is selected
        self.side_special_check.setEnabled(False)
        
        side_layout.addWidget(self.side_radio)
        side_layout.addWidget(self.side_special_check, 1)  # Stretch to fill available space
        image_type_layout.addWidget(side_section)
        
        main_layout.addWidget(image_type_group)
        
        # Container for normal corner options (when special issue is NOT checked)
        self.normal_corner_options = QGroupBox("Corner Details")
        normal_corner_layout = QVBoxLayout()
        self.normal_corner_options.setLayout(normal_corner_layout)
        
        # Front or Back (renamed to Card Face for clarity)
        front_back_group = QGroupBox("Card Face")
        front_back_layout = QHBoxLayout()
        front_back_group.setLayout(front_back_layout)
        
        self.front_back_group = QButtonGroup(self)
        self.front_radio = QRadioButton("Front")
        self.back_radio = QRadioButton("Back")
        self.front_back_group.addButton(self.front_radio, 1)
        self.front_back_group.addButton(self.back_radio, 2)
        
        # Select front by default
        self.front_radio.setChecked(True)
        
        front_back_layout.addWidget(self.front_radio)
        front_back_layout.addWidget(self.back_radio)
        normal_corner_layout.addWidget(front_back_group)
        
        # Factory or NFC
        factory_nfc_group = QGroupBox("Card Type")
        factory_nfc_layout = QHBoxLayout()
        factory_nfc_group.setLayout(factory_nfc_layout)
        
        self.factory_nfc_group = QButtonGroup(self)
        self.factory_radio = QRadioButton("Factory Cut")  # Updated from "Factory/Real Card"
        self.nfc_radio = QRadioButton("NFC Card")
        self.factory_nfc_group.addButton(self.factory_radio, 1)
        self.factory_nfc_group.addButton(self.nfc_radio, 2)
        
        # Select factory by default
        self.factory_radio.setChecked(True)
        
        factory_nfc_layout.addWidget(self.factory_radio)
        factory_nfc_layout.addWidget(self.nfc_radio)
        normal_corner_layout.addWidget(factory_nfc_group)
        
        # Corner quality checkboxes
        quality_group = QGroupBox("Corner Quality (Optional)")
        quality_layout = QHBoxLayout()
        quality_group.setLayout(quality_layout)
        
        self.wonky_check = QCheckBox("Wonky Corner")
        self.square_check = QCheckBox("Square Corner")
        
        quality_layout.addWidget(self.wonky_check)
        quality_layout.addWidget(self.square_check)
        normal_corner_layout.addWidget(quality_group)
        
        main_layout.addWidget(self.normal_corner_options)
        
        # Container for corner special issue options
        self.corner_special_options = QGroupBox("Corner Issue Details")
        corner_special_layout = QVBoxLayout()
        self.corner_special_options.setLayout(corner_special_layout)
        
        # Add explanation about the classification priority
        corner_issue_explanation = QLabel(
            "Note: Classification priority - orientation issues are checked first. "
            "If an image has wrong orientation, it cannot be classified for blurriness."
        )
        corner_issue_explanation.setWordWrap(True)
        corner_special_layout.addWidget(corner_issue_explanation)
        
        # Create radio button group for corner issue classification hierarchy
        self.corner_issue_group = QButtonGroup(self)
        self.corner_orientation_radio = QRadioButton("Wrong Orientation (corners)")
        self.corner_blurry_radio = QRadioButton("Blurry (corners)")
        
        # Remove the "Normal" option as it contradicts the special issue checkbox
        self.corner_issue_group.addButton(self.corner_orientation_radio, 1)
        self.corner_issue_group.addButton(self.corner_blurry_radio, 2)
        
        # Select orientation by default
        self.corner_orientation_radio.setChecked(True)
        
        corner_special_layout.addWidget(self.corner_orientation_radio)
        corner_special_layout.addWidget(self.corner_blurry_radio)
        
        main_layout.addWidget(self.corner_special_options)
        
        # Container for normal side options (when special issue is NOT checked)
        self.normal_side_options = QGroupBox("Side Details")
        side_normal_layout = QVBoxLayout()
        self.normal_side_options.setLayout(side_normal_layout)
        
        # Front or Back for sides
        side_face_group = QGroupBox("Card Face")
        side_face_layout = QHBoxLayout()
        side_face_group.setLayout(side_face_layout)
        
        self.side_face_group = QButtonGroup(self)
        self.side_front_radio = QRadioButton("Front")
        self.side_back_radio = QRadioButton("Back")
        self.side_face_group.addButton(self.side_front_radio, 1)
        self.side_face_group.addButton(self.side_back_radio, 2)
        
        # Select front by default
        self.side_front_radio.setChecked(True)
        
        side_face_layout.addWidget(self.side_front_radio)
        side_face_layout.addWidget(self.side_back_radio)
        side_normal_layout.addWidget(side_face_group)
        
        # Factory or NFC for sides
        side_type_group = QGroupBox("Card Type")
        side_type_layout = QHBoxLayout()
        side_type_group.setLayout(side_type_layout)
        
        self.side_type_group = QButtonGroup(self)
        self.side_factory_radio = QRadioButton("Factory Cut")
        self.side_nfc_radio = QRadioButton("NFC Card")
        self.side_type_group.addButton(self.side_factory_radio, 1)
        self.side_type_group.addButton(self.side_nfc_radio, 2)
        
        # Select factory by default
        self.side_factory_radio.setChecked(True)
        
        side_type_layout.addWidget(self.side_factory_radio)
        side_type_layout.addWidget(self.side_nfc_radio)
        side_normal_layout.addWidget(side_type_group)
        
        # Die cut vs Rough cut radio buttons
        cut_group = QGroupBox("Side Cut Type")
        cut_layout = QHBoxLayout()
        cut_group.setLayout(cut_layout)
        
        self.cut_type_group = QButtonGroup(self)
        self.die_cut_radio = QRadioButton("Die Cut")
        self.rough_cut_radio = QRadioButton("Rough Cut")
        self.cut_type_group.addButton(self.die_cut_radio, 1)
        self.cut_type_group.addButton(self.rough_cut_radio, 2)
        
        # Select die cut by default
        self.die_cut_radio.setChecked(True)
        
        cut_layout.addWidget(self.die_cut_radio)
        cut_layout.addWidget(self.rough_cut_radio)
        side_normal_layout.addWidget(cut_group)
        
        # Add explanation about die cut vs rough cut
        cut_explanation = QLabel(
            "Die Cut vs Rough Cut: Cards are cut in 2 stages - first roughly cut into rectangles "
            "(slitting machine), then corners and excess cardstock are trimmed (die cutting machine). "
            "Select 'Die Cut' for normal factory edges, or 'Rough Cut' when the first cut appears "
            "rougher than the rest of the die cut edge."
        )
        cut_explanation.setWordWrap(True)
        cut_explanation.setStyleSheet("font-style: italic; color: #666;")
        side_normal_layout.addWidget(cut_explanation)
        
        main_layout.addWidget(self.normal_side_options)
        
        # Container for side special issue options
        self.side_special_options = QGroupBox("Side Issue Details")
        side_special_layout = QVBoxLayout()
        self.side_special_options.setLayout(side_special_layout)
        
        # Add explanation about the classification priority for sides
        side_issue_explanation = QLabel(
            "Note: Classification priority - orientation issues are checked first. "
            "If an image has wrong orientation, it cannot be classified for blurriness."
        )
        side_issue_explanation.setWordWrap(True)
        side_special_layout.addWidget(side_issue_explanation)
        
        # Create radio button group for side issue classification hierarchy
        self.side_issue_group = QButtonGroup(self)
        self.side_orientation_radio = QRadioButton("Wrong Orientation (sides)")
        self.side_blurry_radio = QRadioButton("Blurry (sides)")
        
        # Remove the "Normal" option as it contradicts the special issue checkbox
        self.side_issue_group.addButton(self.side_orientation_radio, 1)
        self.side_issue_group.addButton(self.side_blurry_radio, 2)
        
        # Select orientation by default
        self.side_orientation_radio.setChecked(True)
        
        side_special_layout.addWidget(self.side_orientation_radio)
        side_special_layout.addWidget(self.side_blurry_radio)
        
        main_layout.addWidget(self.side_special_options)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Connect signals for UI updates
        self.corner_radio.toggled.connect(self._update_ui_for_selection)
        self.side_radio.toggled.connect(self._update_ui_for_selection)
        self.corner_special_check.toggled.connect(self._update_ui_for_selection)
        self.side_special_check.toggled.connect(self._update_ui_for_selection)
        
        # Add connections for factory/NFC radio buttons to update cut type visibility
        self.side_factory_radio.toggled.connect(self._update_cut_type_visibility)
        self.side_nfc_radio.toggled.connect(self._update_cut_type_visibility)
        
        # Load previous settings before setting default values
        self.load_settings()
        
        # Update UI based on loaded settings
        self._update_ui_for_selection()
        self._update_cut_type_visibility()
    
    def accept(self):
        """Override accept to save settings and window geometry before closing dialog"""
        # Save current settings and window geometry
        self.save_settings()
        save_window_geometry(self, "NFC-Detector", "ImageLabeling")
        
        # Continue with standard accept behavior
        super().accept()
    
    def reject(self):
        """Override reject to save window geometry before closing dialog"""
        # Save window geometry (but not settings since user canceled)
        save_window_geometry(self, "NFC-Detector", "ImageLabeling")
        
        # Continue with standard reject behavior
        super().reject()
    
    def load_settings(self):
        """Load saved settings and apply them to the UI elements"""
        settings = QSettings("NFC-Detector", "ImageLabeling")
        
        # Load image type (corner/side)
        image_type = settings.value("image_type", "corner", type=str)
        if image_type == "corner":
            self.corner_radio.setChecked(True)
            
            # Load special issue checkbox state
            special_issue = settings.value("corner/special_issue", False, type=bool)
            self.corner_special_check.setChecked(special_issue)
            
            if special_issue:
                # Load special issue selection (orientation/blurry)
                issue_type = settings.value("corner/issue_type", "orientation", type=str)
                if issue_type == "blurry":
                    self.corner_blurry_radio.setChecked(True)
                else:
                    self.corner_orientation_radio.setChecked(True)
            else:
                # Load corner-specific settings
                # Card face (front/back)
                card_face = settings.value("corner/card_face", "front", type=str)
                if card_face == "back":
                    self.back_radio.setChecked(True)
                else:
                    self.front_radio.setChecked(True)
                
                # Card type (factory/nfc)
                card_type = settings.value("corner/card_type", "factory", type=str)
                if card_type == "nfc":
                    self.nfc_radio.setChecked(True)
                else:
                    self.factory_radio.setChecked(True)
                
                # Load quality checkboxes
                self.wonky_check.setChecked(settings.value("corner/wonky", False, type=bool))
                self.square_check.setChecked(settings.value("corner/square", False, type=bool))
        else:  # side
            self.side_radio.setChecked(True)
            
            # Load special issue checkbox state
            special_issue = settings.value("side/special_issue", False, type=bool)
            self.side_special_check.setChecked(special_issue)
            
            if special_issue:
                # Load special issue selection (orientation/blurry)
                issue_type = settings.value("side/issue_type", "orientation", type=str)
                if issue_type == "blurry":
                    self.side_blurry_radio.setChecked(True)
                else:
                    self.side_orientation_radio.setChecked(True)
            else:
                # Load side-specific settings
                # Card face (front/back)
                card_face = settings.value("side/card_face", "front", type=str)
                if card_face == "back":
                    self.side_back_radio.setChecked(True)
                else:
                    self.side_front_radio.setChecked(True)
                
                # Card type (factory/nfc)
                card_type = settings.value("side/card_type", "factory", type=str)
                if card_type == "nfc":
                    self.side_nfc_radio.setChecked(True)
                else:
                    self.side_factory_radio.setChecked(True)
                
                # Cut type (die-cut/rough-cut)
                cut_type = settings.value("side/cut_type", "die", type=str)
                if cut_type == "rough":
                    self.rough_cut_radio.setChecked(True)
                else:
                    self.die_cut_radio.setChecked(True)
    
    def save_settings(self):
        """Save current UI selection state to settings"""
        settings = QSettings("NFC-Detector", "ImageLabeling")
        
        # Save image type (corner/side)
        settings.setValue("image_type", "corner" if self.corner_radio.isChecked() else "side")
        
        # Save corner-specific settings
        if self.corner_radio.isChecked():
            # Save special issue state
            is_special = self.corner_special_check.isChecked()
            settings.setValue("corner/special_issue", is_special)
            
            if is_special:
                # Save special issue type
                is_blurry = self.corner_blurry_radio.isChecked()
                settings.setValue("corner/issue_type", "blurry" if is_blurry else "orientation")
            else:
                # Save normal corner settings
                settings.setValue("corner/card_face", "back" if self.back_radio.isChecked() else "front")
                settings.setValue("corner/card_type", "nfc" if self.nfc_radio.isChecked() else "factory")
                settings.setValue("corner/wonky", self.wonky_check.isChecked())
                settings.setValue("corner/square", self.square_check.isChecked())
        
        # Save side-specific settings
        if self.side_radio.isChecked():
            # Save special issue state
            is_special = self.side_special_check.isChecked()
            settings.setValue("side/special_issue", is_special)
            
            if is_special:
                # Save special issue type
                is_blurry = self.side_blurry_radio.isChecked()
                settings.setValue("side/issue_type", "blurry" if is_blurry else "orientation")
            else:
                # Save normal side settings
                settings.setValue("side/card_face", "back" if self.side_back_radio.isChecked() else "front")
                settings.setValue("side/card_type", "nfc" if self.side_nfc_radio.isChecked() else "factory")
                settings.setValue("side/cut_type", "rough" if self.rough_cut_radio.isChecked() else "die")
    
    def _update_ui_for_selection(self):
        """Update UI based on the selected options"""
        # Handle corner radio button state
        if self.corner_radio.isChecked():
            # Enable corner special issue checkbox
            self.corner_special_check.setEnabled(True)
            
            # Disable and uncheck side special issue checkbox
            self.side_special_check.setEnabled(False)
            self.side_special_check.setChecked(False)
            
            # Hide all side-related UI elements
            self.normal_side_options.setVisible(False)
            self.side_special_options.setVisible(False)
            
            # Show or hide corner options based on special issue checkbox
            if self.corner_special_check.isChecked():
                self.normal_corner_options.setVisible(False)
                self.corner_special_options.setVisible(True)
            else:
                self.normal_corner_options.setVisible(True)
                self.corner_special_options.setVisible(False)
        
        # Handle side radio button state
        elif self.side_radio.isChecked():
            # Enable side special issue checkbox
            self.side_special_check.setEnabled(True)
            
            # Disable and uncheck corner special issue checkbox
            self.corner_special_check.setEnabled(False)
            self.corner_special_check.setChecked(False)
            
            # Hide all corner-related UI elements
            self.normal_corner_options.setVisible(False)
            self.corner_special_options.setVisible(False)
            
            # Show or hide side options based on special issue checkbox
            if self.side_special_check.isChecked():
                self.normal_side_options.setVisible(False)
                self.side_special_options.setVisible(True)
            else:
                self.normal_side_options.setVisible(True)
                self.side_special_options.setVisible(False)
                # Update cut type visibility based on factory/NFC selection
                self._update_cut_type_visibility()
    
    def _update_cut_type_visibility(self):
        """Update visibility of cut type options based on card type selection"""
        # Only proceed if we're in side mode and normal (non-special issue) options
        if self.side_radio.isChecked() and not self.side_special_check.isChecked():
            # Get all child widgets of normal_side_options
            for i in range(self.normal_side_options.layout().count()):
                widget = self.normal_side_options.layout().itemAt(i).widget()
                # Check if this is the cut group widget
                if isinstance(widget, QGroupBox) and widget.title() == "Side Cut Type":
                    # Show for factory-cut, hide for NFC
                    widget.setVisible(self.side_factory_radio.isChecked())
                # Also handle the cut explanation label
                elif isinstance(widget, QLabel) and widget.text().startswith("Die Cut vs Rough Cut:"):
                    widget.setVisible(self.side_factory_radio.isChecked())
    
    def get_image_labels(self):
        """Get selected labels based on user choices"""
        labels = []
        
        # Handle corner image labels
        if self.corner_radio.isChecked():
            if self.corner_special_check.isChecked():
                # Special issues are exceptions - orientation and blurry ignore other attributes
                if self.corner_orientation_radio.isChecked():
                    labels.append("corners-wrong-orientation")  # Simple label for orientation issues
                elif self.corner_blurry_radio.isChecked():
                    labels.append("corners-blurry")  # Simple label for blurry issues
            else:
                # Get base label components for normal corner cases
                card_type = "factory-cut" if self.factory_radio.isChecked() else "nfc"
                card_face = "fronts" if self.front_radio.isChecked() else "backs"
                
                # Special qualities get their own comprehensive labels
                if self.wonky_check.isChecked():
                    labels.append(f"{card_type}-corners-{card_face}-wonky")
                elif self.square_check.isChecked():
                    labels.append(f"{card_type}-corners-{card_face}-square")
                else:
                    # Normal corner without special qualities
                    labels.append(f"{card_type}-corners-{card_face}")
        
        # Handle side image labels
        elif self.side_radio.isChecked():
            if self.side_special_check.isChecked():
                # Special issues are exceptions - orientation and blurry ignore other attributes
                if self.side_orientation_radio.isChecked():
                    labels.append("sides-wrong-orientation")  # Simple label for orientation issues
                elif self.side_blurry_radio.isChecked():
                    labels.append("sides-blurry")  # Simple label for blurry issues
            else:
                # Get base components for normal side labels
                card_type = "factory-cut" if self.side_factory_radio.isChecked() else "nfc"
                card_face = "fronts" if self.side_front_radio.isChecked() else "backs"
                
                # Add comprehensive label including cut type (only for factory-cut)
                if card_type == "factory-cut":
                    if self.rough_cut_radio.isChecked():
                        labels.append(f"{card_type}-sides-{card_face}-rough-cut")
                    else:
                        labels.append(f"{card_type}-sides-{card_face}-die-cut")
                else:
                    # For NFC, don't include cut type
                    labels.append(f"{card_type}-sides-{card_face}")
        
        return labels
    
    def display_image_preview(self):
        """Display the captured image preview"""
        if self.image is not None:
            # Convert CV2 image to Qt format for display
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # Create QImage and QPixmap
            qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale down if needed
            scaled_pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set the pixmap
            self.image_preview.setPixmap(scaled_pixmap)

def save_labeled_image(image, labels, base_dir="newly-captured-data"):
    """
    Save the captured image to appropriate directories based on labels.
    Returns the list of saved file paths.
    
    Labels are expected to follow the new naming convention:
    - factory-cut-corners-fronts, nfc-corners-fronts, etc. for normal corner images
    - corners-blurry, corners-wrong-orientation for special cases
    - wonky-corner, square-corner for corner qualities
    """
    # Create base directory if it doesn't exist
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    saved_paths = []
    
    # For each label, save to the corresponding directory
    for label in labels:
        # Create directory for this label if needed
        label_dir = base_path / label
        label_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp and label
        filename = f"{timestamp}-{label}.jpg"
        filepath = label_dir / filename
        
        # Save the image
        cv2.imwrite(str(filepath), image)
        saved_paths.append(filepath)
    
    return saved_paths

def main():
    """Main function to run the video stream handler"""
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set organization name early so it applies to all dialogs
    QApplication.setOrganizationName("NFC-Detector")
    
    parser = argparse.ArgumentParser(description='USB Microscope Video Stream Handler')
    parser.add_argument('--device', type=int, help='Specify device ID to use')
    parser.add_argument('--list', action='store_true', help='List all available video devices')
    args = parser.parse_args()
    
    print("USB Microscope Video Stream Handler")
    print("==================================")
    
    # List all devices if explicitly requested
    if (args.list):
        print("Searching for video devices...")
        devices = list_video_devices()
        return
    
    # Find available devices with simplified check
    print("Searching for video devices...")
    devices = list_video_devices()
    
    if not devices:
        show_error_dialog("No Video Devices", 
                         "No video devices were found.\nPlease check connections and try again.")
        return 1
    
    # Device selection - either from args or from setup dialog
    if args.device is not None and args.device in devices:
        device_id = args.device
    else:
        # Show setup dialog
        dialog = SetupDialog(devices)
        result = dialog.exec_()
        
        # If dialog was rejected or no CUDA
        if result == QDialog.Rejected or not torch.cuda.is_available():
            return 1
            
        # Get selected device
        device_id = dialog.get_selected_device()
    
    # Make sure we have a device ID
    if device_id is None:
        show_error_dialog("No Device Selected", 
                         "No camera device was selected.\nPlease restart and select a device.")
        return 1
    
    # Display information about selected device
    info = devices[device_id]
    print(f"Selected device {device_id}: {info['name']}")
    
    # Launch the video stream
    window = display_video_stream(device_id)
    
    # Enter the Qt event loop and wait until the window is closed
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())  # Pass the exit code to sys.exit
