import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import sys
import subprocess
import platform
import re
import torch
import torch.cuda as cuda
import io
from contextlib import redirect_stdout

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
                              QHeaderView)
    from PyQt5.QtCore import Qt, QTimer, QSize
    from PyQt5.QtGui import QImage, QPixmap, QColor
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
        
        # Create output directory
        self.output_dir = Path("captured_frames")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up the UI
        self.setWindowTitle(f"USB Microscope (Device {device_id})")
        self.setMinimumSize(1024, 768)  # Larger window for model results
        
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
        """Capture the current frame and run inference"""
        if self.last_frame is not None:
            self.frame_count += 1
            filename = self.output_dir / f"captured_frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(str(filename), self.last_frame)
            
            # Update status
            self.statusBar.showMessage(f"Captured: {filename}")
            print(f"Captured frame: {filename}")
            
            # Run inference if models are loaded
            if self.models:
                self.statusBar.showMessage(f"Running inference on {filename}...")
                
                # Run inference
                results = run_inference(self.last_frame, self.models)
                
                # Update results table
                self.update_results_table(results)
                
                # Update status
                self.statusBar.showMessage(f"Captured: {filename} - Analysis complete")
            else:
                self.statusBar.showMessage(f"Captured: {filename} - No models loaded for analysis")
    
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
            
            # Add confidence with color based on value
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
            
            # Add tooltips with all class probabilities
            all_probs_text = "\n".join(f"{c}: {p:.2f}%" for c, p in result['all_probs'])
            model_item.setToolTip(all_probs_text)
            pred_item.setToolTip(all_probs_text)
            conf_item.setToolTip(all_probs_text)

class SetupDialog(QDialog):
    """Dialog to show GPU status and allow camera selection."""
    
    def __init__(self, devices=None, parent=None):
        super().__init__(parent)
        self.devices = devices or {}
        self.selected_device = None
        self.preview_widgets = {}
        
        self.setWindowTitle("NFC Detector Setup")
        self.setMinimumSize(800, 600)
        
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
    
    def accept(self):
        """Handle OK button."""
        self.cleanup()
        super().accept()
    
    def reject(self):
        """Handle Cancel button."""
        self.cleanup()
        super().reject()

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
    # Create output directory for captured frames
    output_dir = Path("captured_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Create and show the UI
    window = MicroscopeUI(device_id, target_resolution)
    window.show()
    
    # Bring window to front and give it focus
    window.raise_()
    window.activateWindow()
    
    # Return the window instance so we can keep a reference to it
    return window

def find_and_load_models(models_dir="nfc_models"):
    """Find and load all model files in the models directory"""
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
            # Load model with fastai
            model = load_learner(model_file)
            models[model_name] = model
            print(f"    ✓ Model loaded successfully: {len(model.dls.vocab)} classes")
        except Exception as e:
            print(f"    ✗ Error loading model {model_name}: {str(e)}")
    
    return models

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
            # Get prediction
            pred, pred_idx, probs = model.predict(pil_img)
            
            # Get class names from the model's vocabulary
            class_names = model.dls.vocab
            
            # Create a list of (class_name, probability) tuples sorted by probability
            class_probs = [(str(class_names[i]), float(probs[i]) * 100.0) for i in range(len(class_names))]
            class_probs.sort(key=lambda x: x[1], reverse=True)  # Sort by probability, highest first
            
            # Store results
            results[model_name] = {
                'prediction': str(pred),
                'confidence': float(probs[pred_idx]) * 100.0,  # Convert to percentage
                'all_probs': class_probs
            }
            
        except Exception as e:
            print(f"Error running inference with model {model_name}: {str(e)}")
            results[model_name] = {
                'prediction': "ERROR",
                'confidence': 0.0,
                'all_probs': [("ERROR", 0.0)]
            }
    
    return results

def main():
    """Main function to run the video stream handler"""
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    parser = argparse.ArgumentParser(description='USB Microscope Video Stream Handler')
    parser.add_argument('--device', type=int, help='Specify device ID to use')
    parser.add_argument('--list', action='store_true', help='List all available video devices')
    args = parser.parse_args()
    
    print("USB Microscope Video Stream Handler")
    print("==================================")
    
    # List all devices if explicitly requested
    if args.list:
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
