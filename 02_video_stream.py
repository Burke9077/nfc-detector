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
import matplotlib.pyplot as plt

# Add PyQt5 imports - exit if not installed
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                              QVBoxLayout, QHBoxLayout, QLabel, QStatusBar)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
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
    List all available video devices and their properties.
    Returns a dictionary of device IDs and their information.
    Shows progress during detection to avoid silent delays.
    """
    devices = {}
    print("Scanning for video devices (this may take a moment)...")
    
    # Try a reasonable range of device IDs
    for i in range(10):  # Usually video devices are 0-9
        print(f"  Checking device {i}...", end='', flush=True)
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get basic device information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f" found! ({width}x{height})")
                
                # Try to get more detailed device info
                print(f"  Getting additional info for device {i}...", end='', flush=True)
                extra_info = get_device_info(i)
                print(" done")
                
                devices[i] = {
                    'id': i,
                    'resolution': (width, height),
                    'fps': fps,
                    'name': extra_info["name"],
                    'manufacturer': extra_info["manufacturer"]
                }
                
                # Try to set the resolution to 1280x720 to test if supported
                print(f"  Testing 720p support for device {i}...", end='', flush=True)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                devices[i]['supports_720p'] = (actual_width == 1280 and actual_height == 720)
                supports_text = "✓ supported" if devices[i]['supports_720p'] else "not supported"
                print(f" {supports_text}")
                
                # Release the device
                cap.release()
            else:
                elapsed = time.time() - start_time
                print(f" not available ({elapsed:.1f}s)")
        except Exception as e:
            print(f" error: {str(e)}")
    
    print(f"Device scan complete. Found {len(devices)} device(s).")
    return devices

def find_microscope(devices):
    """
    Try to identify which device is likely the USB microscope.
    Returns a list of devices that support 1280x720 resolution.
    """
    # Find all devices that support 720p
    matching_devices = []
    for dev_id, info in devices.items():
        if info['supports_720p']:
            matching_devices.append(dev_id)
            print(f"Found possible microscope: Device {dev_id} (supports 720p)")
    
    # If we found 720p devices, return the list
    if matching_devices:
        return matching_devices
    
    # If no 720p device, try to find one close to that resolution
    closest_dev = None
    closest_diff = float('inf')
    
    for dev_id, info in devices.items():
        width, height = info['resolution']
        # Calculate how close this is to 1280x720
        diff = abs(width - 1280) + abs(height - 720)
        if diff < closest_diff:
            closest_diff = diff
            closest_dev = dev_id
    
    if closest_dev is not None:
        print(f"Found best match for microscope: Device {closest_dev} - resolution {devices[closest_dev]['resolution']}")
        return [closest_dev]
    
    return []

def select_device(devices, candidates):
    """
    Prompt user to select a device from candidates.
    Shows list with option numbers distinct from device IDs to avoid confusion.
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    print("\nMultiple possible microscope devices found. Please select one:")
    # Display options with simple option numbers (1, 2, 3...)
    for i, dev_id in enumerate(candidates):
        info = devices[dev_id]
        mfg = f" ({info['manufacturer']})" if info['manufacturer'] != "Unknown" else ""
        print(f"[{i+1}] Device {dev_id}: {info['name']}{mfg}: {info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.1f}fps")
    
    while True:
        try:
            choice_str = input(f"\nEnter option number [1-{len(candidates)}]: ")
            # Convert to int and validate as option number (1-based)
            choice = int(choice_str)
            if 1 <= choice <= len(candidates):
                # Convert option number (1-based) back to device ID
                selected_id = candidates[choice-1]
                print(f"Selected: Device {selected_id}")
                return selected_id
            else:
                print(f"Please enter a number between 1 and {len(candidates)}")
        except ValueError:
            print("Please enter a valid number")

def check_opencv_gui_support():
    """Check if OpenCV has GUI support by trying to create a window."""
    try:
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        print("Warning: OpenCV was compiled without GUI support.")
        return False

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
        
        # Create output directory
        self.output_dir = Path("captured_frames")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up the UI
        self.setWindowTitle(f"USB Microscope (Device {device_id})")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)
        
        # Button panel
        button_layout = QHBoxLayout()
        
        # Capture button
        self.capture_btn = QPushButton("Capture Frame (C)")
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
        
        # Try to set the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
        
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
        """Capture the current frame"""
        if self.last_frame is not None:
            self.frame_count += 1
            filename = self.output_dir / f"captured_frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(str(filename), self.last_frame)
            self.statusBar.showMessage(f"Captured: {filename}")
            print(f"Captured frame: {filename}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the timer and release the camera
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        event.accept()

def display_video_stream(device_id, target_resolution=(1280, 720)):
    """
    Display video stream using PyQt5 UI.
    """
    # Create output directory for captured frames
    output_dir = Path("captured_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show the UI
    window = MicroscopeUI(device_id, target_resolution)
    window.show()
    
    # Run the application
    return app.exec_() == 0

def check_gpu_status():
    """Check and display GPU information at startup"""
    print("\nChecking GPU status...")
    
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

def main():
    """Main function to run the video stream handler"""
    parser = argparse.ArgumentParser(description='USB Microscope Video Stream Handler')
    parser.add_argument('--device', type=int, help='Specify device ID to use')
    parser.add_argument('--list', action='store_true', help='List all available video devices')
    parser.add_argument('--fix-opencv', action='store_true', help='Attempt to fix OpenCV GUI issues by reinstalling')
    parser.add_argument('--skip-gpu-check', action='store_true', help='Skip the GPU requirement check (not recommended)')
    args = parser.parse_args()
    
    print("USB Microscope Video Stream Handler")
    print("==================================")
    
    # Check GPU status at startup
    cuda_available = check_gpu_status()
    if not cuda_available and not args.skip_gpu_check:
        print("\nExiting: CUDA-enabled GPU is required. Use --skip-gpu-check to override (not recommended).")
        return 1  # Return non-zero exit code to indicate error
    
    # If user wants to fix OpenCV, provide instructions
    if args.fix_opencv:
        print("Attempting to fix OpenCV GUI issues...")
        print("\nOption 1: Install a pre-built version with GUI support:")
        print("  pip uninstall opencv-python")
        print("  pip install opencv-python-headless")
        print("  pip install opencv-contrib-python")
        
        print("\nOption 2: If on Windows, try:")
        print("  pip install opencv-python==4.5.4.60")
        
        print("\nAfter reinstalling, run this script again without the --fix-opencv flag.")
        return
    
    print("USB Microscope Video Stream Handler")
    print("==================================")
    
    # List all devices
    print("Searching for video devices...")
    devices = list_video_devices()
    
    if not devices:
        print("No video devices found. Please check connections and try again.")
        return
    
    print(f"Found {len(devices)} video device(s):")
    for dev_id, info in devices.items():
        mfg = f" ({info['manufacturer']})" if info['manufacturer'] != "Unknown" else ""
        print(f"  Device {dev_id}: {info['name']}{mfg} - {info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.1f}fps")
        if info['supports_720p']:
            print("    ✓ Supports 720p - likely microscope")
    
    # If --list flag is passed, just list devices and exit
    if args.list:
        return
    
    # Use specified device if provided
    if args.device is not None:
        if args.device in devices:
            device_id = args.device
        else:
            print(f"Error: Specified device {args.device} not found")
            return
    else:
        # Try to find the microscope automatically
        candidate_devices = find_microscope(devices)
        device_id = select_device(devices, candidate_devices)
        if device_id is None:
            # If no device was found, use the first available one
            device_id = list(devices.keys())[0] if devices else None
            
    if device_id is None:
        print("Could not identify a suitable video device")
        return
        
    print(f"\nUsing device {device_id} for video stream")
    display_video_stream(device_id)

if __name__ == "__main__":
    sys.exit(main())  # Pass the exit code to sys.exit
