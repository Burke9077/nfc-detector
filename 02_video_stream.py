import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def list_video_devices():
    """
    List all available video devices and their properties.
    Returns a dictionary of device IDs and their information.
    """
    devices = {}
    # Try a reasonable range of device IDs
    for i in range(10):  # Usually video devices are 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get basic device information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            devices[i] = {
                'id': i,
                'resolution': (width, height),
                'fps': fps,
                'name': f"Camera {i}"  # Default name since OpenCV doesn't provide device names
            }
            
            # Try to set the resolution to 1280x720 to test if supported
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            devices[i]['supports_720p'] = (actual_width == 1280 and actual_height == 720)
            
            # Release the device
            cap.release()
    
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
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    print("\nMultiple possible microscope devices found. Please select one:")
    for i, dev_id in enumerate(candidates):
        info = devices[dev_id]
        print(f"[{i+1}] Device {dev_id}: {info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.1f}fps")
    
    while True:
        try:
            choice = int(input("\nEnter number [1-{}]: ".format(len(candidates))))
            if 1 <= choice <= len(candidates):
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

def display_video_stream(device_id, target_resolution=(1280, 720)):
    """
    Display video stream from specified device with target resolution.
    Allow capturing of still frames with keyboard input.
    Falls back to matplotlib if OpenCV GUI is not supported.
    """
    # Create output directory for captured frames
    output_dir = Path("captured_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Open video capture
    cap = cv2.VideoCapture(device_id)
    
    # Try to set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])
    
    # Get actual resolution (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Streaming at resolution: {actual_width}x{actual_height}")
    
    if not cap.isOpened():
        print(f"Error: Could not open device {device_id}")
        return False
    
    frame_count = 0  # For naming captured frames
    
    # Check if OpenCV GUI is supported
    opencv_gui = check_opencv_gui_support()
    
    if opencv_gui:
        # Use OpenCV's native GUI
        print("\nControls:")
        print("  Press 'c' to capture a frame")
        print("  Press 'q' to quit")
        
        while True:
            # Read frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            try:
                # Display the frame
                cv2.imshow(f"USB Microscope (Device {device_id})", frame)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    print("Exiting video stream...")
                    break
                
                # 'c' to capture a frame
                elif key == ord('c'):
                    frame_count += 1
                    filename = output_dir / f"captured_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"Captured frame: {filename}")
            except cv2.error:
                print("OpenCV GUI failed. Switching to matplotlib...")
                cv2.destroyAllWindows()
                opencv_gui = False
                break
        
        # Release resources
        cap.release()
        if opencv_gui:
            cv2.destroyAllWindows()
            return True
    
    if not opencv_gui:
        # Fall back to matplotlib for display
        print("\nUsing matplotlib for display (OpenCV GUI not available)")
        print("Controls:")
        print("  Press 'c' to capture a frame")
        print("  Press 'q' to quit")
        print("  Close the window to exit")
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Initialize with a blank frame
        img_display = ax.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))
        ax.set_title(f"USB Microscope (Device {device_id})")
        ax.axis('off')
        
        # Variable to store the last frame for capture
        last_frame = None
        
        def capture_frame(event):
            nonlocal frame_count, last_frame
            if event.key == 'c' and last_frame is not None:
                frame_count += 1
                filename = output_dir / f"captured_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(filename), last_frame)
                print(f"Captured frame: {filename}")
            elif event.key == 'q':
                plt.close()
        
        # Connect the key press event
        fig.canvas.mpl_connect('key_press_event', capture_frame)
        
        def update(frame_num):
            nonlocal last_frame
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_display.set_data(frame_rgb)
                last_frame = frame
                return [img_display]
            return []
        
        # Create animation to update the display
        ani = FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)
        
        plt.show()
        
        # Clean up resources
        cap.release()
    
    return True

def main():
    """Main function to run the video stream handler"""
    parser = argparse.ArgumentParser(description='USB Microscope Video Stream Handler')
    parser.add_argument('--device', type=int, help='Specify device ID to use')
    parser.add_argument('--list', action='store_true', help='List all available video devices')
    parser.add_argument('--fix-opencv', action='store_true', help='Attempt to fix OpenCV GUI issues by reinstalling')
    args = parser.parse_args()
    
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
        print(f"  Device {dev_id}: {info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.1f}fps")
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
    main()
