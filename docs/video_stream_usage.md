# Video Stream Capture Guide

## Overview
The `02_video_stream.py` script provides a graphical interface for capturing and automatically classifying card images using your trained models. It lets you:

- View a live camera feed from a USB microscope or webcam
- Run real-time inference on captured frames using your trained models
- Label and save images with proper category tags
- Build your dataset for further training

## Requirements
- A USB camera or microscope
- PyQt5 properly installed
- Trained models in the `nfc_models` directory
- CUDA-capable GPU (for model inference)

## Usage
Basic usage:
```bash
python 02_video_stream.py
```

### Common Options
- `--list`: Show a list of available camera devices and exit
- `--device <id>`: Specify the camera device ID to use (skips selection dialog)

### Examples
List all connected cameras:
```bash
python 02_video_stream.py --list
```

Use a specific camera device (ID 1):
```bash
python 02_video_stream.py --device 1
```

## Interface Guide

### Camera Selection
When launched, the application will:
1. Check for GPU capabilities
2. Scan for connected cameras
3. Show a setup dialog where you can:
   - View GPU information
   - Select and preview available cameras

### Main Interface
The main window consists of:

1. **Video Display (Left)**: Shows the live camera feed
2. **Model Predictions (Right)**: Displays prediction results from your models
3. **Control Buttons (Bottom)**:
   - **Capture Frame and Analyze (C)**: Captures the current frame and runs inference
   - **Quit (Q)**: Exits the application

### Image Labeling
After capturing an image, a labeling dialog appears where you can:

1. Select the card element type (corner or side)
2. Mark if the image has special issues (blurry/wrong orientation)
3. Choose detailed attributes:
   - For corners: front/back, factory/NFC, quality attributes
   - For sides: front/back, factory/NFC, cut type (die-cut/rough-cut)

## Saved Images
Images are automatically saved to the `newly-captured-data` directory in appropriate subfolders based on labels:

- `factory-cut-corners-fronts` - Factory-cut card front corners
- `nfc-corners-backs` - NFC card back corners
- `sides-wrong-orientation` - Card sides with incorrect orientation
- etc.

## Workflow Integration
This script fits in the NFC detection workflow as follows:
1. Train initial models using `01_run_image_tests.py`
2. Use `02_video_stream.py` to capture and label additional images
3. Merge the new data into your training set using `03_merge_data.py`
4. Retrain models with the expanded dataset

## Tips
- Use proper lighting for best results
- For corners, ensure the corner is clearly visible in the frame
- For sides, position the camera to clearly show the edge cut pattern
- Try to maintain consistent angles and distances for each category
- Use the model predictions to help guide your labeling
