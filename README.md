# NFC Card Detector

An image classification project aimed at detecting subtle differences between factory-cut and non-factory cut (NFC) trading cards using deep learning.

## Project Overview

This project uses deep learning to identify subtle edge differences between factory-cut and home-cut cards. The detector trains multiple classification models:

1. **Factory vs NFC Fronts** - Classifies card fronts as factory or non-factory cuts
2. **Factory vs NFC Backs** - Classifies card backs as factory or non-factory cuts 
3. **Combined Front/Back Test** - Combines front and back data for classification
4. **All Categories Test** - Classifies cards into all four categories (factory fronts, factory backs, NFC fronts, NFC backs)

## Requirements

- Python 3.8+ 
- NVIDIA GPU with at least 4GB VRAM
- CUDA 11.0+ and cuDNN properly installed
- 16GB+ system RAM recommended
- Approximately 10GB storage for models and temporary files
- PyQt5 for the video capture UI

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nfc-detector.git
   cd nfc-detector
   ```

2. Install the required packages using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify your GPU setup:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## Data Organization

Place your images in a folder named `data` with the following structure:
```
data/
    factory-cut-corners-fronts/
        image1.jpg
        image2.jpg
        ...
    factory-cut-corners-backs/
        image1.jpg
        image2.jpg
        ...
    nfc-corners-fronts/
        image1.jpg
        image2.jpg
        ...
    nfc-corners-backs/
        image1.jpg
        image2.jpg
        ...
```

Recommended image specifications:
- Resolution: 720p (1280x720) or higher
- Format: JPG or PNG
- At least 300 images per category for reasonable results, better results at 500,1k,2k+

## Running Tests

Run the complete test suite:
```bash
python 01_run_image_tests.py
```

### Command Line Options:

- **Resume from checkpoints** (useful if previous run was interrupted):
  ```bash
  python 01_run_image_tests.py --resume
  ```

- **Skip tests that have already completed**:
  ```bash
  python 01_run_image_tests.py --skip-completed
  ```

## Project Structure

The project follows a numbered file naming convention to indicate workflow sequence:

1. **01_run_image_tests.py** - The first step: trains initial models from a dataset of images
2. **02_video_stream.py** - Video capture utility for collecting additional data using USB microscope

## Output

The script will:
1. Create temporary working directories
2. Process images (including optional edge enhancement)
3. Train models at high resolution to preserve edge details
4. Generate visualizations (confusion matrices, top losses)
5. Save trained models to the `nfc_models` directory

## Key Features

- **Edge Enhancement**: Highlights card edges using OpenCV for better differentiation
- **High-resolution Processing**: Maintains high image resolution to preserve subtle edge differences
- **Test Time Augmentation (TTA)**: Improves accuracy through multiple predictions on the same image
- **Automatic Checkpointing**: Save progress and resume training if interrupted
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## Model Visualization

The script will display:
- Sample batches from your dataset
- GPU memory usage
- Training progress and metrics
- Confusion matrices
- Samples of the most challenging images

## Troubleshooting

- **CUDA out of memory error**: Reduce the batch size in `image_test_utils.py` (look for `bs=8`)
- **Missing directories**: Ensure your data folders follow the exact naming convention listed above
- **Training failure**: Use the `--resume` flag to continue from the last checkpoint

## License

This project is licensed under the MIT License - see the LICENSE file for details.
