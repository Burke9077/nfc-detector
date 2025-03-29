# NFC Card Detector

An image classification project aimed at detecting subtle differences between factory-cut and non-factory cut (NFC) trading cards using deep learning.

## Project Context

Within the Pokemon trading card community, error cards (particularly miscuts) are highly collectible items. The value of these cards depends on their authenticity - specifically whether they were genuinely miscut during factory production or intentionally cut at home from uncut sheets (known as "Non-Factory Cut" or NFC).

While some collectors appreciate and enjoy NFCs as an affordable alternative to expensive factory errors (authentic uncut sheets can cost $700+), problems arise when NFCs are misrepresented and sold as genuine factory errors at premium prices.

This tool aims to help enthusiasts verify their cards using AI-powered image classification, ensuring transparency in the marketplace and protecting both buyers and sellers.

For more detailed information about Pokemon error card collecting and NFCs, please see our [Collecting Guide](docs/COLLECTING_GUIDE.md).

## Project Overview

This project uses deep learning to identify subtle edge differences between factory-cut and home-cut cards. The detector trains multiple classification models in a hierarchical approach:

1. **Image Quality Model (01)** - Classifies images into corner, side, wrong-orientation, or blurry
2. **Front/Back Identification Models (10-11)** - Determine if the image shows the front or back of a card
   - Model 10: Corner Front/Back Classification
   - Model 11: Side Front/Back Classification
3. **Factory vs NFC Models (30-33)** - Compare factory-cut vs NFC cards for each card position
   - Model 30: Factory vs NFC (Corner Front)
   - Model 31: Factory vs NFC (Corner Back)
   - Model 32: Factory vs NFC (Side Front)
   - Model 33: Factory vs NFC (Side Back)

The goal is to provide collectors with a reliable tool to authenticate error cards and protect themselves from potential scams in the marketplace.

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
- **GPU Optimized**: Takes full advantage of powerful GPUs for high-resolution training

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
