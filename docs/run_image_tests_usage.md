# Model Training Guide

## Overview
This guide explains how to use `01_run_image_tests.py` to train the various models needed for NFC card detection. The script trains multiple specialized models that work together in a hierarchical classification approach.

## What It Does
The script trains a complete set of models for NFC detection:

1. **Quality Check Model (01)** - Determines if an image is usable or problematic
2. **Front/Back Models (10-11)** - Identifies if a card shows the front or back side
3. **Factory vs NFC Models (30-33)** - Specialized models that compare factory-cut vs NFC cards for each position

## Requirements
- NVIDIA GPU with 4GB+ VRAM
- Properly organized dataset (see Data Organization below)
- 16GB+ system RAM recommended
- Python 3.8+ with all dependencies installed

## Usage
Basic usage (runs all seven tests):
```bash
python 01_run_image_tests.py
```

### Command Line Options
- `--resume`: Continue from previous checkpoints without asking and skip completed models
- `--skip-completed`: Skip tests that have already successfully completed
- `--only <test>`: Run only a specific test

Valid options for `--only` are:
- `quality`: Image quality model (01)
- `corner-front-back`: Corner front/back classification (10)
- `side-front-back`: Side front/back classification (11)
- `corner-front`: Corner front factory vs NFC (30)
- `corner-back`: Corner back factory vs NFC (31)
- `side-front`: Side front factory vs NFC (32)
- `side-back`: Side back factory vs NFC (33)

### Examples
Resume training after an interruption:
```bash
python 01_run_image_tests.py --resume
```

Skip models that are already trained:
```bash
python 01_run_image_tests.py --skip-completed
```

Train only the image quality model:
```bash
python 01_run_image_tests.py --only quality
```

Train only the corner front factory vs NFC model:
```bash
python 01_run_image_tests.py --only corner-front
```

## Data Organization
The script expects your data in this structure:
```
data/
    corners-blurry/
        image1.jpg
        image2.jpg
        ...
    factory-cut-corners-backs/
        ...
    nfc-corners-backs/
        ...
    sides-blurry/
        ...
```

## Training Process
For each test, the script:

1. **Sets up a temporary environment** in the `nfc_detector_work_dir` folder
2. **Copies images** to temporary directories with the proper class structure
3. **Applies optional edge enhancement** to highlight card edge features
4. **Trains models** using a ResNet50 architecture with fine-tuning
5. **Evaluates** using test-time augmentation (TTA)
6. **Saves models** to the `nfc_models` directory
7. **Generates visualizations** including confusion matrices and top losses
8. **Cleans up** temporary working files

## Output
The script produces seven model files in the `nfc_models` directory:

- `quality_check_model.pkl`
- `corner_front_back_model.pkl`
- `side_front_back_model.pkl`
- `corner_front_model.pkl`
- `corner_back_model.pkl`
- `side_front_model.pkl`
- `side_back_model.pkl`

## Training Configuration
Default settings include:
- Image resolution: 720x1280 (high resolution to preserve edge details)
- Edge enhancement: Applied to 30% of images
- Test Time Augmentation (TTA): Enabled
- Basic epochs: 25 (30 for all-categories model)
- Early stopping: Enabled with patience=5

## Troubleshooting

### Memory Issues
If you receive CUDA out-of-memory errors:
- Check GPU usage before starting: The script displays memory info at startup
- Reduce batch size: Look for `bs=8` in `image_test_utils.py`
- Use progressive resizing: Not recommended for edge detection, but can help with memory

### Training Failures
If training fails before completion:
1. The script preserves working directory for inspection
2. Use `--resume` to continue from the last checkpoint
3. Error messages and traceback will indicate the specific issue
4. Check the GPU memory graph displayed at startup

### Long Training Times
Training time depends on:
- Dataset size (more images = longer training)
- Image resolution (higher resolution = longer training)
- GPU capabilities (faster GPU = faster training)

For a few thousand images, expect 1-3 hours total training time on a mid-range GPU.

## Workflow Integration
This script fits in the NFC detection workflow as follows:

1. Organize your initial dataset in the `data` directory
2. Run `01_run_image_tests.py` to train initial models
3. Use `02_video_stream.py` to capture additional images
4. Merge new images with `03_merge_data.py`
5. Retrain models by running `01_run_image_tests.py` again

## Advanced Usage

### Custom Training
If you need to customize the training process beyond the command-line options:
- Model architecture can be changed in `image_test_utils.py` (`resnet50` is default)
- Image augmentation parameters can be adjusted in the same file
- Training hyperparameters like learning rate are automatically determined

### Using Trained Models
After training completes, the models are ready to use with:
- The video stream interface (`02_video_stream.py`)
- Any Python code using FastAI's `load_learner()`
