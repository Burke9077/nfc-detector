# Model Training Guide

## Overview
The `01_run_image_tests.py` script trains multiple CNN models to detect differences between factory-cut and NFC (non-factory cut) cards. This is the core model training component of the workflow, using FastAI to rapidly create and evaluate different classification models.

## What It Does
The script runs four different classification tests:

1. **Fronts-Only Test**: Classifies card front corners as either factory or NFC
2. **Backs-Only Test**: Classifies card back corners as either factory or NFC
3. **Combined Corners Test**: Combines fronts and backs into a binary classifier
4. **All Categories Test**: Four-way classifier between all corner types

## Requirements
- Images sorted into the proper data folders (created by `03_merge_data.py`)
- CUDA-capable GPU with at least 4GB VRAM
- FastAI and PyTorch properly installed
- Approximately 10GB free disk space for temporary files

## Usage
Basic usage (runs all four tests):
```bash
python 01_run_image_tests.py
```

### Command Line Options
- `--resume`: Continue from previous checkpoints without asking and skip completed models
- `--skip-completed`: Skip tests that have already successfully completed

### Examples
Resume training after an interruption:
```bash
python 01_run_image_tests.py --resume
```

Skip models that are already trained:
```bash
python 01_run_image_tests.py --skip-completed
```

## Data Organization
The script expects your data in this structure:
```
data/
    factory-cut-corners-fronts/
        image1.jpg
        image2.jpg
        ...
    factory-cut-corners-backs/
        ...
    nfc-corners-fronts/
        ...
    nfc-corners-backs/
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
The script produces four model files in the `nfc_models` directory:

- `fronts_only_model.pkl`
- `backs_only_model.pkl`
- `combined_corners_model.pkl`
- `all_categories_model.pkl`

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
