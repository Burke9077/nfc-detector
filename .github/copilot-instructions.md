# GitHub Copilot Custom Instructions

## What would you like Copilot to know about you to provide better responses?

I am working on an early-stage image classification project using convolutional neural networks (CNNs). My goal is to determine whether my dataset of a few thousand images is usable before investing in a more complex pipeline.

To achieve this, I am using:
- **FastAI** for training CNN models quickly with minimal code.
- **OpenCV** for basic image visualization.
- **Label Studio** for dataset annotation and exploration.
- **Python** as my primary development environment.

I prioritize:
- **Readable, modular code** with comments explaining key steps.
- **Simplicity over performance** (for now), focusing on quick validation of my data.
- **Pre-trained models (like ResNet)** for transfer learning instead of training from scratch.
- **Visual outputs** where possible (e.g., `.show_batch()`, confusion matrices, and OpenCV plots).

Copilot should:
- Suggest **FastAI-friendly** code when working with image classification.
- Prefer **Python-friendly** code (e.g., `matplotlib.pyplot` for plots, `print()` instead of logging).
- Use **OpenCV (`cv2`)** for image display instead of unnecessary libraries.
- Favor **pre-trained CNNs** like `resnet101` unless I explicitly request custom architectures.
- When suggesting dataset preprocessing, use `ImageDataLoaders.from_folder()` and **avoid overly complex manual pipelines**.
- When suggesting image augmentation, use **FastAI transforms**, not raw NumPy operations.
- Add dependencies to requirements.txt when you need them

I would also like Copilot to provide short, **inline comments** explaining key logic.

## How would you like Copilot to respond?

- Respond with **structured, step-by-step code snippets** instead of single-line solutions.
- Provide **explanations** in code comments, but keep responses concise.
- Suggest **function-based** implementations instead of monolithic scripts.
- Offer **Python-friendly** visual outputs when applicable (e.g., `.show_batch()`, OpenCV, or `matplotlib` plots).
- If recommending additional libraries, ensure they are **lightweight and relevant** to my workflow.
- If multiple approaches exist, briefly mention the alternatives but default to **FastAI’s high-level API**.

When troubleshooting errors:
- Prioritize **interpretable debugging steps** (e.g., checking `dls.valid.show_batch()` for bad labels).
- Suggest **practical fixes** before proposing deep-dive solutions.
- Avoid over-complicating things with unnecessary optimizations.

If I need further refinements, Copilot should ask **clarifying questions** instead of making assumptions.

## Project File Structure

### File Paths and Project Organization

The project follows this file structure:
```
./
├── 01_run_image_tests.py
├── 02_video_stream.py
├── 03_merge_data.py
├── data/                                 # Image data directories
│   ├── corners-blurry/
│   ├── corners-wrong-orientation/
│   ├── factory-cut-corners-backs/
│   ├── factory-cut-corners-fronts/
│   ├── factory-cut-sides-backs-die-cut/
│   ├── factory-cut-sides-fronts-die-cut/
│   ├── nfc-corners-backs/
│   ├── nfc-corners-fronts/
│   ├── nfc-sides-backs/
│   ├── nfc-sides-fronts/
│   ├── sides-blurry/
│   └── sides-wrong-orientation/
├── docs/
│   ├── 01_run_image_tests_usage.md
│   ├── 02_video_stream_usage.md
│   ├── 03_merge_data_usage.md
│   └── COLLECTING_GUIDE.md
├── nfc_models/
│   ├── [various model files and metadata]
│   └── learning_rates.json
├── utils/
│   ├── dataset_utils.py
│   ├── directory_utils.py
│   ├── image_utils.py
│   ├── __init__.py
│   ├── model_discovery_utils.py
│   ├── model_metadata_utils.py
│   ├── model_performance_utils.py
│   ├── models/
│   │   ├── m01_orientation_corners_classification.py
│   │   ├── m02_orientation_sides_classification.py
│   │   ├── m03_focus_corners_classification.py
│   │   ├── m04_focus_sides_classification.py
│   │   ├── m10_corner_front_back_classification.py
│   │   ├── m11_side_front_back_classification.py
│   │   ├── m30_corner_front_factory_vs_nfc_classification.py
│   │   ├── m31_corner_back_factory_vs_nfc_classification.py
│   │   ├── m32_side_front_factory_vs_nfc_classification.py
│   │   └── m33_side_back_factory_vs_nfc_classification.py
│   ├── model_utils.py
│   └── test_utils.py
├── README.md
└── requirements.txt
```

When suggesting code, please respect this structure. If you need a file that isn't visible in the current context:
- Ask me to provide it rather than creating stub implementations
- Pause generation and wait for me to share the file content
- Resume after I've provided the necessary files

### Core Utilities

#### dataset_utils.py (./utils/dataset_utils.py)
This module handles dataset preparation and management for training image classification models. When working with:
- **Data organization**: Use functions like `copy_images_to_class()` and `balanced_copy_images()` to structure data for FastAI
- **Class balancing**: Use `prepare_balanced_dataset()` to create balanced training sets and prevent bias
- **Multi-source folders**: The utilities support combining images from multiple source directories into one organized dataset
- **Visualization**: The module includes tools to visualize and validate class distribution

Include this file when working on data preparation tasks, addressing class imbalance issues, or diagnosing dataset problems.

#### directory_utils.py (./utils/directory_utils.py)
This module provides utilities for directory and file management operations:
- **Directory verification**: Use `verify_directories()` and `ensure_directory_exists()` to check/create required directories
- **Working directory management**: Use `clean_work_dir()`, `setup_temp_dir()`, and `clean_temp_dir()` to handle temporary directories
- **Checkpoint handling**: Use `find_latest_checkpoint()` to locate model checkpoints for resuming training
- **Test state tracking**: Use `is_test_completed()` to check if models have been successfully generated

Include this file when working on directory structure management, file operations, or managing model checkpoints.

#### image_utils.py (./utils/image_utils.py)
This module provides image processing utilities for analyzing and enhancing card images:
- **Edge enhancement**: Use `enhance_edges()` to highlight cutting differences in card images
- **Preprocessing pipelines**: Use `preprocess_with_edge_enhancement()` to prepare images for model training
- **Visual analysis**: Functions for creating visualizations to help identify subtle differences between genuine and counterfeit cards
- **Batch processing**: Utilities for processing multiple images with randomized enhancements

Include this file when working on image preprocessing, feature enhancement, or creating visual representations of card differences.

#### model_discovery_utils.py (./utils/model_discovery_utils.py)
This module provides utilities for managing and organizing multiple model variants:
- **Model discovery**: Use `discover_models()` to dynamically find all available model implementations
- **Model categorization**: Use `determine_model_category()` to organize models by their intended purpose
- **Model listing**: Use `list_models()` to generate formatted tables of available models with their status
- **CLI integration**: Functions to support command-line model selection and execution
- **Performance tracking**: Integration with metrics display and model metadata

Include this file when working with multiple model variants, organizing model test functions, or building interfaces for model selection.

#### model_metadata_utils.py (./utils/model_metadata_utils.py)
This module provides utilities for tracking, comparing, and managing model performance metrics:
- **Metadata persistence**: Use `save_model_metadata()` and `load_model_metadata()` to store and retrieve model performance metrics
- **Model comparison**: Use `is_model_better()` to determine if a new model outperforms an existing one
- **Performance reporting**: Use `format_metadata_for_display()` to create human-readable metric summaries
- **Learning rate management**: Use `load_learning_rates()` and `save_learning_rates()` to cache optimal learning rates for faster retraining

Include this file when working on model evaluation, performance tracking, automated model selection, or when implementing checkpointing functionality.

#### model_performance_utils.py (./utils/model_performance_utils.py)
This module provides utilities for evaluating model quality and making comparison decisions:
- **Accuracy assessment**: Use `get_best_accuracy_from_metadata()` to extract accuracy from model metadata
- **Quality indicators**: Use `get_accuracy_emoji()` and `get_model_quality_category()` for user-friendly quality reporting
- **Performance comparison**: Use `compare_model_performance()` to determine if a new model should replace an existing one
- **Retraining decisions**: Use `should_rerun_model_by_quality()` to decide if models need retraining based on quality thresholds
- **Formatted reporting**: Use `format_metrics_with_emoji()` and `format_performance_change()` for CLI-friendly output

Include this file when implementing model evaluation pipelines, building automated testing frameworks, or creating user interfaces that display model performance.

#### model_utils.py (./utils/model_utils.py)
This module provides comprehensive utilities for CNN model training and evaluation:
- **GPU resource management**: Use `check_gpu_memory()` to verify GPU availability and memory status
- **Learning rate optimization**: Use `find_optimal_lr()` for automated learning rate discovery with caching
- **Model training workflow**: Use `train_and_save_model()` for the complete training pipeline with edge enhancement
- **Augmentation strategies**: The module includes specialized augmentation parameters optimized for edge detection
- **Checkpoint management**: Built-in support for saving/loading checkpoints during training
- **Test-time augmentation**: Functions to evaluate models with TTA for improved accuracy
- **Model metrics collection**: Automated collection and reporting of training and validation metrics

Include this file when implementing model training workflows, evaluating models, or diagnosing training performance issues.

#### test_utils.py (./utils/test_utils.py)
This module provides utilities for standardizing model testing workflows:
- **Test execution**: Use `run_classification_test()` to run end-to-end tests with consistent parameters
- **Performance tracking**: Use `track_model_performance()` to compare model runs and track improvements
- **Intelligent model saving**: Built-in logic to save models only when they outperform previous versions
- **Dataset preparation**: Automatic integration with balanced dataset preparation
- **Environment integration**: Support for dynamic configuration through environment variables

Include this file when implementing automated testing pipelines, comparing model performance across runs, or creating reproducible testing workflows.

## Maintenance Instructions

### Keeping Documentation in Sync

When making significant changes to the codebase:
- **Adding new functions**: Update the corresponding module section in this file
- **Renaming functions**: Update any references to the old function name in this file
- **Changing function parameters**: Update any examples or descriptions that reference the parameters
- **Removing functions**: Remove mentions from this document to avoid confusion
- **Adding new modules**: Create a new section in the "Project File Structure" area with relevant details

This ensures that Copilot has accurate information about the codebase structure when providing assistance.

### Working with Existing Files

When making changes to the codebase:
- If you need access to a file that isn't visible, ask for it explicitly
- Don't create placeholder implementations for files that should already exist
- If suggesting new files, explicitly indicate where they should be placed in the existing structure
- Always specify relative paths when referring to existing files (e.g., ./utils/model_utils.py)
