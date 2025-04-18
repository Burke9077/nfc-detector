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
