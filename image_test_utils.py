import os
import shutil
import cv2
import traceback
import json
from pathlib import Path
from fastai.vision.all import *
from collections import Counter
# Import ML configuration
from utils.ml_config import (DEFAULT_ARCHITECTURE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
                           DEFAULT_BATCH_TFMS, DEFAULT_ITEM_TFMS, SMALL_DATASET_BATCH_TFMS,
                           SMALL_DATASET_THRESHOLD, DEFAULT_TTA, EARLY_STOPPING_PATIENCE,
                           EARLY_STOPPING_MIN_DELTA, get_optimal_batch_size, METRICS_TO_TRACK,
                           get_recommended_architecture, optimize_cuda_settings)

def setup_temp_dir(base_path):
    """Create temporary directory for test images"""
    temp_dir = Path(base_path) / "temp_test_dir"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir

def copy_images_to_class(source_folders, target_dir, class_name):
    """
    Copy images from source folders to a new class directory in target_dir
    
    Args:
        source_folders: List of folder paths containing source images
        target_dir: Base temporary directory
        class_name: Target class name/folder
    """
    # Create class directory
    class_dir = Path(target_dir) / class_name
    class_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy images from each source folder
    for folder in source_folders:
        src_path = Path(folder)
        if not src_path.exists():
            print(f"Warning: Source folder {src_path} does not exist")
            continue
            
        for img_file in src_path.glob("*.jpg"):
            shutil.copy(img_file, class_dir / img_file.name)
        for img_file in src_path.glob("*.png"):
            shutil.copy(img_file, class_dir / img_file.name)
            
    print(f"Copied {len(list(class_dir.glob('*.*')))} images to {class_name}")

def clean_temp_dir(temp_dir):
    """Remove temporary directory after test"""
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"Cleaned up {temp_dir}")

def enhance_edges(img_path, dest_path, kernel_size=3, sigma=1.0):
    """
    Enhance edges in an image to highlight cutting differences
    
    Args:
        img_path: Path to source image
        dest_path: Path to save enhanced image
        kernel_size: Size of Gaussian kernel for edge detection
        sigma: Standard deviation for Gaussian kernel
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Couldn't read image {img_path}")
        return False
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    # Detect edges using Cannyre
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Add edge overlay to original image
    edge_overlay = img.copy()
    edge_overlay[dilated_edges > 0] = [0, 255, 0]  # Highlight edges in green
    
    # Blend original with edge overlay
    result = cv2.addWeighted(img, 0.7, edge_overlay, 0.3, 0)
    
    # Save the result
    cv2.imwrite(str(dest_path), result)
    return True

def preprocess_with_edge_enhancement(source_dir, target_dir, enhance_probability=0.5):
    """
    Preprocess images with edge enhancement for some percentage of images
    
    Args:
        source_dir: Directory containing original images
        target_dir: Directory to save processed images
        enhance_probability: Probability of applying edge enhancement
    """
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Process all image files
    total = 0
    enhanced = 0
    
    for img_file in source_dir.glob("*.jpg"):
        total += 1
        if np.random.random() < enhance_probability:
            if enhance_edges(img_file, target_dir / img_file.name):
                enhanced += 1
        else:
            shutil.copy(img_file, target_dir / img_file.name)
    
    for img_file in source_dir.glob("*.png"):
        total += 1
        if np.random.random() < enhance_probability:
            if enhance_edges(img_file, target_dir / img_file.name):
                enhanced += 1
        else:
            shutil.copy(img_file, target_dir / img_file.name)
    
    print(f"Processed {total} images, enhanced {enhanced} edges")

def find_optimal_lr(learn, model_name=None, recalculate=False, start_lr=1e-7, end_lr=1e-1):
    """
    Find the optimal learning rate for the model
    
    Args:
        learn: fastai Learner object
        model_name: Name of the model for caching LR
        recalculate: Force recalculation even if cached
        start_lr: Start value for learning rate finder
        end_lr: End value for learning rate finder
    """
    # If no model name provided or recalculation requested, always calculate
    if model_name is None or recalculate:
        print("Finding optimal learning rate...")
        lr_finder = learn.lr_find(start_lr=start_lr, end_lr=end_lr)
        suggested_lr = lr_finder.valley
        print(f"Suggested learning rate: {suggested_lr:.6f}")
        
        # Cache the learning rate if model name is provided
        if model_name is not None:
            # Load existing rates
            rates_dict = load_learning_rates()
            # Update with new rate
            rates_dict[model_name] = float(suggested_lr)
            # Save back to file
            save_learning_rates(rates_dict)
            print(f"Cached learning rate for {model_name}")
            
        return suggested_lr
    
    # Try to load cached learning rate
    rates_dict = load_learning_rates()
    if model_name in rates_dict:
        cached_lr = rates_dict[model_name]
        print(f"Using cached learning rate for {model_name}: {cached_lr:.6f}")
        return cached_lr
    
    # If not cached, calculate and cache
    print(f"No cached learning rate found for {model_name}. Calculating...")
    lr_finder = learn.lr_find(start_lr=start_lr, end_lr=end_lr)
    suggested_lr = lr_finder.valley
    print(f"Suggested learning rate: {suggested_lr:.6f}")
    
    # Cache the new learning rate
    rates_dict[model_name] = float(suggested_lr)
    save_learning_rates(rates_dict)
    print(f"Cached learning rate for {model_name}")
    
    return suggested_lr

def ensure_directory_exists(dir_path):
    """Make sure a directory exists, creating it if necessary"""
    path = Path(dir_path)
    try:
        path.mkdir(exist_ok=True, parents=True)
        print(f"Verified directory exists: {path}")
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        print(traceback.format_exc())
        return False

def load_learning_rates(file_path="nfc_models/learning_rates.json"):
    """Load cached learning rates from a JSON file"""
    file_path = Path(file_path)
    if not file_path.exists():
        # Create a new empty dictionary if the file doesn't exist
        save_learning_rates({}, file_path)
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load learning rates: {e}")
        return {}

def save_learning_rates(rates_dict, file_path="nfc_models/learning_rates.json"):
    """Save learning rates to a JSON file"""
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(rates_dict, f, indent=2)
        return True
    except IOError as e:
        print(f"Warning: Failed to save learning rates: {e}")
        return False

def train_and_save_model(temp_dir, model_save_path, work_path, epochs=None, img_size=(720, 1280), 
                         enhance_edges_prob=0.3, use_tta=True, max_rotate=5.0, recalculate_lr=False,
                         resume_from_checkpoint=None, save_model=True, architecture=None,
                         try_multiple_architectures=False, max_architectures=3):
    """
    Train a model optimized for detecting subtle differences in card cuts
    
    Args:
        temp_dir: Directory containing the training data
        model_save_path: Path to save the trained model
        work_path: Working directory for all temporary processing
        epochs: Maximum number of training epochs (None uses DEFAULT_EPOCHS)
        img_size: Image size for training - keep high for subtle edge detection
        enhance_edges_prob: Probability of applying edge enhancement
        use_tta: Whether to use Test Time Augmentation for evaluation
        max_rotate: Maximum rotation angle for data augmentation
        recalculate_lr: Force recalculation of learning rate even if cached
        resume_from_checkpoint: Path to checkpoint to resume training from
        save_model: Whether to save the model (defaults to True)
        architecture: Model architecture to use (None uses auto-selection)
        try_multiple_architectures: Whether to try multiple architectures and select the best
        max_architectures: Maximum number of architectures to try
        
    Returns:
        tuple: (learn, metrics) - the fastai Learner object and a dict of training metrics
    """
    # Apply CUDA optimizations for GPU training
    is_gpu_available = optimize_cuda_settings()
    if is_gpu_available:
        print("GPU optimizations applied for training")
    else:
        print("Warning: GPU not available, training on CPU will be slower")
    
    # Use config defaults if not specified
    if epochs is None:
        epochs = DEFAULT_EPOCHS
        
    # For multiple architecture testing, use fewer epochs for initial testing
    initial_epochs = max(5, epochs // 3) if try_multiple_architectures else epochs
    
    # Verify all paths are absolute to avoid nested path issues
    temp_dir = Path(temp_dir).resolve()
    model_save_path = Path(model_save_path).resolve()
    work_path = Path(work_path).resolve()
    
    print(f"Training with data from: {temp_dir}")
    print(f"Will save model to: {model_save_path}")
    print(f"Using working directory: {work_path}")
    
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    # Verify the source directory exists and is not empty
    if not temp_dir.exists():
        raise RuntimeError(f"Training directory does not exist: {temp_dir}")
    
    class_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {temp_dir}")
    
    # Create parent directory for model if it doesn't exist
    model_parent = model_save_path.parent
    if not ensure_directory_exists(model_parent):
        raise RuntimeError(f"Failed to create model directory: {model_parent}")
    
    # Preprocess images with edge enhancement if requested
    if enhance_edges_prob > 0:
        print(f"Preprocessing images with edge enhancement (prob={enhance_edges_prob})...")
        # Create processed_images directory in work_path
        processed_dir = work_path / "processed_images"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        # Create processed_dir with parents=True to ensure all parent directories exist
        if not ensure_directory_exists(processed_dir):
            raise RuntimeError(f"Failed to create processed directory: {processed_dir}")
        
        for class_dir in class_dirs:
            target_class_dir = processed_dir / class_dir.name
            if not ensure_directory_exists(target_class_dir):
                raise RuntimeError(f"Failed to create target class directory: {target_class_dir}")
            
            preprocess_with_edge_enhancement(class_dir, target_class_dir, enhance_edges_prob)
        
        # Use processed directory for training
        train_dir = processed_dir
    else:
        train_dir = temp_dir
    
    # Double-check training directory is set up correctly
    print(f"Using training data from: {train_dir}")
    if not train_dir.exists():
        raise RuntimeError(f"Training directory does not exist after preprocessing: {train_dir}")
    
    print(f"Training at resolution: {img_size} to preserve edge details")
    
    # Count dataset size to determine which augmentations to use
    total_images = sum(len(list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.png'))) for d in class_dirs)
    num_classes = len(class_dirs)
    
    # Select batch transforms based on dataset size
    if total_images < SMALL_DATASET_THRESHOLD:
        print(f"Small dataset detected ({total_images} < {SMALL_DATASET_THRESHOLD}). Using more aggressive augmentations.")
        batch_tfms = SMALL_DATASET_BATCH_TFMS
    else:
        batch_tfms = DEFAULT_BATCH_TFMS
    
    # Determine the best architecture based on dataset characteristics if not specified
    if architecture is None:
        architecture = get_recommended_architecture(total_images, num_classes)
        print(f"Auto-selected architecture: {architecture}")
    
    # Set up a directory for temporary model checkpoints
    checkpoint_dir = work_path / "model_checkpoints"
    if not ensure_directory_exists(checkpoint_dir):
        raise RuntimeError(f"Failed to create checkpoint directory: {checkpoint_dir}")
    
    # Extract model name from the save path to use for LR caching
    model_name = model_save_path.stem
    
    # Try multiple architectures if requested
    if try_multiple_architectures and not architecture:
        # Get list of architectures to try based on available GPU memory and dataset characteristics
        architectures_to_try = get_architecture_candidates(
            num_images=total_images,
            num_classes=num_classes,
            max_candidates=max_architectures
        )
        
        print(f"Trying multiple architectures: {', '.join(architectures_to_try)}")
        
        # Track results for each architecture
        architecture_results = {}
        best_arch = None
        best_accuracy = 0.0
        best_learner = None
        best_metrics = None
        
        # Train with each architecture
        for arch_name in architectures_to_try:
            print(f"\n{'='*80}\nTraining with architecture: {arch_name}\n{'='*80}")
            
            try:
                # Calculate batch size for this architecture
                batch_size = get_optimal_batch_size(arch_name, img_size[0])
                print(f"Using batch size: {batch_size} for {arch_name}")
                
                # Create data loaders
                dls = ImageDataLoaders.from_folder(
                    train_dir, 
                    valid_pct=0.2,
                    seed=42,  # Fixed seed for reproducibility
                    item_tfms=[
                        Resize(img_size, method='pad', pad_mode='zeros'),
                        CropPad(img_size)
                    ],
                    batch_tfms=batch_tfms,
                    num_workers=0,
                    bs=batch_size
                )
                
                # Create learner with this architecture
                arch = eval(arch_name) if isinstance(arch_name, str) else arch_name
                learn = vision_learner(dls, arch, metrics=METRICS_TO_TRACK)
                learn = learn.to_fp16()
                
                # Show a batch for the first architecture only
                if arch_name == architectures_to_try[0]:
                    print("Sample batch:")
                    dls.show_batch(max_n=4)
                
                # Find optimal learning rate
                opt_lr = find_optimal_lr(learn, model_name=f"{model_name}_{arch_name}", recalculate=recalculate_lr)
                
                # Add callbacks for early stopping
                arch_checkpoint_path = checkpoint_dir / f'arch_{arch_name}'
                callbacks = [
                    SaveModelCallback(monitor='valid_loss', fname=str(arch_checkpoint_path)),
                    EarlyStoppingCallback(monitor='valid_loss', 
                                        patience=EARLY_STOPPING_PATIENCE, 
                                        min_delta=EARLY_STOPPING_MIN_DELTA)
                ]
                
                # Train for fewer epochs in the comparison phase
                print(f"Training {arch_name} for up to {initial_epochs} epochs with early stopping...")
                learn.fine_tune(initial_epochs, base_lr=opt_lr, cbs=callbacks)
                
                # Evaluate architecture
                arch_metrics = {}
                
                # Get training metrics
                if len(learn.recorder.values) > 0:
                    arch_metrics['train_loss'] = learn.recorder.values[-1][0]
                    arch_metrics['valid_loss'] = learn.recorder.values[-1][1]
                    arch_metrics['error_rate'] = learn.recorder.values[-1][2]
                
                # Get TTA metrics if requested
                if use_tta:
                    print(f"\nEvaluating {arch_name} with Test Time Augmentation...")
                    tta_preds, tta_targets = learn.tta()
                    tta_accuracy = (tta_preds.argmax(dim=1) == tta_targets).float().mean()
                    arch_metrics['tta_accuracy'] = float(tta_accuracy)
                    print(f"TTA Accuracy for {arch_name}: {tta_accuracy:.4f}")
                
                # Get standard validation metrics
                valid_metrics = learn.validate()
                if len(valid_metrics) >= 2:
                    arch_metrics['valid_loss'] = float(valid_metrics[0])
                    arch_metrics['valid_error_rate'] = float(valid_metrics[1])
                    arch_metrics['accuracy'] = 1.0 - float(valid_metrics[1])  # Convert error rate to accuracy
                
                # Store results for this architecture
                architecture_results[arch_name] = {
                    'metrics': arch_metrics,
                    'learner': learn
                }
                
                # Update best architecture if this one is better
                current_accuracy = arch_metrics.get('tta_accuracy', arch_metrics.get('accuracy', 0.0))
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_arch = arch_name
                    best_learner = learn
                    best_metrics = arch_metrics.copy()
                    print(f"New best architecture: {arch_name} (Accuracy: {current_accuracy:.4f})")
                
            except Exception as e:
                print(f"Error training with architecture {arch_name}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with the next architecture
        
        # Final training with the best architecture if needed
        if best_arch and best_learner:
            print(f"\n{'='*80}\nSelected best architecture: {best_arch} (Accuracy: {best_accuracy:.4f})\n{'='*80}")
            
            # If we used fewer epochs for testing, do a full training with the best architecture
            if initial_epochs < epochs:
                print(f"Performing full training with {best_arch} for up to {epochs} epochs...")
                
                # Set up callbacks for final training
                final_checkpoint_path = checkpoint_dir / f'best_model'
                callbacks = [
                    SaveModelCallback(monitor='valid_loss', fname=str(final_checkpoint_path)),
                    EarlyStoppingCallback(monitor='valid_loss', 
                                         patience=EARLY_STOPPING_PATIENCE, 
                                         min_delta=EARLY_STOPPING_MIN_DELTA)
                ]
                
                # Get the learning rate for the best architecture
                opt_lr = find_optimal_lr(best_learner, model_name=f"{model_name}_{best_arch}", recalculate=False)
                
                # Continue training with the best architecture for more epochs
                best_learner.fine_tune(epochs - initial_epochs, base_lr=opt_lr, cbs=callbacks)
                
                # Update metrics after final training
                if len(best_learner.recorder.values) > 0:
                    best_metrics['train_loss'] = best_learner.recorder.values[-1][0]
                    best_metrics['valid_loss'] = best_learner.recorder.values[-1][1]
                    best_metrics['error_rate'] = best_learner.recorder.values[-1][2]
                
                # Re-evaluate with TTA if requested
                if use_tta:
                    tta_preds, tta_targets = best_learner.tta()
                    tta_accuracy = (tta_preds.argmax(dim=1) == tta_targets).float().mean()
                    best_metrics['tta_accuracy'] = float(tta_accuracy)
                    print(f"Final TTA Accuracy for {best_arch}: {tta_accuracy:.4f}")
                
                # Get final validation metrics
                valid_metrics = best_learner.validate()
                if len(valid_metrics) >= 2:
                    best_metrics['valid_loss'] = float(valid_metrics[0])
                    best_metrics['valid_error_rate'] = float(valid_metrics[1])
                    best_metrics['accuracy'] = 1.0 - float(valid_metrics[1])
            
            # Use the best model for returning
            learn = best_learner
            metrics = best_metrics
            
            # Add architecture information to metrics
            metrics['architecture'] = best_arch
            metrics['architectures_tried'] = list(architecture_results.keys())
            
            # Add comparison results
            arch_comparison = {}
            for arch_name, result in architecture_results.items():
                arch_accuracy = result['metrics'].get('tta_accuracy', result['metrics'].get('accuracy', 0.0))
                arch_comparison[arch_name] = float(arch_accuracy)
            metrics['architecture_comparison'] = arch_comparison
            
            # Save the best model
            if save_model:
                learn.export(model_save_path)
                print(f"Best model ({best_arch}) saved to {model_save_path}")
                
                # Also save metrics to metadata file
                metadata_path = model_save_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Metrics saved to {metadata_path}")
            
            return learn, metrics
        else:
            raise RuntimeError("Failed to find a working architecture")
    
    # Original code path for single architecture
    # Calculate optimal batch size
    batch_size = get_optimal_batch_size(architecture, img_size[0])
    print(f"Using batch size: {batch_size} for {architecture}")
    
    # Create data loaders at full resolution
    dls = ImageDataLoaders.from_folder(
        train_dir, 
        valid_pct=0.2,
        seed=42,  # Fixed seed for reproducibility
        item_tfms=[
            Resize(img_size, method='pad', pad_mode='zeros'),  # Explicit padding with zeros
            CropPad(img_size)  # Ensure exact dimensions after resize
        ],
        batch_tfms=batch_tfms,
        num_workers=0,
        bs=batch_size
    )
        
    # Print class distribution
    print("Class distribution in training set:")
    train_labels = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
    train_counts = Counter(train_labels)
    class_counts = {}
    for label_idx, count in train_counts.items():
        class_name = dls.vocab[label_idx]
        class_counts[class_name] = count
    print(class_counts)
    
    # Get the model architecture
    arch = eval(architecture) if isinstance(architecture, str) else architecture
    
    # Create learner with metrics from config
    learn = vision_learner(dls, arch, metrics=METRICS_TO_TRACK)
    
    # Enable mixed precision for faster training
    learn = learn.to_fp16()
    
    # Show a batch of images
    print("Sample batch:")
    dls.show_batch(max_n=4)
    
    # If resuming from checkpoint, load the weights
    if resume_from_checkpoint:
        print(f"Loading weights from checkpoint: {resume_from_checkpoint}")
        try:
            learn.load(resume_from_checkpoint)
            print("Successfully loaded checkpoint weights")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Will start training from scratch")
    
    # Find optimal learning rate with caching support
    opt_lr = find_optimal_lr(learn, model_name=model_name, recalculate=recalculate_lr)
    
    # Add callbacks
    checkpoint_path = checkpoint_dir / f'best_model'
    callbacks = [
        SaveModelCallback(monitor='valid_loss', fname=str(checkpoint_path)),
        EarlyStoppingCallback(monitor='valid_loss', 
                             patience=EARLY_STOPPING_PATIENCE, 
                             min_delta=EARLY_STOPPING_MIN_DELTA)
    ]
    
    # Train the model
    print(f"Training for up to {epochs} epochs with early stopping...")
    learn.fine_tune(epochs, base_lr=opt_lr, cbs=callbacks)
    
    # Get final metrics
    metrics = {}
    
    # Get training metrics from the recorder
    recorder = learn.recorder
    if len(recorder.values) > 0:
        metrics['train_loss'] = recorder.values[-1][0]
        metrics['valid_loss'] = recorder.values[-1][1]
        metrics['error_rate'] = recorder.values[-1][2]
    
    # Final evaluation with TTA if requested
    if use_tta:
        print("\nEvaluating with Test Time Augmentation...")
        tta_preds, tta_targets = learn.tta()
        tta_accuracy = (tta_preds.argmax(dim=1) == tta_targets).float().mean()
        metrics['tta_accuracy'] = float(tta_accuracy)
        print(f"TTA Accuracy: {tta_accuracy:.4f}")
    
    # Get standard validation metrics
    valid_metrics = learn.validate()
    if len(valid_metrics) >= 2:
        metrics['valid_loss'] = float(valid_metrics[0])
        metrics['valid_error_rate'] = float(valid_metrics[1])
        metrics['accuracy'] = 1.0 - float(valid_metrics[1])  # Convert error rate to accuracy
    
    # Save final model to the specified model path if requested
    if save_model:
        learn.export(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    # Add timestamp to metrics
    from datetime import datetime
    import time
    timestamp = time.time()
    metrics['timestamp'] = timestamp
    metrics['created'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Get interpretation for metrics but don't display visualizations
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Skip confusion matrix visualization on server
    # interp.plot_confusion_matrix(figsize=(10,10))
    
    # Skip top losses visualization on server
    # Just log that we're on a server without display
    print("Note: Running on server - visualizations disabled")
    
    return learn, metrics
