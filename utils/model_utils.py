"""
Model Utilities for NFC Detector

This module provides comprehensive utilities for CNN model training and evaluation, including:
- GPU resource checking and memory management
- Learning rate optimization with caching for faster retraining
- Model training with edge-enhancement preprocessing
- Test-time augmentation for improved accuracy
- Advanced data augmentation strategies optimized for edge detection
- Checkpoint management and training resumption
- Model evaluation and metrics collection

These utilities support the image classification pipeline by handling the core
model training workflow while optimizing for subtle edge differences in card images.
"""

import torch
import torch.cuda as cuda
import matplotlib.pyplot as plt
import shutil
import os
import traceback
from pathlib import Path
from fastai.vision.all import *
from collections import Counter

from utils.directory_utils import ensure_directory_exists
from utils.model_metadata_utils import load_learning_rates, save_learning_rates
from utils.image_utils import preprocess_with_edge_enhancement

def check_gpu_memory():
    """
    Check if GPU is available and print memory info.
    No GUI elements - only console output.
    Warns if less than 8GB GPU memory is available.
    """
    if not cuda.is_available():
        print("CUDA not available, using CPU only")
        return None
    
    # Get device information
    device = torch.device("cuda")
    gpu_name = cuda.get_device_name(0)
    
    # Get memory information
    total_memory = round(cuda.get_device_properties(0).total_memory / 1e9, 2)  # Convert to GB
    allocated_memory = round(cuda.memory_allocated(0) / 1e9, 2)  # Convert to GB
    free_memory = round(total_memory - allocated_memory, 2)
    
    # Print GPU information to console
    print(f"\nGPU Information:")
    print(f"  Device: {gpu_name}")
    print(f"  Total Memory: {total_memory:.2f} GB")
    print(f"  Currently Allocated: {allocated_memory:.2f} GB")
    print(f"  Free Memory: {free_memory:.2f} GB")
    
    # Check if GPU has at least 8GB available
    if total_memory < 8.0:
        print(f"  WARNING: GPU has less than 8GB total memory ({total_memory:.2f} GB). Performance may be impacted.")
    elif free_memory < 8.0:
        print(f"  WARNING: Less than 8GB GPU memory available ({free_memory:.2f} GB). Performance may be impacted.")
    else:
        print(f"  GPU memory is sufficient for training.")
    
    return device

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

def train_and_save_model(temp_dir, model_save_path, work_path, epochs=60, img_size=(720, 1280), 
                         enhance_edges_prob=0.3, use_tta=True, max_rotate=5.0, recalculate_lr=False,
                         resume_from_checkpoint=None, save_model=True):
    """
    Train a model optimized for detecting subtle differences in card cuts
    
    Args:
        temp_dir: Directory containing the training data
        model_save_path: Path to save the trained model
        work_path: Working directory for all temporary processing
        epochs: Maximum number of training epochs
        img_size: Image size for training - keep high for subtle edge detection
        enhance_edges_prob: Probability of applying edge enhancement
        use_tta: Whether to use Test Time Augmentation for evaluation
        max_rotate: Maximum rotation angle for data augmentation (default: 5.0)
        recalculate_lr: Force recalculation of learning rate even if cached
        resume_from_checkpoint: Path to checkpoint to resume training from
        save_model: Whether to save the model (defaults to True)
        
    Returns:
        tuple: (learn, metrics) - the fastai Learner object and a dict of training metrics
    """
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
    
    print(f"Training at full resolution: {img_size} to preserve edge details")
    
    # Use augmentation strategy with custom rotation limit
    tfms = [
        *aug_transforms(
            max_rotate=max_rotate,  # Use the parameter instead of hardcoded value
            max_zoom=1.02,      # Minimal zoom to keep edges intact
            max_warp=0,         # No warping to avoid distorting edges
            max_lighting=0.1,   # Minimal lighting changes
            p_affine=0.5,       # Apply affine transforms with 50% probability
            p_lighting=0.7      # Apply lighting transforms with 70% probability
        ),
        Normalize.from_stats(*imagenet_stats)  # Standard normalization
    ]
    
    # Set up a directory for temporary model checkpoints
    checkpoint_dir = work_path / "model_checkpoints"
    if not ensure_directory_exists(checkpoint_dir):
        raise RuntimeError(f"Failed to create checkpoint directory: {checkpoint_dir}")
    
    # Create data loaders at full resolution
    dls = ImageDataLoaders.from_folder(
        train_dir, 
        valid_pct=0.2,
        seed=42,  # Fixed seed for reproducibility
        item_tfms=[
            Resize(img_size, method='pad', pad_mode='zeros'),  # Explicit padding with zeros
            CropPad(img_size)  # Ensure exact dimensions after resize
        ],
        batch_tfms=tfms,
        num_workers=0,
        bs=8  # Adjust batch size according to your GPU memory
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
    
    # Extract model name from the save path to use for LR caching
    model_name = model_save_path.stem
    
    # Create learner
    learn = vision_learner(dls, resnet101, metrics=[error_rate, accuracy])
    
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
        EarlyStoppingCallback(monitor='valid_loss', patience=15)
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
        
        # Add actual epochs trained (useful for comparing models with early stopping)
        metrics['epochs'] = len(recorder.values)
    
    # Final evaluation with TTA if requested
    if use_tta:
        print("\nEvaluating with Test Time Augmentation...")
        tta_preds, tta_targets = learn.tta()
        tta_accuracy = (tta_preds.argmax(dim=1) == tta_targets).float().mean()
        metrics['tta_accuracy'] = float(tta_accuracy)
        print(f"TTA Accuracy: {tta_accuracy:.4f}")
        
        # Add confidence metrics from TTA predictions
        # Confidence is the mean probability assigned to the predicted class
        confidence = tta_preds.max(dim=1)[0].mean().item()
        metrics['confidence'] = float(confidence)
        print(f"Mean prediction confidence: {confidence:.4f}")
    
    # Get standard validation metrics
    valid_metrics = learn.validate()
    if len(valid_metrics) >= 2:
        metrics['valid_loss'] = float(valid_metrics[0])
        metrics['valid_error_rate'] = float(valid_metrics[1])
        metrics['accuracy'] = 1.0 - float(valid_metrics[1])  # Convert error rate to accuracy
    
    # Add model size if already saved
    if save_model:
        learn.export(model_save_path)
        print(f"Model saved to {model_save_path}")
        try:
            metrics['model_size_bytes'] = Path(model_save_path).stat().st_size
        except Exception:
            pass
    
    # Get interpretation for metrics but don't display visualizations
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Skip confusion matrix visualization on server
    # interp.plot_confusion_matrix(figsize=(10,10))
    
    # Skip top losses visualization on server
    # Just log that we're on a server without display
    print("Note: Running on server - visualizations disabled")
    
    return learn, metrics
