import os
import shutil
import cv2
import traceback
from pathlib import Path
from fastai.vision.all import *
from collections import Counter

def setup_temp_dir(base_path):
    """Create temporary directory for test images"""
    temp_dir = Path(base_path) / "temp_test_dir"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    # Create with parents=True to ensure all parent directories exist
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
    class_dir.mkdir(exist_ok=True, parents=True)  # Added parents=True
    
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

def find_optimal_lr(learn, start_lr=1e-7, end_lr=1e-1):
    """Find the optimal learning rate for the model"""
    print("Finding optimal learning rate...")
    lr_finder = learn.lr_find(start_lr=start_lr, end_lr=end_lr)
    suggested_lr = lr_finder.valley
    print(f"Suggested learning rate: {suggested_lr:.6f}")
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

def train_and_save_model(temp_dir, model_save_path, work_path, epochs=15, img_size=(720, 1280), 
                         enhance_edges_prob=0.3, use_tta=True, progressive_resizing=False,
                         resume_from_checkpoint=None, max_rotate=5.0):
    """
    Train a model optimized for detecting subtle differences in card cuts
    
    Args:
        temp_dir: Directory containing the training data
        model_save_path: Path to save the trained model
        work_path: Working directory for all temporary processing
        epochs: Maximum number of training epochs
        img_size: Final image size for training - keep high for subtle edge detection
        enhance_edges_prob: Probability of applying edge enhancement
        use_tta: Whether to use Test Time Augmentation for evaluation
        progressive_resizing: Whether to use progressive resizing (turned off by default for edge detection)
        resume_from_checkpoint: Path to checkpoint to resume training from
        max_rotate: Maximum rotation angle for data augmentation (default: 5.0)
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
    
    # Define image sizes for progressive resizing
    if progressive_resizing:
        # Train in three stages with increasing resolution
        image_sizes = [
            (360, 640),   # Stage 1: Lower resolution
            (540, 960),   # Stage 2: Medium resolution
            img_size      # Stage 3: Full resolution
        ]
        print("Using progressive resizing training strategy (not recommended for subtle edge detection)")
    else:
        image_sizes = [img_size]  # Just use full resolution
        print(f"Training directly at full resolution: {img_size} to preserve edge details")
    
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
    
    # Train with progressive resizing if enabled
    learn = None
    
    # If resuming from checkpoint, find which stage we should start from
    starting_stage = 0
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.exists():
            # Parse stage number from checkpoint filename (best_model_stage1, etc.)
            try:
                stage_str = checkpoint_path.stem
                if 'stage' in stage_str:
                    stage_num = int(stage_str.split('stage')[1])
                    starting_stage = stage_num - 1  # Convert to 0-based index
                    print(f"Resuming from stage {stage_num}")
            except (ValueError, IndexError):
                print("Could not parse stage number from checkpoint, starting from beginning")
    
    # Loop through each training stage (progressive resizing or just one stage)
    for i, size in enumerate(image_sizes):
        # Skip stages we've already completed if resuming
        if i < starting_stage:
            print(f"Skipping stage {i+1} (already completed)")
            continue
            
        print(f"\n--- Training stage {i+1}/{len(image_sizes)}: Resolution {size} ---")
        
        # Create data loaders for this resolution
        dls = ImageDataLoaders.from_folder(
            train_dir, 
            valid_pct=0.2,
            seed=42,  # Fixed seed for reproducibility
            item_tfms=[
                Resize(size, method='pad', pad_mode='zeros'),  # Explicit padding with zeros
                CropPad(size)  # Ensure exact dimensions after resize
            ],
            batch_tfms=tfms,
            num_workers=0,
            bs=8  # Smaller batch size for higher resolution
        )
        
        # Show batch to verify data
        dls.show_batch(max_n=9, figsize=(10,10))
        
        # Print class distribution - FIX: properly count classes
        print("Class distribution in training set:")
        train_labels = [dls.train_ds[i][1] for i in range(len(dls.train_ds))]
        train_counts = Counter(train_labels)
        class_counts = {}
        for label_idx, count in train_counts.items():
            class_name = dls.vocab[label_idx]
            class_counts[class_name] = count
        print(class_counts)
        
        # Create learner or load weights from previous stage
        if learn is None:
            # First stage: create new learner
            learn = vision_learner(dls, resnet50, metrics=[error_rate, accuracy])
            
            # If resuming from checkpoint, load the weights
            if resume_from_checkpoint and i == starting_stage:
                print(f"Loading weights from checkpoint: {resume_from_checkpoint}")
                try:
                    learn.load(resume_from_checkpoint)
                    print("Successfully loaded checkpoint weights")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Will start training from scratch")
            
            # Find optimal learning rate
            opt_lr = find_optimal_lr(learn)
        else:
            # Subsequent stages: create new learner and load weights
            old_learn = learn
            learn = vision_learner(dls, resnet50, metrics=[error_rate, accuracy])
            learn.model.load_state_dict(old_learn.model.state_dict())
            # Find optimal learning rate for fine-tuning
            opt_lr = find_optimal_lr(learn) / 2  # Lower learning rate for fine-tuning
        
        # Add callbacks - save checkpoints to work directory
        checkpoint_path = checkpoint_dir / f'best_model_stage{i+1}'
        callbacks = [
            SaveModelCallback(monitor='valid_loss', fname=str(checkpoint_path)),
            EarlyStoppingCallback(monitor='valid_loss', patience=5)  # Increased patience
        ]
        
        # Training approach depends on the stage
        if i == 0:
            # First stage: regular fine_tune
            print(f"Training stage {i+1} for up to {epochs} epochs with early stopping...")
            learn.fine_tune(epochs, base_lr=opt_lr, cbs=callbacks)
        else:
            # Later stages: shorter fine-tuning with discriminative learning rates
            stage_epochs = max(5, epochs // 2)  # Fewer epochs for later stages
            print(f"Fine-tuning stage {i+1} for up to {stage_epochs} epochs...")
            learn.fine_tune(stage_epochs, base_lr=opt_lr, cbs=callbacks)
    
    # Final evaluation with TTA if requested
    if use_tta:
        print("\nEvaluating with Test Time Augmentation...")
        tta_preds, tta_targets = learn.tta()
        tta_accuracy = (tta_preds.argmax(dim=1) == tta_targets).float().mean()
        print(f"TTA Accuracy: {tta_accuracy:.4f}")
    
    # Save final model to the specified model path (outside work directory)
    learn.export(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Show confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(10,10))
    
    # Show top losses to examine misclassified images
    interp.plot_top_losses(9, figsize=(15,15))
    
    return learn
