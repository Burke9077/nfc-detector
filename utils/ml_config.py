"""
Global ML configuration settings for NFC detector models.
This file centralizes machine learning parameters to optimize training
across all models in the project.
"""

import torch
from fastai.vision.all import *

# ========================================
# HARDWARE CONFIGURATION
# ========================================

# RTX 3090 has 24GB VRAM - optimize accordingly
GPU_MEM_GB = 24
IS_HIGH_PERFORMANCE_GPU = True

# Only modify these if you're experiencing CUDA out-of-memory errors
MIXED_PRECISION = True  # Use mixed precision training (fp16) to reduce memory usage and speed up training
CUDNN_BENCHMARK = True  # Set to False if you encounter stability issues

# Optimize CUDA settings for RTX 3090
def optimize_cuda_settings():
    """Apply optimal CUDA settings for high-performance GPUs"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner to find the best algorithm
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK 
        
        # Set to deterministic for reproducibility (but slightly slower)
        # Uncomment if you need 100% reproducible results
        # torch.backends.cudnn.deterministic = True
        # torch.manual_seed(42)
        
        # Prefetch CUDA tensors to minimize transfer delays
        torch.cuda.empty_cache()
        
        # Pin memory for faster host to GPU transfers
        torch.set_float32_matmul_precision('high')
        
        return True
    return False

# ========================================
# MODEL CONFIGURATION
# ========================================

# Default architecture settings
DEFAULT_ARCHITECTURE = 'resnet34'  # Good balance of accuracy and speed
ALTERNATIVE_ARCHITECTURES = {
    'standard': 'resnet34',        # Good default
    'higher_accuracy': 'resnet50', # Better accuracy, slightly slower training
    'maximum_accuracy': 'resnet101', # Best accuracy, slower training
    'efficient': 'efficientnet_b0', # Faster inference, same accuracy as ResNet34
    'mobile': 'mobilenet_v3_small', # Optimized for inference speed, lower accuracy
}

# Architectures to try when doing auto-selection
ARCHITECTURES_TO_TRY = [
    'resnet50',     # Good balance of accuracy and training speed
    'resnet101',    # Higher accuracy, slower training
    'densenet121',  # Excellent for capturing fine details
    'efficientnet_b2', # Efficient performance with good accuracy
]

# Auto-select architecture based on dataset size
def get_recommended_architecture(num_images, num_classes):
    """Return recommended architecture based on dataset characteristics"""
    # For very small datasets, simpler models avoid overfitting
    if num_images < 500:
        return 'resnet18'
    # For larger datasets with many classes, more complex models work better
    elif num_images > 5000 and num_classes > 10:
        return 'resnet50'
    # Default for most scenarios
    else:
        return DEFAULT_ARCHITECTURE

def get_architecture_candidates(gpu_memory_gb=None, num_images=None, num_classes=None, max_candidates=3):
    """
    Get a list of architectures to try based on available GPU memory and dataset characteristics
    
    Args:
        gpu_memory_gb: Available GPU memory in GB (None = use detection)
        num_images: Number of images in dataset 
        num_classes: Number of classes in dataset
        max_candidates: Maximum number of architectures to try
        
    Returns:
        list: Architectures to try, in preferred order
    """
    # Detect available GPU memory if not provided
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif gpu_memory_gb is None:
        gpu_memory_gb = 2  # Assume minimal GPU memory if not available
    
    # Select architectures based on available GPU memory
    if gpu_memory_gb >= 20:  # High-end GPU (like RTX 3090)
        candidates = ['resnet101', 'densenet121', 'efficientnet_b2', 'resnet50']
    elif gpu_memory_gb >= 8:  # Mid-range GPU
        candidates = ['resnet50', 'densenet121', 'efficientnet_b2', 'resnet34']
    else:  # Low-end GPU or CPU
        candidates = ['resnet34', 'efficientnet_b0', 'resnet18']
    
    # Further adjust based on dataset size if provided
    if num_images is not None and num_classes is not None:
        if num_images < 500:  # Very small dataset
            # For small datasets, prefer simpler models to prevent overfitting
            candidates = ['resnet18', 'resnet34', 'efficientnet_b0'] + candidates
        elif num_images > 5000 and num_classes > 10:  # Large multi-class dataset
            # For larger datasets, prioritize more complex models
            candidates = ['resnet101', 'densenet121'] + candidates
    
    # Return unique architectures, keeping original order
    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)
            
    # Limit to max_candidates
    return unique_candidates[:max_candidates]

# ========================================
# TRAINING PARAMETERS
# ========================================

# General training settings
DEFAULT_EPOCHS = 15           # More epochs for 3090 GPU (vs. 8 for a laptop)
DEFAULT_BATCH_SIZE = 64       # Larger batch size for 3090 GPU
DEFAULT_IMAGE_SIZE = 224      # Standard size for most pre-trained models

# Set batch size dynamically based on model and GPU
def get_optimal_batch_size(model_name, image_size=DEFAULT_IMAGE_SIZE):
    """Calculate optimal batch size based on model and available GPU memory"""
    # Base multiplier for RTX 3090 (24GB)
    base = 64
    
    # Adjust batch size based on model complexity
    model_multipliers = {
        'resnet18': 1.5,   # Simpler model = larger batches
        'resnet34': 1.0,   # Base reference
        'resnet50': 0.75,  # More complex
        'resnet101': 0.5,  # Much more complex
        'efficientnet_b0': 1.2,
    }
    
    # Adjust for image size (quadratic impact)
    size_factor = (DEFAULT_IMAGE_SIZE / image_size)**2
    
    # Get multiplier, default to 0.75 if model not in list
    model_factor = model_multipliers.get(model_name, 0.75)
    
    # Calculate and round to nearest multiple of 4 (for GPU efficiency)
    batch_size = int(base * model_factor * size_factor // 4 * 4)
    
    # Ensure minimum reasonable batch size
    return max(16, batch_size)

# Learning rate settings
LR_MIN = 1e-5
LR_MAX = 1e-2
LR_DEFAULT = 1e-3

# Custom fit_one_cycle settings
DEFAULT_FIT_ONE_CYCLE_PARAMS = {
    'pct_start': 0.3,      # Spend 30% of time in warmup
    'div_factor': 25.0,    # LR_max/div_factor = starting LR
    'final_div_factor': 10000.0,  # LR_max/final_div_factor = final LR
}

# ========================================
# EARLY STOPPING & FINE-TUNING
# ========================================

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5   # Stop after 5 epochs without improvement
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement to count

# Progressive resizing settings (train on smaller images first, then fine-tune on larger)
USE_PROGRESSIVE_RESIZING = True
PROGRESSIVE_SIZES = [128, 224, 320]  # Start small, then increase resolution

# Progressive unfreezing settings 
USE_PROGRESSIVE_UNFREEZING = True
DISCRIMINATIVE_LR_MULT = 10  # Lower layers get 10x smaller learning rate

# Fine-tuning stages
FINE_TUNING_STAGES = [
    # Stage 1: Train only the head
    {'epochs': 4, 'freeze_to': -1, 'lr': LR_DEFAULT},
    # Stage 2: Train last 2 layer groups
    {'epochs': 6, 'freeze_to': -2, 'lr': LR_DEFAULT/5},
    # Stage 3: Train all layers with discriminative learning rates
    {'epochs': 8, 'freeze_to': 0, 'lr': LR_DEFAULT/10}
]

# ========================================
# DATA AUGMENTATION
# ========================================

# Default augmentations (FastAI friendly)
DEFAULT_ITEM_TFMS = [
    Resize(DEFAULT_IMAGE_SIZE, method=ResizeMethod.Squish),
]

DEFAULT_BATCH_TFMS = [
    *aug_transforms(
        max_rotate=10.0,       # Rotation in degrees
        max_zoom=1.1,          # Zoom factor
        max_lighting=0.2,      # Lighting adjustment
        max_warp=0.2,          # Perspective warping
        p_affine=0.75,         # Probability of affine transforms
        p_lighting=0.75,       # Probability of lighting transforms
    ),
    Normalize.from_stats(*imagenet_stats)  # Normalize using ImageNet stats
]

# More aggressive augmentations for small datasets
SMALL_DATASET_BATCH_TFMS = [
    *aug_transforms(
        max_rotate=20.0,
        max_zoom=1.2,
        max_lighting=0.3,
        max_warp=0.4,
        p_affine=0.9,
        p_lighting=0.9,
        flip_vert=True,        # Include vertical flips for small datasets
    ),
    Normalize.from_stats(*imagenet_stats)
]

# Define what constitutes a "small" dataset
SMALL_DATASET_THRESHOLD = 1000  # Total images

# ========================================
# MODEL EXPORT & EVALUATION
# ========================================

# Test time augmentation (TTA) settings
DEFAULT_TTA = 4  # Number of augmented predictions to average

# Export settings
EXPORT_WITH_GCP = True  # Use gradient checkpointing for efficient inference

# Evaluation metrics to track
METRICS_TO_TRACK = [
    accuracy,
    error_rate,
    F1Score(average='macro'),  # Good for imbalanced datasets
]

# ========================================
# LOGGING & REPORTING
# ========================================

# CSV logger settings
CSV_LOGGER_FNAME = 'training_history.csv'
SAVE_MODEL_FREQUENCY = 5  # Save every N epochs
REPORT_INTERMEDIATE_RESULTS = True
