"""
Model training script that utilizes the ML configuration settings.
This file serves as a reference for proper usage of ml_config.py.
"""

import argparse
import os
from pathlib import Path
import json
from fastai.vision.all import *

# Import project utilities
from image_test_utils import (
    train_and_save_model, setup_temp_dir, copy_images_to_class, clean_temp_dir
)
from utils.ml_config import (
    optimize_cuda_settings, DEFAULT_ARCHITECTURE, DEFAULT_IMAGE_SIZE,
    DEFAULT_EPOCHS, DEFAULT_FIT_ONE_CYCLE_PARAMS
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification model using FastAI")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing class folders with images")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the output model")
    parser.add_argument("--architecture", type=str, default=None, 
                       help=f"Model architecture (default: auto-select, fallback to {DEFAULT_ARCHITECTURE})")
    parser.add_argument("--img_size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, 
                       help=f"Image size for training (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, 
                       help=f"Maximum training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--edge_enhance", type=float, default=0.3, 
                       help="Probability of applying edge enhancement (default: 0.3)")
    parser.add_argument("--skip_tta", action="store_true", help="Skip Test Time Augmentation evaluation")
    parser.add_argument("--recalculate_lr", action="store_true", help="Force recalculation of optimal learning rate")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Apply CUDA optimizations
    cuda_available = optimize_cuda_settings()
    if cuda_available:
        print("CUDA optimizations applied")
    else:
        print("CUDA not available, running on CPU")
    
    # Create model directory if it doesn't exist
    model_dir = Path("nfc_models")
    model_dir.mkdir(exist_ok=True)
    
    # Create work directory for temporary files
    work_dir = Path("work_dir")
    work_dir.mkdir(exist_ok=True)
    
    # Set up model paths
    model_path = model_dir / f"{args.model_name}.pkl"
    
    # Check if we're working with an existing dataset or need a temporary one
    if os.path.exists(args.data_dir):
        data_dir = Path(args.data_dir)
    else:
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Train the model
    print(f"Training model '{args.model_name}' using data from {data_dir}")
    learn, metrics = train_and_save_model(
        data_dir,
        model_path,
        work_dir,
        epochs=args.epochs,
        img_size=tuple(args.img_size),
        enhance_edges_prob=args.edge_enhance,
        use_tta=not args.skip_tta,
        recalculate_lr=args.recalculate_lr,
        architecture=args.architecture
    )
    
    print("\nTraining complete!")
    print(f"Final metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()
