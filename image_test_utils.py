import os
import shutil
from pathlib import Path
from fastai.vision.all import *

def setup_temp_dir(base_path):
    """Create temporary directory for test images"""
    temp_dir = Path(base_path) / "temp_test_dir"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
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
    class_dir.mkdir(exist_ok=True)
    
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

def train_and_save_model(temp_dir, model_save_path, epochs=5, img_size=224):
    """Train a model on the data in temp_dir and save it"""
    # Create data loaders
    dls = ImageDataLoaders.from_folder(
        temp_dir, 
        valid_pct=0.2,
        item_tfms=Resize(img_size),
        batch_tfms=aug_transforms(),
        num_workers=0
    )
    
    # Show batch to verify data
    dls.show_batch()
    
    # Create learner with ResNet34
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    
    # Train
    learn.fine_tune(epochs)
    
    # Save model
    learn.export(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Show confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    
    return learn
