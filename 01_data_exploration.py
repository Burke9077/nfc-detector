from fastai.vision.all import *
from collections import Counter  # used to count class occurrences
from fastprogress.fastprogress import progress_bar    # added for progress display
import torch                                      # added for GPU check
import os
import subprocess
import sys
from datetime import datetime
import tempfile, shutil

def run_experiment(subfolders, label):
    path = Path("data")
    temp_work_dir = Path(tempfile.mkdtemp())  # create a unique temp directory
    try:
        # Copy only requested subfolders to the temp directory
        for sf in subfolders:
            shutil.copytree(path/sf, temp_work_dir/sf)
        
        # Create DataLoaders
        dls = ImageDataLoaders.from_folder(
            temp_work_dir,
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize((720,1280)),
            batch_tfms=aug_transforms(),
            num_workers=0
        )
        
        # Filter classes to include only the specified subfolders
        dls.vocab = [cls for cls in dls.vocab if cls in subfolders]
        dls.train_ds = dls.train_ds.filter(lambda x: x[1] in dls.vocab)
        dls.valid_ds = dls.valid_ds.filter(lambda x: x[1] in dls.vocab)
        
        dls.show_batch(max_n=9, figsize=(7,7))
        print("Found classes:", dls.vocab)
        print(f"Total number of classes: {len(dls.vocab)}")
        train_labels = [dls.train_ds[i][1] for i in progress_bar(range(len(dls.train_ds)))]
        train_counts = Counter(train_labels)
        print("Class distribution in training set:")
        print({dls.vocab[cls_idx]: count for cls_idx, count in train_counts.items()})  # Single summary output
        learn = vision_learner(
            dls, 
            resnet34, 
            metrics=accuracy
        ).to_fp16()
        
        # Add callbacks for dynamic training
        callbacks = [
            EarlyStoppingCallback(monitor='valid_loss', patience=3),  # Stop if no improvement for 3 epochs
            SaveModelCallback(monitor='valid_loss', fname=f"best-{label}")  # Save the best model
        ]
        
        # Train the model dynamically
        learn.fine_tune(20, cbs=callbacks)  # Train for up to 20 epochs, but stop early if needed
        
        preds, targs = learn.get_preds()
        avg_confidence = preds.max(dim=1)[0].mean().item()
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix(figsize=(6,6))  # still display the plot
        interp.plot_top_losses(5, figsize=(10,10))
        model_path = Path("model_outputs")
        model_path.mkdir(parents=True, exist_ok=True)
        model_name = f"model-{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
        learn.export(model_path/model_name)
        
        # Save confusion matrix, confidence, and metrics to a text report
        report_name = f"report-{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        with open(model_path/report_name, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Average Prediction Confidence: {avg_confidence:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(interp.confusion_matrix()))
            f.write("\nRecorded Metrics:\n")
            f.write(str(learn.recorder.metrics))
    finally:
        shutil.rmtree(temp_work_dir)  # remove temp directory when done

def main():
    debug_val = os.getenv("DEBUG", "")
    print(f"DEBUG env variable value: '{debug_val}'")
    
    if debug_val.lower() in ("1", "true", "yes"):
        print("DEBUG mode enabled. Printing environment details:")
        print(f"Python version: {sys.version}")
        
        # Show the current python path for clarity (Windows uses 'where', Linux/others use 'which').
        py_path_cmd = "where" if os.name == "nt" else "which"
        print(f"Attempting to run '{py_path_cmd} python':")
        try:
            py_path_result = subprocess.run([py_path_cmd, "python"], capture_output=True, text=True, check=True)
            print(py_path_result.stdout)
        except Exception as e:
            print(f"Failed to locate Python via '{py_path_cmd}':", e)
        
        # Check which torch build is installed: CPU-only or GPU-enabled.
        print("Trying 'pip show torch' to see if a GPU-enabled version is installed:")
        try:
            pip_show = subprocess.run(["pip", "show", "torch"], capture_output=True, text=True, check=True)
            print(pip_show.stdout)
        except Exception as e:
            print("Failed to run 'pip show torch':", e)
        
        print("Attempting to run 'nvidia-smi':")
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            print(result.stdout)
        except FileNotFoundError:
            print("Error: 'nvidia-smi' not found in PATH.")
        except subprocess.CalledProcessError as e:
            print(f"'nvidia-smi' returned a non-zero exit code: {e.returncode}")
            print("Output:", e.output)
        
        # Additional Torch, CUDA checks:
        print(f"Torch version: {torch.__version__}")
        print(f"Torch CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        try:
            test_tensor = torch.rand(1).cuda()
            print("✓ Successfully allocated a test tensor on GPU.")
        except Exception as e:
            print("Error allocating a test tensor on GPU:", e)
    
    # Ensure running on a GPU immediately; exit if not.
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this run. Exiting...")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Set the data folder path.
    path = Path("data")
    
    # Collect image file paths with a progress indicator.
    print("Collecting image file paths...")
    files = list(get_image_files(path))
    for _ in progress_bar(files, total=len(files)):
        pass  # iterate to show progress
    print(f"Collected {len(files)} image files.")
    
    # Monkey-patch get_image_files so from_folder uses our pre-collected list.
    import fastai.data.transforms as fdt
    fdt.get_image_files = lambda p, recurse=True: files
    
    # Load images with a resize transform and standard augmentations.
    dls = ImageDataLoaders.from_folder(
        path, 
        valid_pct=0.2, 
        seed=42, 
        item_tfms=Resize((720,1280)),   # keep images at original resolution
        batch_tfms=aug_transforms(),    # apply standard augmentations
        num_workers=0
    )
    
    # dls.show_batch displays a grid of sample images from the dataset.
    # max_n determines the number of images; figsize sets the display window size.
    dls.show_batch(max_n=9, figsize=(7,7))
    
    # New code: Report found classes and their aggregated distribution in one line.
    print("Found classes:", dls.vocab)
    print(f"Total number of classes: {len(dls.vocab)}")
    train_labels = [dls.train_ds[i][1] for i in progress_bar(range(len(dls.train_ds)))]
    train_counts = Counter(train_labels)
    print("Class distribution in training set:")
    print({dls.vocab[cls_idx]: count for cls_idx, count in train_counts.items()})  # Single summary output
    
    # Build a model using a pre-trained resnet34 for transfer learning.
    # FastAI automatically uses the GPU if available.
    # Optionally enable mixed precision for faster GPU training.
    learn = vision_learner(dls, resnet34, metrics=accuracy).to_fp16()
    
    # Train the model for one epoch.
    learn.fine_tune(5)  # Train for 5 epochs instead of 1
    
    # Verbose output: print training metrics.
    print("Training complete.")
    print("Recorded Metrics:", learn.recorder.metrics)
    
    # Evaluate predictions on the validation set.
    preds, targs = learn.get_preds()
    # Compute average confidence (max probability per prediction).
    avg_confidence = preds.max(dim=1)[0].mean().item()
    print(f"Average prediction confidence: {avg_confidence:.4f}")
    
    # Generate and display a confusion matrix.
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(6,6))
    print("Confusion Matrix:")
    print(interp.confusion_matrix())
    
    # Plot the top losses to inspect misclassified images.
    interp.plot_top_losses(5, figsize=(10,10))
    
    # Create a “model_outputs” folder if it doesn’t already exist.
    model_path = Path("model_outputs")
    model_path.mkdir(parents=True, exist_ok=True)
    # Export the trained model with a timestamped filename into that folder.
    model_name = f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
    learn.export(model_path/model_name)
    
    run_experiment(["factory-cut-corner-fronts", "nfc-corners-fronts"], "fronts")
    run_experiment(["factory-cut-corner-backs",  "nfc-corners-backs"],  "backs")
    run_experiment([
        "factory-cut-corner-fronts", "nfc-corners-fronts",
        "factory-cut-corner-backs",  "nfc-corners-backs"
    ], "both")
    
if __name__ == '__main__':
    main()
