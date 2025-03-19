# NFC Detector Project

A basic image classification project using:
- FastAI
- OpenCV
- Label Studio
- Jupyter Notebook

## Installation
```bash
pip install fastai opencv-python label-studio
```

## Usage
1. Run the Python script: `python 01_data_exploration.py`
2. Explore the output in your terminal or preferred Python environment.

## Data Organization
Place your images in a folder named `data` with subfolders for each class, for example:
```
data/
    class1/
        img001.jpg
        img002.jpg
    class2/
        img010.jpg
        img011.jpg
```

FastAI will automatically label images based on folder names.
