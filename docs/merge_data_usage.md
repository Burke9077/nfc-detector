# Data Merging Guide

## Overview
The `03_merge_data.py` script helps you merge images from the `newly-captured-data` directory (created during camera captures) into your main `data` directory for model training.

## Usage
Basic usage (will prompt before making changes):
```bash
python 03_merge_data.py
```

### Common Options
- `--dry-run`: Show what would happen without making any changes
- `--copy`: Copy files instead of moving them
- `--create-missing`: Create target directories that don't exist
- `--rename-duplicates`: If a file with the same name exists, rename the new file instead of skipping

### Examples
Test what would happen without making changes:
```bash
python 03_merge_data.py --dry-run
```

Only merge specific categories:
```bash
python 03_merge_data.py --categories factory-cut-corners-fronts nfc-corners-backs
```

Copy files instead of moving them:
```bash
python 03_merge_data.py --copy
```

Comprehensive example - copy files, create any missing directories, and rename duplicates:
```bash
python 03_merge_data.py --copy --create-missing --rename-duplicates --log
```

## Directory Mappings
The script uses these mappings to handle special cases:

| Source Directory | Target Directory |
|------------------|------------------|
| factory-cut-corners-fronts | factory-cut-corners-fronts |
| factory-cut-corners-backs | factory-cut-corners-backs |
| nfc-corners-fronts | nfc-corners-fronts |
| nfc-corners-backs | nfc-corners-backs |
| factory-cut-corners-fronts-square | factory-cut-corners-fronts |
| factory-cut-corners-fronts-wonky | factory-cut-corners-fronts |
| corners-wrong-orientation | corners-wrong-orientation |
| corners-blurry | corners-blurry |

If a source directory has no mapping, the script will either:
1. Create a matching directory in the data folder (if `--create-missing` is used)
2. Skip the directory (if `--create-missing` is not used)

## Workflow Integration
This script fits in the NFC detection workflow as follows:
1. Capture images using `02_video_stream.py`
2. Merge captured data into training set using `03_merge_data.py`
3. Train models on the enhanced dataset using `01_run_image_tests.py`
