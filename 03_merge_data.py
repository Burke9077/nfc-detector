import argparse
import shutil
from pathlib import Path
import sys
import datetime
import csv
import os

# Define mappings between source folders and target folders
# This handles cases where folder names might not match exactly
DIRECTORY_MAPPINGS = {
    # Direct mappings
    "factory-cut-corners-fronts": "factory-cut-corners-fronts",
    "factory-cut-corners-backs": "factory-cut-corners-backs",
    "nfc-corners-fronts": "nfc-corners-fronts",
    "nfc-corners-backs": "nfc-corners-backs",
    
    # Special quality attributes folders map to their base folders
    "factory-cut-corners-fronts-square": "factory-cut-corners-fronts",
    "factory-cut-corners-backs-square": "factory-cut-corners-backs",
    "factory-cut-corners-fronts-wonky": "factory-cut-corners-fronts", 
    "factory-cut-corners-backs-wonky": "factory-cut-corners-backs",
    
    # Special cases that go to their own folders
    "corners-blurry": "corners-blurry",
    "corners-wrong-orientation": "corners-wrong-orientation",
    "sides-blurry": "sides-blurry",
    "sides-wrong-orientation": "sides-wrong-orientation",
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Merge newly captured data into training data directories')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually moving/copying files')
    parser.add_argument('--copy', action='store_true', 
                        help='Copy files instead of moving them (preserves source files)')
    parser.add_argument('--delete-source', action='store_true',
                        help='Delete source folders and parent directory after successful operation')
    parser.add_argument('--categories', nargs='+', 
                        help='Specific categories to merge (default: all)')
    parser.add_argument('--source', default='newly-captured-data', 
                        help='Source directory with newly captured data')
    parser.add_argument('--target', default='data', 
                        help='Target directory for training data')
    parser.add_argument('--create-missing', action='store_true', 
                        help='Create missing target directories')
    parser.add_argument('--rename-duplicates', action='store_true', 
                        help='Rename files that already exist in the target')
    parser.add_argument('--log', action='store_true', 
                        help='Create a CSV log of operations')
    return parser.parse_args()

def get_target_directory(src_dir_name, target_path, args):
    """Determine the target directory for a given source directory"""
    # Check if we have a mapping for this directory
    if src_dir_name in DIRECTORY_MAPPINGS:
        mapped_name = DIRECTORY_MAPPINGS[src_dir_name]
        target_dir = target_path / mapped_name
        
        # Check if target exists or should be created
        if target_dir.exists():
            return target_dir
        elif args.create_missing:
            # Create the directory
            target_dir.mkdir(exist_ok=True, parents=True)
            print(f"Created target directory: {target_dir}")
            return target_dir
    
    # Try direct name match if no mapping found
    target_dir = target_path / src_dir_name
    if target_dir.exists():
        return target_dir
    elif args.create_missing:
        # Create the directory with the same name
        target_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created target directory: {target_dir}")
        return target_dir
    
    # No suitable target
    return None

def process_directory(src_dir, target_dir, stats, args, log_entries):
    """Process files in a single directory"""
    # Get all image files
    image_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    
    if not image_files:
        print(f"  No image files found in {src_dir}")
        return
    
    print(f"  Found {len(image_files)} image files")
    
    for img_file in image_files:
        target_file = target_dir / img_file.name
        
        # Check if target file already exists
        if target_file.exists():
            if args.rename_duplicates:
                # Rename the file with a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{img_file.stem}_{timestamp}{img_file.suffix}"
                target_file = target_dir / new_name
                print(f"  Renaming duplicate {img_file.name} to {new_name}")
                stats['renamed'] += 1
            else:
                print(f"  Skipping {img_file.name} (already exists in target)")
                stats['skipped'] += 1
                if args.log:
                    log_entries.append([
                        datetime.datetime.now().isoformat(),
                        'SKIP',
                        str(img_file),
                        str(target_file),
                        'File already exists'
                    ])
                continue
        
        try:
            if args.dry_run:
                action = 'copy' if args.copy else 'move'
                print(f"  Would {action} {img_file.name} to {target_dir}")
                stats['moved' if not args.copy else 'copied'] += 1
            elif args.copy:
                shutil.copy2(img_file, target_file)
                print(f"  Copied {img_file.name} to {target_dir}")
                stats['copied'] += 1
                if args.log:
                    log_entries.append([
                        datetime.datetime.now().isoformat(),
                        'COPY',
                        str(img_file),
                        str(target_file),
                        'Success'
                    ])
            else:
                shutil.move(img_file, target_file)
                print(f"  Moved {img_file.name} to {target_dir}")
                stats['moved'] += 1
                if args.log:
                    log_entries.append([
                        datetime.datetime.now().isoformat(),
                        'MOVE',
                        str(img_file),
                        str(target_file),
                        'Success'
                    ])
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
            stats['errors'] += 1
            if args.log:
                log_entries.append([
                    datetime.datetime.now().isoformat(),
                    'ERROR',
                    str(img_file),
                    str(target_file),
                    str(e)
                ])

def main():
    """Main function for data merging"""
    args = parse_arguments()
    
    # Setup paths
    source_path = Path(args.source).resolve()
    target_path = Path(args.target).resolve()
    
    # Verify paths exist
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist")
        return 1
        
    if not target_path.exists():
        print(f"Error: Target directory '{target_path}' does not exist")
        if args.create_missing:
            print(f"Creating target directory: {target_path}")
            target_path.mkdir(parents=True)
        else:
            return 1
    
    # Print help message if no source directories exist
    source_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not source_dirs:
        print(f"No source directories found in '{source_path}'")
        print(f"Run '{sys.argv[0]} --help' for usage information")
        return 1
    
    # Filter categories if specified
    if args.categories:
        source_dirs = [d for d in source_dirs if d.name in args.categories]
    
    if not source_dirs:
        print(f"No matching directories found in {source_path}")
        return 0
    
    # Initialize stats and log
    stats = {'moved': 0, 'copied': 0, 'skipped': 0, 'errors': 0, 'renamed': 0}
    log_entries = []
    
    print(f"\nStarting data merge from '{source_path}' to '{target_path}'")
    print("-" * 60)
    print(f"Mode: {'Dry run - no changes will be made' if args.dry_run else 'Copy files' if args.copy else 'Move files'}")
    print(f"Create missing directories: {'Yes' if args.create_missing else 'No'}")
    print(f"Rename duplicates: {'Yes' if args.rename_duplicates else 'No'}")
    print(f"Delete source after completion: {'Yes' if args.delete_source else 'No'}")
    print("-" * 60)
    
    # First pass - count files and check for issues
    total_files = 0
    print("Scanning source directories...")
    for src_dir in source_dirs:
        image_count = len(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
        total_files += image_count
        target_dir = get_target_directory(src_dir.name, target_path, args)
        if target_dir:
            print(f"  {src_dir.name} -> {target_dir.name} ({image_count} files)")
        else:
            print(f"  {src_dir.name} -> No target directory found ({image_count} files)")
    
    print(f"\nFound {total_files} total files to process across {len(source_dirs)} directories")
    
    # Confirm with user before proceeding
    if not args.dry_run and total_files > 0:
        response = input("\nProceed with merging data? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled by user")
            return 0
    
    # Second pass - process files
    print("\nProcessing directories...")
    for src_dir in source_dirs:
        # Get target directory using our mapping logic
        target_dir = get_target_directory(src_dir.name, target_path, args)
        
        if target_dir is None:
            print(f"Skipping {src_dir.name} - no suitable target directory found")
            if args.log:
                log_entries.append([
                    datetime.datetime.now().isoformat(),
                    'SKIP_DIR',
                    str(src_dir),
                    '',
                    'No target directory'
                ])
            continue
            
        print(f"Processing {src_dir.name} -> {target_dir.name}")
        
        # Process files in this directory
        process_directory(src_dir, target_dir, stats, args, log_entries)
    
    # Write log if requested
    if args.log and log_entries:
        log_file = Path(f"data_merge_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Source', 'Destination', 'Status'])
            writer.writerows(log_entries)
        print(f"\nLog file created: {log_file}")
    
    # Delete source if requested and not in dry-run mode
    if args.delete_source and not args.dry_run and (stats['errors'] == 0):
        # Only proceed if we had successful operations
        if (stats['moved'] + stats['copied']) > 0:
            print("\nDeleting source directories...")
            
            # First delete individual category folders
            for src_dir in source_dirs:
                try:
                    if src_dir.exists():  # Check if it still exists (might have been moved already)
                        if not list(src_dir.glob("*")):  # Only delete if empty
                            shutil.rmtree(src_dir)
                            print(f"  Deleted: {src_dir}")
                        else:
                            print(f"  Skipped deletion: {src_dir} (not empty)")
                except Exception as e:
                    print(f"  Error deleting {src_dir}: {e}")
            
            # Then try to delete parent directory if empty
            try:
                if source_path.exists() and not list(source_path.glob("*")):
                    shutil.rmtree(source_path)
                    print(f"Deleted source directory: {source_path}")
            except Exception as e:
                print(f"Error deleting source directory {source_path}: {e}")
        else:
            print("\nNo files were processed, skipping source directory deletion")
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    if args.dry_run:
        print("DRY RUN - no files were actually moved or copied")
    
    action = 'Would copy' if args.dry_run and args.copy else 'Would move' if args.dry_run else 'Copied' if args.copy else 'Moved'
    print(f"{action}: {stats['moved'] + stats['copied']} files")
    print(f"Files renamed: {stats['renamed']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    # Suggest next steps
    if not args.dry_run and (stats['moved'] + stats['copied']) > 0:
        print("\nNext steps:")
        print("  1. Run '01_run_image_tests.py' to train models with the updated dataset")
        print("  2. Use '02_video_stream.py' to test the new models with live video")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
