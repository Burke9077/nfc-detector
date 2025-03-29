import argparse
import shutil
from pathlib import Path
import sys
import datetime
import csv
import os
import hashlib

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

def calculate_checksum(file_path):
    """Calculate MD5 checksum for a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def are_directories_in_sync(src_dir, target_dir):
    """Check if source and target directories have the same files with the same content"""
    # Get all image files in source directory
    src_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    
    # No files to sync
    if not src_files:
        return True
        
    # Check if all source files exist in target with same checksum
    for src_file in src_files:
        target_file = target_dir / src_file.name
        
        # If target file doesn't exist, directories are not in sync
        if not target_file.exists():
            return False
            
        # If checksums don't match, directories are not in sync
        if calculate_checksum(src_file) != calculate_checksum(target_file):
            return False
            
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Merge newly captured data into training data directories')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually moving/copying files')
    
    # Operation mode group (mutually exclusive)
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument('--move-files', action='store_true',
                        help='Move files from source to target (removes from source)')
    operation_group.add_argument('--copy-files', action='store_true',
                        help='Copy files from source to target (preserves source)')
    
    # Conflict resolution group (mutually exclusive)
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument('--overwrite-existing', action='store_true',
                        help='Automatically overwrite existing files without prompting')
    conflict_group.add_argument('--skip-existing', action='store_true',
                        help='Automatically skip existing files without prompting')
    
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
    parser.add_argument('--no-create-missing', action='store_true',
                        help='Do not create missing target directories')
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

def process_directory(src_dir, target_dir, stats, args, log_entries, operation_mode, conflict_mode):
    """Process files in a single directory"""
    # Get all image files
    image_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    
    if not image_files:
        print(f"  No image files found in {src_dir}")
        return []
    
    # Track successfully processed files
    processed_files = []
    
    # Check if directories are in sync
    if are_directories_in_sync(src_dir, target_dir):
        print(f"  Directories are already in sync: {src_dir.name} and {target_dir.name}")
        stats['in_sync'] += 1
        
        # If delete_source is enabled, we should still track synced files for deletion
        if args.delete_source:
            for img_file in image_files:
                # Add to processed files for deletion
                processed_files.append(img_file)
                print(f"  Tracking synced file for deletion: {img_file.name}")
        
        return processed_files
    
    print(f"  Found {len(image_files)} image files")
    
    for img_file in image_files:
        target_file = target_dir / img_file.name
        
        # Check if target file already exists
        if target_file.exists():
            # Check if files are different by comparing checksums
            src_checksum = calculate_checksum(img_file)
            target_checksum = calculate_checksum(target_file)
            
            if src_checksum == target_checksum:
                print(f"  Skipping {img_file.name} (identical file already exists in target)")
                stats['skipped'] += 1
                continue
                
            # Files are different, handle according to conflict mode
            if conflict_mode == 'prompt' and not args.dry_run:
                print(f"  File conflict: {img_file.name}")
                print(f"    Source checksum: {src_checksum}")
                print(f"    Target checksum: {target_checksum}")
                print(f"    (Hint: Use --overwrite-existing or --skip-existing to avoid this prompt)")
                response = input(f"  Overwrite existing file {target_file.name}? (y/n): ").lower()
                if response != 'y':
                    print(f"  Skipping {img_file.name}")
                    stats['skipped'] += 1
                    if args.log:
                        log_entries.append([
                            datetime.datetime.now().isoformat(),
                            'SKIP',
                            str(img_file),
                            str(target_file),
                            'User skipped overwrite'
                        ])
                    continue
            elif conflict_mode == 'skip':
                print(f"  Skipping {img_file.name} (conflict resolution set to skip)")
                stats['skipped'] += 1
                if args.log:
                    log_entries.append([
                        datetime.datetime.now().isoformat(),
                        'SKIP',
                        str(img_file),
                        str(target_file),
                        'Automatically skipped (--skip-existing)'
                    ])
                continue
            elif conflict_mode == 'overwrite':
                print(f"  Will overwrite {img_file.name} (conflict resolution set to overwrite)")
                # Continue with file operation
            
        try:
            if args.dry_run:
                print(f"  Would {operation_mode} {img_file.name} to {target_dir}")
                stats[operation_mode + 'd'] += 1
            elif operation_mode == 'copy':
                shutil.copy2(img_file, target_file)
                print(f"  Copied {img_file.name} to {target_dir}")
                stats['copied'] += 1
                # Add to processed files list if in copy mode and delete_source is True
                if args.delete_source:
                    processed_files.append(img_file)
                if args.log:
                    log_entries.append([
                        datetime.datetime.now().isoformat(),
                        'COPY',
                        str(img_file),
                        str(target_file),
                        'Success'
                    ])
            else:  # move
                shutil.move(img_file, target_file)
                print(f"  Moved {img_file.name} to {target_dir}")
                stats['moved'] += 1
                # Track moved files
                processed_files.append(img_file)
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
    
    # Return the list of successfully processed files for deletion if needed
    return processed_files

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
        
        # Determine if we should create missing target directory
        create_missing = None
        if args.create_missing:
            create_missing = True
        elif args.no_create_missing:
            create_missing = False
        else:
            print("\nTarget directory does not exist.")
            print("(Hint: Use --create-missing or --no-create-missing to skip this prompt)")
            response = input("Create target directory? (y/n): ").lower()
            create_missing = response == 'y'
        
        if create_missing:
            print(f"Creating target directory: {target_path}")
            target_path.mkdir(parents=True)
        else:
            print("Operation cancelled - target directory does not exist")
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
    stats = {'moved': 0, 'copied': 0, 'skipped': 0, 'errors': 0, 'in_sync': 0}
    log_entries = []
    
    # Track all files that were successfully processed
    all_processed_files = []
    
    # Determine operation mode
    operation_mode = None
    if args.move_files:
        operation_mode = 'move'
    elif args.copy_files:
        operation_mode = 'copy'
    
    # Prompt for operation mode if not specified
    if operation_mode is None and not args.dry_run:
        print("\nHow would you like to process the files?")
        print("(Hint: Use --move-files or --copy-files to skip this prompt)")
        print("  1: Copy files (preserve source files)")
        print("  2: Move files (remove from source after copying)")
        while True:
            choice = input("Enter choice (1/2): ").strip()
            if choice == '1':
                operation_mode = 'copy'
                break
            elif choice == '2':
                operation_mode = 'move'
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    elif args.dry_run:
        # For dry run, just set a default
        operation_mode = 'move'
    
    # Determine conflict resolution mode
    conflict_mode = None
    if args.overwrite_existing:
        conflict_mode = 'overwrite'
    elif args.skip_existing:
        conflict_mode = 'skip'
    else:
        conflict_mode = 'prompt'  # Default to prompt for each conflict
    
    # Print operation settings
    print(f"\nStarting data merge from '{source_path}' to '{target_path}'")
    print("-" * 60)
    
    # Show mode with hint if it wasn't explicitly set
    if args.dry_run:
        print("Mode: Dry run - no changes will be made")
    else:
        if args.move_files or args.copy_files:
            print(f"Mode: {operation_mode.capitalize()} files")
        else:
            print(f"Mode: {operation_mode.capitalize()} files (Hint: Use --{operation_mode}-files to set this directly)")
    
    # Show conflict resolution with hint if it wasn't explicitly set
    if args.overwrite_existing or args.skip_existing:
        print(f"Conflict resolution: {conflict_mode.capitalize()} existing files")
    else:
        print(f"Conflict resolution: Prompt for each conflict (Hint: Use --overwrite-existing or --skip-existing to set this directly)")
    
    # Show create_missing with hint
    if args.create_missing:
        print("Create missing directories: Yes")
    elif args.no_create_missing:
        print("Create missing directories: No")
    else:
        print("Create missing directories: Yes (default) (Hint: Use --create-missing or --no-create-missing to set this directly)")
    
    # Show delete source with hint
    if args.delete_source:
        if operation_mode == 'move':
            print("Delete source after completion: Yes")
        else:
            print("Delete source after completion: No (ignored because files are being copied)")
    else:
        print("Delete source after completion: No (Hint: Use --delete-source to enable this)")
    
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
        print("\n(Hint: You can add all options as command line flags to skip all prompts)")
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
        
        # Process files in this directory and collect processed files
        processed_files = process_directory(src_dir, target_dir, stats, args, log_entries, operation_mode, conflict_mode)
        all_processed_files.extend(processed_files)
    
    # Write log if requested
    if args.log and log_entries:
        log_file = Path(f"data_merge_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Source', 'Destination', 'Status'])
            writer.writerows(log_entries)
        print(f"\nLog file created: {log_file}")
    
    # Delete source files if requested and files were successfully processed
    if args.delete_source and not args.dry_run and (stats['errors'] == 0) and all_processed_files:
        deleted_count = 0
        print("\nDeleting processed source files...")
        
        # Delete individual files that were processed
        for file in all_processed_files:
            try:
                if file.exists():  # May already be gone if it was moved
                    file.unlink()
                    deleted_count += 1
                    print(f"  Deleted source file: {file.name}")
            except Exception as e:
                print(f"  Error deleting file {file}: {e}")
        
        print(f"\nDeleted {deleted_count} source files")
        
        # Now try to delete empty directories
        empty_dirs_deleted = 0
        print("\nChecking for empty source directories...")
        
        # First check individual category directories
        for src_dir in source_dirs:
            try:
                if src_dir.exists() and not any(src_dir.iterdir()):
                    shutil.rmtree(src_dir)
                    print(f"  Deleted empty directory: {src_dir}")
                    empty_dirs_deleted += 1
            except Exception as e:
                print(f"  Error checking/deleting directory {src_dir}: {e}")
        
        # Then check if parent directory is empty
        try:
            if source_path.exists() and not any(source_path.iterdir()):
                shutil.rmtree(source_path)
                print(f"  Deleted empty source directory: {source_path}")
                empty_dirs_deleted += 1
        except Exception as e:
            print(f"  Error checking/deleting source directory {source_path}: {e}")
            
        print(f"\nDeleted {empty_dirs_deleted} empty directories")
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    if args.dry_run:
        print("DRY RUN - no files were actually moved or copied")
    
    print(f"Directories in sync: {stats['in_sync']}")
    print(f"Files copied: {stats['copied']}")
    print(f"Files moved: {stats['moved']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    # Show command line example for future runs
    if not args.dry_run and (stats['moved'] + stats['copied']) > 0:
        print("\nFor future runs, you can use a command like:")
        cmd = f"python {os.path.basename(sys.argv[0])}"
        if operation_mode == 'copy':
            cmd += " --copy-files"
        else:
            cmd += " --move-files"
        if conflict_mode == 'overwrite':
            cmd += " --overwrite-existing"
        elif conflict_mode == 'skip':
            cmd += " --skip-existing"
        if args.delete_source:
            cmd += " --delete-source"
        if args.create_missing:
            cmd += " --create-missing"
        if args.log:
            cmd += " --log"
        print(f"  {cmd}")
    
    # Suggest next steps
    if not args.dry_run and (stats['moved'] + stats['copied']) > 0:
        print("\nNext steps:")
        print("  1. Run '01_run_image_tests.py' to train models with the updated dataset")
        print("  2. Use '02_video_stream.py' to test the new models with live video")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
