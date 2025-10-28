#!/usr/bin/env python3
"""
Directory Structure Dumper
Recursively scans directory structure and outputs to a text file.
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def should_skip(name):
    """Check if directory/file should be skipped."""
    skip_patterns = {
        # Version control
        '.git', '.svn', '.hg',
        # Build directories
        'build', 'builddir', 'dist', 'target',
        '__pycache__', '.pytest_cache',
        # IDE
        '.vscode', '.idea', '.vs',
        # Dependencies
        'node_modules', 'venv', 'env', '.venv',
        # Temp files
        '.tmp', 'tmp', 'temp',
    }
    return name in skip_patterns


def get_dir_tree(root_path, prefix="", max_depth=None, current_depth=0):
    """
    Generate directory tree structure.
    
    Args:
        root_path: Path to scan
        prefix: Current line prefix for tree structure
        max_depth: Maximum recursion depth (None for unlimited)
        current_depth: Current recursion depth
    
    Yields:
        Lines of the directory tree
    """
    if max_depth is not None and current_depth >= max_depth:
        return
    
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        yield f"{prefix}[Permission Denied]"
        return
    
    # Separate directories and files
    dirs = []
    files = []
    
    for entry in entries:
        if should_skip(entry):
            continue
        
        full_path = os.path.join(root_path, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)
    
    # Process directories first, then files
    all_entries = dirs + files
    
    for i, entry in enumerate(all_entries):
        is_last = (i == len(all_entries) - 1)
        full_path = os.path.join(root_path, entry)
        
        # Tree characters
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        if os.path.isdir(full_path):
            yield f"{prefix}{connector}{entry}/"
            # Recurse into directory
            yield from get_dir_tree(
                full_path, 
                prefix + extension,
                max_depth,
                current_depth + 1
            )
        else:
            # Get file size
            try:
                size = os.path.getsize(full_path)
                size_str = format_size(size)
                yield f"{prefix}{connector}{entry} ({size_str})"
            except OSError:
                yield f"{prefix}{connector}{entry} (size unknown)"


def format_size(size):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def count_items(root_path):
    """Count total directories and files."""
    dir_count = 0
    file_count = 0
    
    for root, dirs, files in os.walk(root_path):
        # Filter out skipped directories
        dirs[:] = [d for d in dirs if not should_skip(d)]
        dir_count += len(dirs)
        file_count += len(files)
    
    return dir_count, file_count


def main():
    """Main function to dump directory structure."""
    # Get current directory
    root_path = os.getcwd()
    root_name = os.path.basename(root_path) or root_path
    
    # Output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"directory_structure_{timestamp}.txt"
    
    print(f"Scanning directory: {root_path}")
    print(f"Output file: {output_file}")
    
    # Optional: Set max depth (None for unlimited)
    max_depth = None  # Change to integer like 5 to limit depth
    
    # Count items
    print("Counting items...")
    dir_count, file_count = count_items(root_path)
    
    # Generate tree and write to file
    print("Generating directory tree...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"Directory Structure Report\n")
        f.write(f"Root: {root_path}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Directories: {dir_count}\n")
        f.write(f"Total Files: {file_count}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write tree structure
        f.write(f"{root_name}/\n")
        
        line_count = 0
        for line in get_dir_tree(root_path, max_depth=max_depth):
            f.write(line + "\n")
            line_count += 1
            
            # Progress indicator every 100 lines
            if line_count % 100 == 0:
                print(f"  Processed {line_count} entries...", end='\r')
        
        # Write footer
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"End of Report - Total Lines: {line_count + 1}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Complete! Directory structure saved to: {output_file}")
    print(f"  Total: {dir_count} directories, {file_count} files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)