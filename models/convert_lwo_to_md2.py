# Multi-Format to OBJ Recursive Converter
#
# Description:
# This script recursively searches a directory for various model files (.lwo, .md5mesh, .ase)
# and attempts to convert them to the Wavefront OBJ format (.obj) using 'assimp'.
# This is a good test to see if the source files can be read correctly.
#
# Author: Gemini
#
# Prerequisites:
# 1. Python 3.x
# 2. Open Asset Import Library (Assimp) command-line tool in your system's PATH.
#
# Usage:
# python convert_models_to_obj.py /path/to/your/models

import os
import subprocess
import argparse
import sys

# Define the source file extensions the script should look for.
SOURCE_EXTENSIONS = ('.lwo', '.md5mesh', '.ase')

def find_source_files(root_dir):
    """
    Recursively finds all specified source model files in the given directory.

    Args:
        root_dir (str): The root directory to start the search from.

    Yields:
        str: The full path to each source file found.
    """
    print(f"[*] Searching for {', '.join(SOURCE_EXTENSIONS)} files in '{root_dir}'...")
    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(SOURCE_EXTENSIONS):
                full_path = os.path.join(dirpath, filename)
                found_files.append(full_path)
    
    if not found_files:
        print(f"[!] No files with supported extensions found.")
    else:
        print(f"[+] Found {len(found_files)} file(s) to convert.")
        
    for file_path in found_files:
        yield file_path

def convert_model_to_obj(source_path):
    """
    Converts a single model file to .obj using the assimp command-line tool.

    Args:
        source_path (str): The full path to the source model file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    # Construct the output path by replacing the extension with .obj
    output_path = os.path.splitext(source_path)[0] + '.obj'
    
    print(f"\n--- Converting '{os.path.basename(source_path)}' to OBJ ---")
    
    # Command with added '-t' flag for triangulation
    command = [
        'assimp',
        'export',
        source_path,
        output_path,
        '-t'  # Pre-transform vertices & Triangulate the mesh
    ]
    
    try:
        # Execute the command
        print(f"[*] Running command: {' '.join(command)}")
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8', # Specify encoding to prevent errors
            errors='replace'   # Replace characters that can't be decoded
        )
        print(f"[+] Successfully converted to '{os.path.basename(output_path)}'")
        if process.stdout.strip():
            # Assimp can be verbose, uncomment for detailed SUCCESS output
            # print(f"    Assimp Output:\n{process.stdout.strip()}")
            pass
        return True
    except FileNotFoundError:
        print("\n[ERROR] 'assimp' command not found.", file=sys.stderr)
        print("Please ensure the Open Asset Import Library (Assimp) is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1) 
    except subprocess.CalledProcessError as e:
        print(f"[!] ERROR: Conversion failed for '{os.path.basename(source_path)}'", file=sys.stderr)
        print(f"    Return code: {e.returncode}", file=sys.stderr)
        
        # --- SUPER-ENHANCED ERROR LOGGING ---
        # Log both stdout and stderr from the failed process, as errors can go to either.
        print("\n    =============== CAPTURED STDOUT (Output) ==============", file=sys.stderr)
        if e.stdout and e.stdout.strip():
            print(e.stdout.strip(), file=sys.stderr)
        else:
            print("    (STDOUT was empty)", file=sys.stderr)
        print("    =======================================================", file=sys.stderr)
        
        print("\n    =============== CAPTURED STDERR (Error) ==============", file=sys.stderr)
        if e.stderr and e.stderr.strip():
            print(e.stderr.strip(), file=sys.stderr)
        else:
            print("    (STDERR was empty)", file=sys.stderr)
        print("    ======================================================", file=sys.stderr)
        
        return False
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}", file=sys.stderr)
        return False

def main():
    """
    Main function to parse arguments and orchestrate the conversion process.
    """
    parser = argparse.ArgumentParser(
        description=f"Recursively convert {', '.join(SOURCE_EXTENSIONS)} files to .obj using assimp."
    )
    parser.add_argument(
        'root_directory',
        type=str,
        help="The root directory containing the model files to convert."
    )
    
    args = parser.parse_args()
    
    root_dir = args.root_directory
    
    if not os.path.isdir(root_dir):
        print(f"[ERROR] The specified directory does not exist: '{root_dir}'", file=sys.stderr)
        sys.exit(1)
        
    success_count = 0
    fail_count = 0
    
    file_list = list(find_source_files(root_dir))
    
    if not file_list:
        return

    for model_file in file_list:
        if convert_model_to_obj(model_file):
            success_count += 1
        else:
            fail_count += 1
            
    print("\n====================")
    print("  Conversion Summary  ")
    print("====================")
    print(f"Total files processed: {success_count + fail_count}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print("====================")

if __name__ == '__main__':
    main()
