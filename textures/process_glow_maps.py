# process_glow_maps.py
# ---------------------------------------------------------------------------
# A dedicated utility to find all Quake 4 glow/additive maps (_g, _add),
# convert them to greyscale, and save them with the _glow suffix
# for compatibility with engines like the Quake II Rerelease.
#
# Usage:
# python process_glow_maps.py --config /path/to/your/convert_config.json
# ---------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

from glow_utils import generate_glow_png

def find_glow_maps(source_root: Path) -> list[Path]:
    """Finds all potential glow maps with _g or _add suffixes."""
    print(f"[INFO] Scanning for glow maps in: {source_root}")
    
    glow_maps = []
    suffixes = ("_g", "_add")
    
    for p in source_root.rglob("*"):
        if p.is_file() and p.stem.endswith(suffixes):
            glow_maps.append(p)
            
    print(f"[INFO] Found {len(glow_maps)} potential glow/additive maps.")
    return glow_maps

def process_and_save_map(source_path: Path, source_root: Path, dest_root: Path):
    """
    Opens an image, converts it to greyscale, and saves it to the
    destination with a _glow suffix.
    """
    try:
        # Determine the new name with the _glow suffix
        stem = source_path.stem
        if stem.endswith("_g"):
            new_stem = stem[:-2] + "_glow"
        elif stem.endswith("_add"):
            new_stem = stem[:-4] + "_glow"
        else:
            # This case should not be hit if called correctly
            return

        new_name = new_stem + ".png"
        
        # Determine the relative path to preserve the directory structure
        relative_path = source_path.relative_to(source_root).parent / new_name
        dest_path = dest_root / relative_path

        # Create the glow PNG
        emitted = generate_glow_png(source_path, dest_path)
        if emitted:
            print(f"  [OK] Converted {source_path.name} -> {emitted}")
        else:
            print(f"  [ERROR] Failed to process {source_path.name}: could not emit glow PNG")

    except Exception as e:
        print(f"  [ERROR] Failed to process {source_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Quake 4 glow maps (_g, _add) to Quake II RR format (_glow)."
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the convert_config.json file used by the main converter."
    )
    args = parser.parse_args()

    # Load configuration to get source and destination paths
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        source_root = Path(config["base_root"])
        # Save glow maps within the same destination texture directory
        dest_root = Path(config["dst_base"])
        
        if not source_root.is_dir():
            print(f"[FATAL] Source directory not found: {source_root}")
            return

    except (FileNotFoundError, KeyError) as e:
        print(f"[FATAL] Could not read config file or find required keys ('base_root', 'dst_base'). Error: {e}")
        return

    # Find all potential glow maps
    glow_maps_to_process = find_glow_maps(source_root)
    
    if not glow_maps_to_process:
        print("[INFO] No glow maps to process.")
        return

    # Process each map
    for map_path in glow_maps_to_process:
        process_and_save_map(map_path, source_root, dest_root)
        
    print("\n[DONE] Glow map processing complete.")

if __name__ == "__main__":
    main()
