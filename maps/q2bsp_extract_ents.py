# extract_ents.py
import os
import glob
import struct

# Configuration
MAPS_DIR = 'maps'
BSP_VERSION = 38  # Standard for Quake II .bsp files
LUMP_INDEX_ENTITIES = 0 # The entity lump is the first one in the directory

def extract_entities_from_bsp(bsp_path):
    """
    Reads a Quake II .bsp file, extracts the entity string, and returns it.
    """
    try:
        with open(bsp_path, 'rb') as f:
            # Read the BSP header: 4-byte magic ('IBSP') and 4-byte version
            magic, version = struct.unpack('<4sI', f.read(8))

            if magic != b'IBSP' or version != BSP_VERSION:
                print(f"  -> Skipping non-Q2 BSP file or wrong version: {bsp_path}")
                return None

            # The entity lump is the first lump entry in the header's lump directory.
            # Each lump entry is 8 bytes (4-byte offset, 4-byte length).
            offset, length = struct.unpack('<II', f.read(8))

            if length <= 1: # Empty or just braces
                print(f"  -> Skipping file with empty entity string: {bsp_path}")
                return ""

            # Seek to the start of the entity data and read it
            f.seek(offset)
            # The data is a null-terminated string, so we read all of it
            # and strip trailing null bytes.
            entity_data = f.read(length)
            
            # Decode from bytes to a string, ignoring potential encoding errors
            # and stripping any trailing null characters.
            entity_string = entity_data.decode('utf-8', errors='ignore').rstrip('\x00')
            
            return entity_string

    except FileNotFoundError:
        print(f"Error: File not found at {bsp_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {bsp_path}: {e}")
        return None

def main():
    """
    Main function to find all .bsp files and process them.
    """
    bsp_files = glob.glob(os.path.join(MAPS_DIR, '*.bsp'))
    
    if not bsp_files:
        print(f"No .bsp files found in the '{MAPS_DIR}/' directory.")
        print("Please ensure this script is run from a directory containing 'maps/'.")
        return

    print(f"Found {len(bsp_files)} .bsp files to process...")
    processed_count = 0

    for bsp_path in bsp_files:
        print(f"Processing '{bsp_path}'...")
        
        entity_string = extract_entities_from_bsp(bsp_path)
        
        if entity_string is not None:
            # Create the output .ent filename by replacing the extension
            ent_path = os.path.splitext(bsp_path)[0] + '.ent'
            
            try:
                with open(ent_path, 'w', encoding='utf-8') as ent_file:
                    ent_file.write(entity_string)
                print(f"  -> Successfully extracted entities to '{ent_path}'")
                processed_count += 1
            except IOError as e:
                print(f"  -> Error writing file '{ent_path}': {e}")
    
    print(f"\nBatch extraction complete. Processed {processed_count} files.")

if __name__ == "__main__":
    main()