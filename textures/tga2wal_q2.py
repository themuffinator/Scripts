#!/usr/bin/env python3
# tga2wal_final.py - recursively convert TGA -> Quake II WAL format.
# Scales input by 50% and dynamically uses the last palette color for transparency.

import struct
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

# ============================== WAL CONSTANTS ==============================

# Quake 2 Surface Flags (Used by the engine/compiler for behavior)
# Definitions based on shared.h (Quake 2/KEX)
SURF_LIGHT = 0x00000001
SURF_WARP = 0x00000008
SURF_TRANS33 = 0x00000010

# Quake 2 Content Flags (Used by the engine/compiler for collision/type)
# Definitions based on shared.h (Quake 2/KEX)
CONTENTS_SOLID = 0x00000001
CONTENTS_WINDOW = 0x00000002
CONTENTS_WATER = 0x00000020

# Light value for Q2 light textures
Q2_LIGHT_VALUE = 1000

# Default Quake 2 Palette (Minimal version: last color is transparency key (0, 0, 0))
# This is a highly truncated, representative 768-byte list (256 colors * 3 bytes)
# In a full conversion tool, this should be the full 768-byte array from Q2.
# For simplicity and functionality demonstration, we use a placeholder array 
# that correctly defines the black transparency key at the end.
# If a full palette is not available, using a placeholder array of 768 bytes
# with a correctly set final entry is required for Pillow initialization.
# Note: For real-world use, replace this with the actual 768 bytes of the Q2 palette.
DEFAULT_Q2_PALETTE_DATA = [
    # Placeholder for the first 255 colors (255 * 3 = 765 bytes)
    # This must be 768 bytes long in total.
    *([0, 0, 0] * 255),
    # The final color (index 255) must be the transparency key, which is usually black (0, 0, 0)
    0, 0, 0 
]

# ============================== WAL FILE STRUCTURE ==============================

def load_palette_and_key_color(colormap_path: Path) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """
    Loads the palette from the PCX file, or uses the default if not found.
    Returns both a Pillow palette image and the transparency key color.
    """
    raw_palette: Optional[list] = None
    transparency_key_color: Tuple[int, int, int] = (0, 0, 0)

    try:
        if not colormap_path.is_file():
            raise FileNotFoundError("Colormap file not found, reverting to default palette.")
        
        colormap_img = Image.open(colormap_path)
        if colormap_img.mode != 'P':
            raise ValueError("Colormap image is not a paletted (P mode) image.")
            
        raw_palette = colormap_img.getpalette()
        print(f"INFO: Successfully loaded palette from {colormap_path}.")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"WARN: {e} Using fallback default Quake 2 palette.")
        raw_palette = DEFAULT_Q2_PALETTE_DATA

    # Ensure the palette data is exactly 768 bytes (256 colors * 3 channels)
    if not raw_palette or len(raw_palette) != 768:
        print(f"ERROR: Palette data is corrupted or incorrect size ({len(raw_palette)} bytes). Using hardcoded default.")
        raw_palette = DEFAULT_Q2_PALETTE_DATA
    
    # Create a palette image that Pillow's quantize method can use.
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(raw_palette)
    
    # The last color (index 255) is the transparency key.
    last_color_index = 255 * 3
    transparency_key_color = tuple(raw_palette[last_color_index : last_color_index + 3])
    
    return palette_img, transparency_key_color

def get_q2_flags(tga_path: Path) -> Tuple[int, int, int]:
    """
    Determines Quake 2 surface flags, content flags, and light value based on path.
    """
    path_str = tga_path.as_posix().lower()
    surfaceflags = 0
    contentflags = CONTENTS_SOLID # Default content flag
    lightvalue = 0
    
    # 1. Check for directory-based flags (normalized Quake 4 dirs)
    if "/q4_lights/" in path_str:
        surfaceflags |= SURF_LIGHT
        lightvalue = Q2_LIGHT_VALUE
    elif "/q4_glass/" in path_str:
        surfaceflags |= SURF_TRANS33
        contentflags = CONTENTS_WINDOW
    elif "/q4_fluids/" in path_str:
        surfaceflags |= SURF_WARP
        contentflags = CONTENTS_WATER
        
    return surfaceflags, contentflags, lightvalue

def create_wal(tga_path: Path, wal_path: Path, palette_img: Image.Image, transparency_color: Tuple[int, int, int], surfaceflags: int, contentflags: int, lightvalue: int):
    """
    Converts a TGA image to a Quake II WAL file, scaling it down by 50% and 
    writing the specified flags and light value.
    """
    # 1. Open the TGA image.
    img_original = Image.open(tga_path).convert("RGBA")

    # 2. Scale the image down by 50% using a high-quality filter.
    new_width = img_original.width // 2
    new_height = img_original.height // 2
    if new_width < 1 or new_height < 1:
        raise ValueError("Image is too small to be scaled down by 50%.")
    img_rgba = img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    width, height = img_rgba.size

    # 3. Process transparency using the dynamically loaded key color.
    img_rgb = Image.new("RGB", (width, height))
    pixels_rgba = img_rgba.load()
    pixels_rgb = img_rgb.load()
    
    ALPHA_THRESHOLD = 128

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels_rgba[x, y]
            # Use the color read from the palette file for transparent pixels.
            if a < ALPHA_THRESHOLD:
                pixels_rgb[x, y] = transparency_color
            else:
                pixels_rgb[x, y] = (r, g, b)

    # 4. Quantize the RGB image to the loaded Quake II palette.
    img_paletted = img_rgb.quantize(palette=palette_img, dither=Image.Dither.NONE)

    # 5. Generate mipmaps.
    mipmaps = [img_paletted]
    for i in range(3):
        mip_w, mip_h = mipmaps[-1].size
        if mip_w <= 1 or mip_h <= 1:
            break
        # The BOX filter is typical for mipmap generation to avoid artifacts
        mipmaps.append(mipmaps[-1].resize((mip_w // 2, mip_h // 2), Image.Resampling.BOX))

    # 6. Write the WAL file header and data.
    with open(wal_path, "wb") as f:
        name_bytes = wal_path.stem.encode('ascii')
        
        # Header (64 bytes)
        f.write(struct.pack("<32s", name_bytes)) # Name (32s)
        f.write(struct.pack("<II", width, height)) # Width, Height (2I)

        header_size = 68 # Start of Mip Offsets
        offsets = [0] * 4
        current_offset = header_size + (4 * 4) + 32 + (4 * 3) 

        for i in range(len(mipmaps)):
            offsets[i] = current_offset
            w, h = mipmaps[i].size
            current_offset += w * h
        for i in range(len(mipmaps), 4):
            offsets[i] = offsets[len(mipmaps) - 1] if mipmaps else 0

        # Mip Offsets (16 bytes)
        f.write(struct.pack("<4I", *offsets))

        # Next/Pad (32 bytes)
        f.write(struct.pack("<32s", b''))
        
        # Flags and Light (12 bytes)
        f.write(struct.pack("<I", surfaceflags)) # Surface Flags (I)
        f.write(struct.pack("<I", contentflags)) # Content Flags (I)
        f.write(struct.pack("<I", lightvalue))   # Light Value (I)
        
        # Write image data
        for mip in mipmaps:
            f.write(mip.tobytes())

def main():
    """
    Main function to find and convert all applicable .tga files.
    """
    # The palette loading is now handled with a fallback inside the function
    palette_img, transparency_color = load_palette_and_key_color(Path("pics/colormap.pcx"))
    print(f"Palette is ready. Transparency key color: {transparency_color}")

    root = Path(".")
    print(f"Scanning for applicable *.tga files in '{root.resolve()}' to convert (scaling to 50%)...")
    
    count = 0
    for tga_file in root.rglob("*.tga"):
        # 1. Skip check: Skip auxiliary maps (_add, _glow)
        if tga_file.stem.endswith("_add") or tga_file.stem.endswith("_glow"):
            # This check is simplified as _add and _glow files should never be in the base 'textures' folder structure 
            # for Quake 2, but we keep the logic to ensure auxiliary maps aren't converted.
            print(f"SKIP -> {tga_file.name}: Auxiliary map.")
            continue

        count += 1
        wal_file = tga_file.with_suffix(".wal")
        
        # 2. Determine Q2 flags based on file path
        surfaceflags, contentflags, lightvalue = get_q2_flags(tga_file)
        
        try:
            wal_file.parent.mkdir(parents=True, exist_ok=True)
            create_wal(tga_file, wal_file, palette_img, transparency_color, surfaceflags, contentflags, lightvalue)
            
            flag_info = []
            if surfaceflags & SURF_LIGHT: flag_info.append(f"LIGHT({lightvalue})")
            if surfaceflags & SURF_TRANS33: flag_info.append("TRANS33")
            if surfaceflags & SURF_WARP: flag_info.append("WARP")
            if contentflags == CONTENTS_WINDOW: flag_info.append("WINDOW")
            if contentflags == CONTENTS_WATER: flag_info.append("WATER")

            if flag_info:
                print(f"OK  -> {wal_file} [Flags: {', '.join(flag_info)}]")
            else:
                print(f"OK  -> {wal_file}")
                
        except Exception as e:
            print(f"FAIL: {tga_file}: {e}")
            
    if count == 0:
        print("No *.tga files found.")

if __name__ == "__main__":
    main()
